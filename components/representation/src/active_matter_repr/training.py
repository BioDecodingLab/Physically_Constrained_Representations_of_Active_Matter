"""Epoch-boundary resume and sample-correct gradient accumulation."""
from pathlib import Path
import torch
from tqdm.auto import tqdm
from .artifacts import save_checkpoint, load_checkpoint, rng_state, restore_rng, write_json, digest
from .data import loader
from .physics import residuals
from .transforms import reconstruction


def optimize_epoch(model, batches, optimizer, loss_function, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pending, total, n = 0, 0., 0
    length = len(batches)
    for i, batch in enumerate(tqdm(batches, desc="train", leave=False)):
        losses = loss_function(batch)
        if losses.ndim != 1 or not torch.isfinite(losses).all():
            raise FloatingPointError("Loss must be a finite per-sample vector")
        losses.sum().backward()
        count = len(losses)
        pending += count
        total += float(losses.detach().sum())
        n += count
        if pending >= cfg["effective_batch"] or i + 1 == length:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
    if not n:
        raise ValueError("Empty training loader")
    return total / n


def train_ae(model, transform, train, valid, cfg, directory, seed, weight, epochs,
             evaluate, warmup_path=None, warmup=False, physics_terms=(0, 1, 2, 3)):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    physics_terms = tuple(physics_terms)
    if not physics_terms or any(i not in range(4) for i in physics_terms):
        raise ValueError("physics_terms must select one or more residual families")
    identity = digest({"config": cfg["id"], "run": cfg.get("run_id"), "seed": seed,
                       "weight": weight, "warmup": warmup, "realizable": model.realizable,
                       "physics_terms": physics_terms})
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    start, best, history = 0, float("inf"), []
    last = directory / "last.pt"
    if last.exists():
        state = load_checkpoint(last)
        if state["identity"] != identity:
            raise ValueError("Incompatible resume configuration")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start, best, history = state["epoch"], state["best"], state["history"]
        # Recover best atomically if interruption occurred between last and best writes.
        best_path = directory / "best.pt"
        if state.get("is_best") and (not best_path.exists() or load_checkpoint(best_path)["epoch"] != state["epoch"]):
            save_checkpoint(directory / "best.pt", state)
    elif warmup_path is not None:
        state = load_checkpoint(warmup_path)
        if state["config"].get("run_id") != cfg.get("run_id") or state["transform"] != transform.as_dict():
            raise ValueError("Warm-up belongs to different data or normalization")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    for epoch in range(start, epochs):
        multiplier = 0 if warmup else weight * min(1., (epoch + 1) / cfg["ramp_epochs"])
        def loss(batch):
            x = batch["x"].to(cfg["device"])
            pred = transform.inverse(model(transform.normalize(x)))
            r = reconstruction(pred, x, transform)
            if multiplier:
                physical = residuals(pred, transform.scales, train.rows[0]["spacing"], cfg["derivative"])
                r = r + multiplier * physical[..., list(physics_terms)].mean(-1)
            return r
        train_loss = optimize_epoch(model, loader(train, cfg, True, seed + epoch), opt, loss, cfg)
        score = evaluate(model)["summary"]
        eligible = warmup or epoch + 1 > cfg["ramp_epochs"]
        is_best = eligible and score["R_rec"] < best
        if is_best:
            best = score["R_rec"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "weight": multiplier, "eligible": eligible, **score})
        state = {"identity": identity, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "epoch": epoch + 1, "best": best, "history": history, "rng": rng_state(), "is_best": is_best,
                 "config": cfg, "transform": transform.as_dict(), "warmup_path": str(warmup_path) if warmup_path else None}
        save_checkpoint(last, state)
        if is_best:
            save_checkpoint(directory / "best.pt", state)
        write_json(directory / "history.json", history)
        print(f"{directory.name}: epoch {epoch+1}/{epochs}, validation R_rec={score['R_rec']:.6g}", flush=True)
    # A warm-up always branches from its last state, not its best validation epoch.
    chosen = last if warmup else directory / "best.pt"
    if not chosen.exists():
        raise RuntimeError("No eligible checkpoint")
    model.load_state_dict(load_checkpoint(chosen)["model"])
    return chosen
