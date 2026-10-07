"""Command-line entry points; errors return a nonzero exit status."""
import argparse
from .config import load_config
from . import pipeline
from .reporting import report
from .synthetic import create_fixture
from .profiling import profile


def main():
    parser = argparse.ArgumentParser(description="Controlled active-matter representations")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-fixture", help="Create explicitly synthetic HDF5 test data")
    make.add_argument("--output", required=True)
    commands = ("audit-data", "train-representations", "analyze-representations", "train-forecasters", "evaluate", "lock", "report", "pilot", "profile", "diagnose-rollouts")
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "train-representations":
            p.add_argument("--pilot-only", action="store_true", help="Only first configured seed; later continue without this flag")
        if name == "profile":
            p.add_argument("--batch-size", type=int, default=1)
        if name in ("evaluate", "report", "diagnose-rollouts"):
            p.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    if args.command == "make-fixture":
        print(create_fixture(args.output))
        return
    cfg = load_config(args.config)
    if args.command == "profile":
        profile(cfg, args.batch_size)
        return
    if args.command == "pilot":
        pipeline.audit_data(cfg)
        pipeline.train_representations(cfg, pilot_only=True)
        for stage in (pipeline.analyze_representations, pipeline.train_forecasters, pipeline.evaluate, pipeline.diagnose_rollouts, report):
            stage(cfg)
        return
    if args.command == "report":
        report(cfg, args.split)
    elif args.command == "evaluate":
        pipeline.evaluate(cfg, args.split)
    elif args.command == "diagnose-rollouts":
        pipeline.diagnose_rollouts(cfg, args.split)
    elif args.command == "lock":
        pipeline.lock_experiment(cfg)
    elif args.command == "train-representations":
        pipeline.train_representations(cfg, args.pilot_only)
    else:
        getattr(pipeline, args.command.replace("-", "_"))(cfg)


if __name__ == "__main__":
    main()
