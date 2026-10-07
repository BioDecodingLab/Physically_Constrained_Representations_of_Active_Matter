# Workflow reference

Execute commands from the repository root after installation. The standard work directory is `artifacts/`; a custom absolute `--work-dir` is supported. Every stage checks reference source integrity and the paper configuration identities. Individual stages require the preceding artifacts; they do not implicitly train missing models.

| Stage | Required input | Main output | Scientific module |
|---|---|---|---|
| `download` | Pinned dataset catalog | HDF5 splits and hash receipt | `scripts/download_data.py` |
| `audit` | Verified splits, paper configuration | Data manifest, transform, observable scales, passed audit | `active_matter_repr.cli audit-data` |
| `representations` | Passed audit | Neural checkpoints, selected weights, POD model | `active_matter_repr.cli train-representations` |
| `analysis` | Selected representations | Latent caches, representation probes | `active_matter_repr.cli analyze-representations` |
| `forecasters` | Latent caches and training statistics | Linear and MLP forecasters | `active_matter_repr.cli train-forecasters` |
| `validation` | Representations and forecasters | Validation scores, rollout diagnostics, experiment lock | `evaluate`, `diagnose-rollouts`, `report`, `lock` |
| `posthoc` | Completed source runs | Full validation reconstruction/rollouts/probes with task markers | `amreview.run` v0.3.0 |
| `freeze` | All three source runs and full posthoc analyses | `final_test/protocol_lock.json` | `amreview.final freeze` v0.4.0 |
| `test` | Frozen protocol lock | Frozen test scores, group records, analysis markers | `amreview.final evaluate` v0.4.0 |
| `visuals` | Lock, source checkpoints and data | Verified field and latent visualization arrays | `ampub.export_visuals` |
| `publication` | Final test results and visual export | Figures, tables, supplementary data and videos | `ampub` |

## Runtime configuration

The runner materializes `artifacts/configs/d128.yaml`, `d256.yaml`, and `d512.yaml` from the prescribed configurations, substituting only execution paths, device, workers, and thread count. `auto_download` is disabled within training: the acquisition stage must first verify the complete pinned dataset. All scientific values remain identical to the paper configurations.

Scientific configuration identities are:

- d = 128: `b9b90989e37e3845ab9873a34797a1fbc2759bf692e81b50efc153efff1cdf3d`
- d = 256: `34cc76f72fd2653d44c43574487be2fac0bd8b9e66daf5b1f22a3da859dc8f4b`
- d = 512: `7ce83570e14072539928cfe682bdfe907ebcd3d7d0cfbb8037d89da930e0a89b`

The runner sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the Agg plotting backend, and BLAS/OpenMP thread counts. The scientific pipeline enables deterministic PyTorch algorithms and disables cuDNN benchmarking. These settings support reproducibility but do not guarantee identical floating-point arithmetic across devices or dependency stacks.

## Direct module access

The orchestrator is the recommended public entry point. To inspect a scientific module directly, set the appropriate isolated import path:

```bash
PYTHONPATH=components/representation/src python -m active_matter_repr.cli --help
PYTHONPATH=components/posthoc:components/posthoc/vendor python -m amreview.run --help
PYTHONPATH=components/final_evaluation:components/final_evaluation/vendor python -m amreview.final --help
PYTHONPATH=components/publication python -m ampub --help
```

A d = 256 training invocation after the runner has materialized its configuration is:

```bash
PYTHONPATH=components/representation/src python -m active_matter_repr.cli train-representations --config artifacts/configs/d256.yaml
```

Do not invoke the older representation module's `evaluate --split test` for the paper's final evaluation. The `study.py test` entry point invokes the unchanged v0.4.0 evaluator against its frozen lock.

## Resumption and independent reruns

Data transfers resume partial files associated with the pinned revision. Completed files are rehashed; mismatches are fatal. Source runs and posthoc tasks validate configuration, data, code, and checkpoint compatibility. A resumed task with incompatible fingerprints raises an error. Freeze creates a new destination; the orchestrator only reuses an existing lock whose identity and workspace paths agree. Final evaluation verifies lock compatibility and task checksums before resuming.

A process failure stops the end-to-end runner immediately and records a failed command in `execution.jsonl`. Correct the execution problem and rerun the failed stage, or rerun `all` with the same work directory. Do not edit protected configuration values or lower physical gates merely to bypass a failure. New scientific experiments require a new work directory and a clearly identified protocol distinct from the paper release.

## Containers

`Dockerfile` builds the same source package with either CPU or CUDA 12.4 wheels. For example:

```bash
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t active-matter:cpu .
docker run --rm -v "$PWD/artifacts:/data" active-matter:cpu tests --work-dir /data --device cpu --workers 0 --threads 1
```

For CUDA, use the default build argument and NVIDIA Container Toolkit:

```bash
docker build -t active-matter:cuda .
docker run --rm --gpus all -v "$PWD/artifacts:/data" active-matter:cuda all --work-dir /data --device cuda:0
```

Container builds are convenience instructions; release verification reports separately whether a container build was exercised.
