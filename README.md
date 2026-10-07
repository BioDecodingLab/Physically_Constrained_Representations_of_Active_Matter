# Physical fidelity and forecasting in compressed active matter representations

The workflow compares unconstrained, physics-constrained, realizability-constrained, and combined autoencoders against proper orthogonal decomposition (POD), then evaluates latent forecasting with linear autoregression, multilayer perceptrons, and persistence baselines.

This repository contains data acquisition, physical measurement checks, training, representation analysis, validation diagnostics, protocol freezing, final test evaluation, and generation of the manuscript's figures, tables, supplementary data, supplementary figures, and videos. Manuscript prose is maintained independently; no module generates or edits manuscript text.

## Scientific protocol

The analysis uses the HDF5 `active_matter` dataset from [The Well](https://polymathic-ai.org/the_well/datasets/active_matter/) and its [official mirror](https://huggingface.co/datasets/polymathic-ai/active_matter). The release pins revision `dc3e9135a75b4e5a0d979086219fc97fe668f7a3`: all 82 file checksums match the study's recorded data hashes. This establishes matching file contents; it does not assert that this revision identifier was recorded during the original training runs.

| Quantity | Prescribed value |
|---|---|
| Train / validation / test trajectories | 175 / 24 / 26 |
| Spatial grid | 256 × 256, periodic domain 10 × 10 |
| Frames per trajectory | 81, sampled at Δt = 0.25 |
| Fields | Concentration, velocity, orientation tensor D, strain-rate tensor E |
| Latent dimensions | 128, 256, 512 |
| Training seeds | 1729, 2718, 31415 |
| Encoder channel widths | 32, 64, 128, 256 |
| Warm-up / branch epochs | 20 / 60 |
| Physics-weight ramp | 5 epochs |
| Batch / effective batch | 16 / 32 |
| Optimizer learning rate / weight decay | 10⁻⁴ / 10⁻⁸ |
| Forecast history / training horizon | 4 / 10 frames |
| Forecaster epochs / hidden width | 60 / 256 |
| Bootstrap samples / seed | 2000 / 1729 |

At d = 256, the first training seed screens physics weights `{0.01, 0.1, 1}` and realizability-aware physics weights `{0.001, 0.01, 0.1}` using validation criteria and a 1.05 reconstruction guard. The selected weights are retained for the remaining seeds. The d = 128 and d = 512 configurations prescribe the paper's transferred weights, 0.01 and 0.001. Selection is never recomputed from final test results. See [Scientific protocol](docs/SCIENTIFIC_PROTOCOL.md) for metrics, aggregation, and interpretation.

Improved reconstruction or admissibility does not establish improved forecasting. Preserve all baselines, missing/invalid scores, and negative forecasting results. The final evaluator retains its disclosure that absence of historical test exposure in earlier project stages is not certified.

## Installation

The supported reference setup is Linux, Python 3.12, and PyTorch 2.6.0. A CUDA-capable NVIDIA GPU is recommended for the full-resolution study. CPU execution is supported for software tests and small diagnostic fixtures. The pinned source data occupy 55,809,409,024 bytes (approximately 52 GiB); allocate additional space for checkpoints, latent caches, intermediate measurements, and publication outputs. A full run is a substantial training and analysis workload. No full-study runtime estimate is certified.

Create and activate an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the CUDA 12.4 build:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install --no-deps -e components/representation -e components/publication
python scripts/study.py check --device cuda:0
```

For CPU execution, replace the PyTorch installation and environment check:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e components/representation -e components/publication
python scripts/study.py check --device cpu
```

The wheel installation follows [PyTorch's version-specific instructions](https://docs.pytorch.org/get-started/previous-versions/). CUDA wheels require a compatible host NVIDIA driver. The two editable installs provide package metadata and console entry points; posthoc and final-evaluation modules run in isolated subprocesses because they share the package name `amreview`. Do not install or combine these two packages in one import namespace.

`requirements.txt` pins the direct numerical and publication dependencies used in release verification. `requirements-lock-cpu.txt` also pins their resolved CPU dependencies; use it in place of `requirements.txt` when reproducing that software-test environment. GPU and hardware differences may prevent bitwise agreement with original trained checkpoints. Run manifests record the actual runtime, dataset, configuration, source, and checkpoint identities.

## Complete reproduction

Run from the repository root:

```bash
python scripts/study.py all --work-dir "$PWD/artifacts" --device cuda:0 --workers 8 --threads 8
```

The runner executes:

1. Download and verify the pinned dataset.
2. Audit data, fit training transforms, and verify physical measurement operators.
3. Train representations at all three latent dimensions and select checkpoints using validation.
4. Cache latents, fit POD and representation probes, and train latent forecasters.
5. Evaluate validation and produce validation rollout diagnostics and experiment locks.
6. Run the complete posthoc validation analysis at every dimension.
7. Freeze the v0.4.0 analysis protocol and its source/checkpoint fingerprints.
8. Evaluate the frozen test protocol.
9. Export verified reference fields, reconstructions, and validation latent arrays.
10. Render all publication figures, tables, supplementary data, and videos.

Inspect the planned commands without running computations or downloading data:

```bash
python scripts/study.py all --work-dir /data/active-matter-reproduction --device cuda:0 --dry-run
```

Work products remain outside the source components. Use the same absolute work directory for every stage. `execution.jsonl` records commands and exit status. Scientific configuration and source verification run before every invocation. Completed training and analysis tasks use the original modules' compatibility checks. A matching final protocol lock is reused; it is never silently overwritten. Rerunning publication stages refreshes the named publication products. For an independent scientific rerun, use a new work directory.

## Run one module at a time

Use a consistent work directory; omit `--work-dir` to use `artifacts/`:

```bash
python scripts/study.py download
python scripts/study.py audit
python scripts/study.py representations
python scripts/study.py analysis
python scripts/study.py forecasters
python scripts/study.py validation
python scripts/study.py posthoc
python scripts/study.py freeze
python scripts/study.py test
python scripts/study.py visuals
python scripts/study.py publication
```

All computational stages accept `--device`, `--workers`, and `--threads`. For example:

```bash
python scripts/study.py representations --dimensions 256 --device cuda:0 --workers 4 --threads 4
```

Run d = 256 before the compression budgets when following the study sequence. `--dimensions` is available for training and validation stages. Freezing, final evaluation, and publication require all three budgets. `--no-videos` omits video rendering and therefore produces an incomplete publication asset set. See [Workflow reference](docs/WORKFLOW.md) for each stage's inputs, outputs, and underlying module command.

## Outputs

| Location under the work directory | Contents |
|---|---|
| `project/data/active_matter/data/` | Verified HDF5 splits, download receipt, verification receipt |
| `project/runs/realizable_d256/` | Weight screening, fitted representations, probes, forecasters, validation |
| `project/runs/compression_d128/`, `compression_d512/` | Prescribed compression-budget experiments |
| `posthoc/d128/`, `d256/`, `d512/` | Full validation diagnostics and task checksum markers |
| `final_test/protocol_lock.json` | Frozen analysis protocol |
| `final_test/d128/`, `d256/`, `d512/` | Final reconstruction, forecast, rollout, probe and group metrics |
| `visual_export/` | Read-only field, reconstruction, and latent exports tied to the final lock |
| `publication/figures/main/` | Five main figures in PDF, SVG, and PNG formats |
| `publication/figures/supplementary/` | Ten supplementary figures |
| `publication/tables/` | Three main tables, two supplementary tables, supplementary data and numerical CSVs |
| `publication/videos/` | Three supplementary MP4 videos |
| `publication/PUBLICATION_MANIFEST.json` | Rendering provenance and generated-asset counts |
| `publication/SHA256SUMS.txt` | Publication asset checksums |

Publication PCA uses d = 256 for the descriptive AE versus R-PC comparison: all 24 validation trajectories and 81 frames per trajectory (1944 snapshots per method and seed). The main view uses seed 1729; supplementary views include all three seeds. It is not a visualization of the best reconstruction budget. It is not used for model selection, quantitative inference, or evidence of forecast skill. See [Publication assets](docs/PUBLICATION_ASSETS.md).

## Tests and verification

```bash
python scripts/study.py tests --device cpu --workers 0 --threads 1
```

The runner launches separate test processes for representation training, posthoc analysis, final evaluation, publication rendering, and orchestration. Synthetic fixtures exercise numerical operators, training, provenance, protocol freezing, final evaluation, and rendering. Synthetic outputs are software tests and cannot substitute for the real-data study. The test suite does not download or train on the full dataset.

GitHub Actions runs the CPU test suite. [Release verification](docs/VALIDATION.md) identifies which checks were actually performed for this release and distinguishes them from the full real-data GPU workflow.

## Repository structure

```text
components/
  representation/       Training and representation analysis v0.2.0
  posthoc/              Posthoc validation analysis v0.3.0
  final_evaluation/     Frozen final evaluator v0.4.0
  publication/          Publication rendering and verified visual exports
configs/paper/          Complete prescribed scientific configurations
scripts/               Data acquisition, environment check, stage runner
provenance/            Dataset catalog, paper hashes, source fingerprints
tests/                 Orchestration and release-integrity checks
docs/                  Protocol, workflow, asset, and validation documentation
```

The reference scientific modules and their vendor copies retain identical source bytes to preserve their established compatibility fingerprints. Publication metadata is cleaned for public use in renderer v1.4.1. No scientific estimator or final-evaluation logic is changed by this repository assembly.

## Attribution and reuse

See [CITATION.cff](CITATION.cff) for repository authorship and [references.bib](references.bib) for the active-fluid simulation paper. The upstream dataset is distributed under CC BY 4.0; its license and attribution apply independently of the code. Dataset files and trained checkpoints are obtained or generated by the workflow and are not bundled in this source release. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
