# Scientific protocol

## Data and splits

Use the recorded HDF5 files and upstream train/validation/test partition without resampling. The data contain 225 trajectories, with 175 used for training, 24 for validation, and 26 for final evaluation. Preserve trajectory identity, parameter tuples, frame order, spatial axes, periodic lengths, and frame spacing. Data acquisition verifies file size and SHA256; auditing additionally checks metadata, trajectory counts, numerical operators, and train/validation physical measurements.

The full data manifest includes hashes and metadata for every split. The original audit can inspect test-file metadata and shared initial-condition fingerprints; this is distinct from selecting models using held-out test performance. The v0.4.0 disclosure explicitly does not certify absence of historical test exposure. Publication should retain this qualification.

## Representation training

The five methods are AE, PC, R-AE, R-PC, and POD. Neural representations use latent dimensions 128, 256, or 512 and the exact parameters in `configs/paper/`. Each seed uses a shared warm-up checkpoint for its unconstrained branches and their realizable counterparts. Physics terms use the specified spectral derivative operators and training-derived scales. The realizability parameterization and explicit projection baselines are distinct interventions and must retain separate labels.

At d = 256, screen PC weights 0.01, 0.1, and 1 and R-PC weights 0.001, 0.01, and 0.1 on the first seed. An eligible physics candidate must pass the prescribed observable-score and reconstruction guard. Minimize validation observable error among eligible candidates, resolving ties by reconstruction error and then weight. If no candidate passes the guard, the pipeline stops; do not relax the threshold to force completion. The selected weight is used for confirmation seeds. The d = 128 and d = 512 budgets use the paper-prescribed single candidate 0.01 for PC and 0.001 for R-PC. New training can yield different selected checkpoints; recorded checkpoint hashes identify an exact fitted realization, not a guarantee of identical retraining across hardware.

Training statistics, POD bases, latent normalizations, parameter normalizations, and probe fits are learned from training data. Hyperparameter and checkpoint selection use validation. Final evaluation does not fit new models or adjust weights.

## Forecasting and diagnostics

Forecasters receive four latent history frames and parameter covariates. The training horizon is ten frames. Linear autoregression is the primary latent predictor; MLP and persistence remain comparison baselines. The posthoc and final analyzers retain complete remaining-horizon rollouts rather than stopping at the training horizon. Spatial reconstruction diagnostics use every tenth frame, while reconstruction metrics use every frame.

Keep the distinction between raw/legacy decoder outputs, realizable decoder outputs, and explicit projected variants. Reconstruction fidelity and physical observables are measured on decoded physical fields. A realizability guarantee applies to the prescribed constraints; it does not establish agreement with true trajectories or dynamic closure.

## Scores, missingness, and uncertainty

The implementation defines reconstruction error (`R_rec`), physical residual error (`F`), and observable error (`Q`) in the frozen numerical modules. Their denominators, field weights, concentration thresholds, and finite-value rules are part of the protocol. Do not replace invalid observable values with zero, clip extreme values to improve summaries, or omit failed methods from comparisons. Positive-domain diagnostics and the legacy observable score are reported separately.

The final analyzer reports parameter-group metrics and aggregates groups with the prescribed weighting. Seed standard deviations represent variation across the three fitted models. Paired parameter-group bootstrap intervals are conditional on those fitted seeds, not calibrated predictive uncertainty and not an estimate of all possible training randomness. The publication renderer verifies agreement between supplied summary metrics and their group-level aggregation.

## Protocol freezing

Complete all three full posthoc validation analyses before invoking `freeze`. The final lock records source fingerprints, dataset identity, source-run identities, checkpoint and latent-cache hashes, validation analyses, probe definitions, and the final analysis settings. The test stage verifies these records and emits checksummed task markers. A changed source, scientific configuration, checkpoint, or incompatible task record must fail verification rather than being silently accepted.

`components/final_evaluation/amreview/` and its `vendor/active_matter_repr/` are unchanged copies of FinalEvaluation v0.4.0. The source manifest provides per-file SHA256 checks. The scientific source fingerprint for representation v0.2.0 is `9e85ef699942c8679b712c0af82ab7d8bc3bc4c50897758eb3dbd8809cf5475b`.

## Interpretation

The study evaluates how compressed representations preserve reconstruction and physical fidelity, and whether these properties transfer to forecasting. A method can improve reconstruction or physical admissibility while failing to improve long-horizon dynamics. Preserve this separation when interpreting all figures and tables. PCA is descriptive visualization of latent validation snapshots, with independent fits across representation spaces; apparent distances across separately fitted spaces are not comparable physical distances.
