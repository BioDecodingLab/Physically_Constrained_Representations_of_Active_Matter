# Publication assets

The publication stage reads frozen numerical results and a provenance-verified visual export. It does not train models, select checkpoints, tune scientific settings, or generate manuscript prose. All three latent budgets contribute to the numerical comparison. The descriptive visual export uses d = 256, preserving the comparison budget used in the paper rather than selecting a budget from final-test reconstruction performance.

## Main figures

1. Dataset/reference-field examples from the fixed visual export.
2. Reconstruction comparison across methods and latent budgets.
3. Physical fidelity and admissibility comparisons.
4. Forecast dynamics, including linear, MLP, and persistence comparisons.
5. Descriptive validation latent PCA and probe summaries.

The renderer's filenames are the authoritative asset identifiers. Main figures are written in PNG, SVG, and PDF. Ten supplementary figures include detailed numerical, decoder, rollout, and all-seed latent diagnostics. `Figure_S5_decoder_diagnostics` distinguishes recorded extrema from the decoder arithmetic audit; values must retain their stated units and aggregation level.

## Tables and supplementary data

Three main tables describe the protocol and quantitative comparisons. Two supplementary tables and six numbered supplementary datasets provide detailed results. Additional numerical CSV files support plotted values. The module writes captions/legends alongside the table products. Table captions state methods, predictors, variants, aggregation, uncertainty, and the meaning of missing or non-finite scores where applicable.

## Videos

The renderer generates three supplementary videos, including `Supplementary_Video_1_reference_fields`. Reference-field axes and labels use the corrected layout. Video 1 uses the provenance-verified raw field export; it must not be replaced with a synthetic example. Video generation uses the FFmpeg executable supplied by `imageio-ffmpeg`.

## PCA interpretation

The main PCA view uses AE and R-PC at d = 256, seed 1729, with all 24 validation trajectories × 81 frames = 1944 snapshots per method. Supplementary panels cover seeds 1729, 2718, and 31415. Each representation space is analyzed independently. Axis labels report the appropriate explained-variance fractions; PCA does not identify causal coordinates, establish physical interpretability, or quantify forecasting performance. d = 512 can yield a different reconstruction score while d = 256 remains a valid, explicitly identified common visualization budget.

## Regeneration

```bash
python scripts/study.py visuals --device cuda:0
python scripts/study.py publication
```

`PUBLICATION_MANIFEST.json` records numerical input identity, dependency versions, generated-asset counts, and visual-input status. `SHA256SUMS.txt` provides a content inventory. The release source contains no generated scientific plots, duplicated figure exports, manuscript drafts, or deprecated rendering modules.
