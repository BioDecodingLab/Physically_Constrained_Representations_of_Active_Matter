"""Canonical publication tables and unique source-data files."""
from pathlib import Path
import pandas as pd
from .results import key_results,dynamics_key_table
from .statistics import uncertainty_tables
from .presentation import endpoint_profile

TABLE_LEGENDS = {
    'main/Table_1_protocol': (
        'Table 1. Data partition, representation families, forecast endpoints, and uncertainty protocol. '
        'AE denotes unconstrained autoencoding; PC-AE adds physical constraints; R-AE uses the realizable '
        'decoder; R-PC-AE combines realizability and physical constraints; POD denotes weighted proper '
        'orthogonal decomposition. Short forecasts use every complete origin after four observed frames; '
        'long rollouts start from the first eligible origin. Bootstrap intervals are conditional on the '
        'three fitted seeds and resample matched parameter groups, not independent frames.'),
    'main/Table_2_reconstruction_key_results': (
        'Table 2. AE and R-PC-AE reconstruction results at latent dimensions 128, 256, and 512. '
        'Q is the prescribed observable error and R_rec is the field reconstruction error; lower values '
        'indicate smaller errors. Each method value is the mean of the three fitted-seed summaries, '
        'with the frozen parameter-group aggregation retained within each summary. Q_reduction_percent '
        'equals 100 × (AE_Q − R-PC-AE_Q) / AE_Q; positive values indicate reduced observable error. '
        'R_rec_relative_change_percent equals 100 × (R-PC-AE_R_rec − AE_R_rec) / AE_R_rec; positive values '
        'indicate increased reconstruction error. The reported admissibility fraction is the maximum '
        'detected fraction across the listed concentration, orientation-eigenvalue, and order constraints '
        'and fitted-seed summaries; zero is a numerical diagnostic, not evidence of dynamic accuracy. '
        'Undefined values remain missing.'),
    'main/Table_3_dynamics_key_results': (
        'Table 3. Selected short-forecast and long-rollout comparisons at d = 256. R-PC-AE uses the '
        'direct-corrected realizable decoder; POD uses the explicit projected variant. Linear latent '
        'autoregression is compared with physical persistence. F and Q retain the frozen field and '
        'observable error definitions and endpoint aggregation; lower values indicate smaller errors. '
        'Values are means across the available fitted-seed summaries, with strict missing-value rules; '
        'n_seed_rows counts recorded seed rows and does not count independent test trajectories. '
        'POD and persistence rows repeated across seed labels do not represent independent stochastic '
        'model fits. Improvements in admissibility do not imply improvements in forecast error.'),
    'supplementary/Table_S1_reconstruction_absolute_contrasts': (
        'Supplementary Table 1. Frozen absolute reconstruction-Q contrasts comparing R-PC-AE '
        '(rpc/direct_corrected) with AE (ae/legacy). Contrast direction, estimates, and interval endpoints '
        'are retained exactly from the final evaluator. The paired bootstrap uses matched parameter '
        'groups, 2000 draws, and seed 1729; intervals are conditional on the fitted model seeds. '
        'The column difference_b_minus_a is AE minus R-PC-AE; positive values indicate reduced observable error for R-PC-AE. Missing intervals or '
        'scores remain missing and must not be interpreted as zero.'),
    'supplementary/Table_S2_reconstruction_percentage_contrasts': (
        'Supplementary Table 2. Publication-derived reconstruction percentage contrasts for the '
        'prescribed method and decoder variants. Each percentage is computed from the paired '
        'parameter-group means using the stated reference method; it is a ratio of means, not a mean '
        'of per-group percentages. The bootstrap resamples matched parameter groups with replacement '
        'and is conditional on the fitted seeds. Positive reduction percentages favor R-PC-AE relative to the reference '
        'method. Invalid or zero denominators produce missing results. These derived contrasts do '
        'not alter the frozen final-evaluation scores.'),
}


def _write(df,base):
    base.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(base.with_suffix('.csv'),index=False)
    if len(df)<=100:base.with_suffix('.md').write_text(df.to_markdown(index=False)+'\n')


def generate_tables(data,out):
    out=Path(out)
    protocol=pd.DataFrame([
        ['Dataset','The Well active_matter','175 train; 24 validation; 26 test trajectories; 81 frames; 256×256; 11 channels'],
        ['Representations','AE, PC-AE, R-AE, R-PC-AE, weighted POD','d=128, 256, 512; three fitted neural seeds; deterministic POD'],
        ['Primary transition','Linear autoregression','4 observed frames; recursive prediction; MLP is a secondary comparison'],
        ['Short forecast','All 68 complete origins per trajectory','10 recursive leads; no teacher forcing'],
        ['Long rollout','First origin per trajectory','77 recursive leads after four observed frames'],
        ['Uncertainty','Matched parameter-group bootstrap','21 groups; 2,000 draws; seed 1729; conditional on fitted models'],
    ],columns=['Item','Protocol','Reporting note'])
    _write(protocol,out/'main/Table_1_protocol')
    _write(key_results(data),out/'main/Table_2_reconstruction_key_results')
    _write(dynamics_key_table(data,256),out/'main/Table_3_dynamics_key_results')
    paired=data.paired
    q=paired[(paired.kind=='reconstruction')&(paired.metric=='Q')&(paired.a=='rpc/direct_corrected')&(paired.b=='ae/legacy')].copy()
    _write(q,out/'supplementary/Table_S1_reconstruction_absolute_contrasts')
    seeds,derived=uncertainty_tables(data)
    if not derived.empty:
        _write(derived[derived.kind=='reconstruction'],out/'supplementary/Table_S2_reconstruction_percentage_contrasts')
    for i,(name,df) in enumerate([('full_summary',data.summary),('paired_comparisons',data.paired),('probe_summary',data.probes),('coverage',data.coverage),('spatial_summary',data.spatial),('decoder_diagnostics',data.decoder)],1):
        _write(df,out/f'source_data/Supplementary_Data_{i}_{name}')
    for name,df in [('endpoint_profile',endpoint_profile(data)),('seed_uncertainty',seeds),('publication_group_contrasts',derived),('frozen_group_metrics',data.group_metrics)]:
        if not df.empty:_write(df,out/'source_data'/name)
    for name, caption in TABLE_LEGENDS.items():
        path = out / name
        if path.with_suffix('.csv').exists():
            path.with_suffix('.caption.md').write_text(caption + '\n')
    mapping=[
        ['Figure 1','visuals.npz dataset_views; study schematic','fixed median test example; d=256'],
        ['Figure 2','Supplementary Data 1; frozen_group_metrics; Table S2','fitted seed SD; matched group ratio intervals'],
        ['Figure 3','Supplementary Data 1 and 2','designated variants; frozen paired group intervals'],
        ['Figure 4','frozen d*/lead_summary.csv; publication_group_contrasts','fitted seed SD; persistence contrast'],
        ['Figure 5','visuals.npz PCA; Supplementary Data 3','24 validation trajectories; d=256; seed 1729; descriptive'],
        ['Supplementary Figure 1','visuals.npz dataset_views','three design-selected reference examples'],
        ['Supplementary Figure 2','visuals.npz gallery arrays','reference-scale reconstructions; saturation annotated'],
        ['Supplementary Figure 3','visuals.npz gallery arrays','full-range signed error maps'],
        ['Supplementary Figure 4','Supplementary Data 5','reconstruction spatial summaries'],
        ['Supplementary Figure 5','Supplementary Data 6','task/predictor separated; recorded run maxima'],
        ['Supplementary Figure 6','frozen d*/lead_summary.csv','linear and MLP; all budgets and horizons'],
        ['Supplementary Figure 7','endpoint_profile; frozen_group_metrics','paired ratios vs AE; separate endpoints'],
        ['Supplementary Figure 8','Supplementary Data 1 and 4','undefined-Q accounting; not independent samples'],
        ['Supplementary Figure 9','Supplementary Data 3','all three probe feature sets'],
        ['Supplementary Figure 10','visuals.npz PCA','24 validation trajectories; all three seeds; d=256'],
        ['Table 1','frozen protocol configuration','dataset and evaluation design'],
        ['Table 2','Supplementary Data 1','three-seed reconstruction means'],
        ['Table 3','Supplementary Data 1','forecast means at d=256'],
        ['Supplementary Table 1','Supplementary Data 2','frozen absolute reconstruction contrasts'],
        ['Supplementary Table 2','publication_group_contrasts','publication-only ratio of means'],
        ['Supplementary Video 1','visuals.npz reference_movie','observed fields; 81 stored frames; fixed scales'],
        ['Supplementary Video 2','frozen d256/lead_summary.csv','77-lead aggregate F; not predicted field video'],
        ['Supplementary Video 3','frozen d256/lead_summary.csv','10-lead aggregate Q; not predicted field video'],
    ]
    pd.DataFrame(mapping,columns=['Display item','Input','Reporting']).to_csv(out.parent/'FIGURE_DATA_MAP.csv',index=False)
