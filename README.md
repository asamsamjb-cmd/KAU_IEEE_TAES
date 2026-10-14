# KAU_IEEE_TAES — reproducibility package

Code, benchmark manifests, figure sources and per-run summary results for

> **Temporal Threat Detection in Satellite Networks Using Semantic Observable Representations**
> IEEE Transactions on Aerospace and Electronic Systems, manuscript TAES-2026-2877 (revised version).

Every number in the revised manuscript and its supplementary material was produced on one benchmark build,
`shared_nominal_v2` (dataset id `478f782c…`; the full id, the SHA-256 of the six generator outputs and the
episode-level split manifest are in `data/manifests/`), with the scripts in this repository.
Intermediate benchmark builds used earlier in the revision are not part of this package.

Headline result (macro-F1 on the common L=10 test endpoints, 38,078 endpoints, ten head seeds):
0.437 ± 0.003 at L=1 → 0.847 ± 0.001 at L=10 (DistilBERT–TST); evidence-availability reference 0.870;
rule baseline 0.344 (L=1) / 0.656 (L=10).

## 1. Contents

| Path | Content |
|---|---|
| `*.py` (repository root) | Generators, pipeline, experiment, statistics and figure scripts, kept under the file names used during the revision (the numeric prefix is the step number used below). All scripts are run from the repository root. |
| `shared_nominal.py`, `shared_context_v4.py` | Shared benign substrate and observation-context modules imported by the six generators. |
| `data/manifests/` | Build manifest, split manifest, benchmark statistics and audit reports of the released benchmark (files ≤ 2 MB). |
| `results/` | Summary outputs of every run on `shared_nominal_v2`: metrics, confusion matrices, bootstrap summaries, gate reports, run manifests and `sources.txt` files (files ≤ 2 MB). The folder layout is the one of `outputs/shared_nominal_v2/` on the machine that produced the results. |
| `graph/` | Figure sources: rendered PDF/PNG/SVG, the plotted values (`*_values.csv`) and the input provenance (`*_sources.txt`) of each manuscript figure. |
| `requirements.txt` | Exact package versions of the environment used (`pip freeze`). |

Release assets (attached to the GitHub Release of the tagged commit; not tracked in git):

| Asset | Content |
|---|---|
| `shared_nominal_v2_benchmark.zip` | `dataset/shared_nominal_v2/`: unified record file(s), episode-level train/validation/test splits, audit outputs, structural-missingness test sets. |
| `benchmark_source_csv.zip` | The six generator output files (one per scenario; raw and observable columns) consumed by `7.build_unified_dataset.py`. |
| `nos3_repeatability.zip` | Step 09 repeatability campaign package from the NOS3 software-in-the-loop environment (folder `09_nos3_repeatability_v1`): raw captures of the ten runs (five ordered normal/replay pairs, `runs/`), the per-pair mapping-audit outputs with the 22-field records (`pair_audits/`), the NOS3-side collector and mapping-audit scripts (`baseline/07_…`, `baseline/08_…`, `09_nos3_repeatability.py`), and `protocol.json` / `campaign_manifest.json` / `journal.json`. This package is the input of `nos3_detector_evaluation.py`. |
| `distilbert_window_baseline_model.zip` | Fine-tuned DistilBERT encoder of Step 8 (optional; regenerable). |

Not released because they are large and fully regenerable from the code, seeds and data above:
per-window embeddings, sequence caches, prediction files and model checkpoints.

## 2. Environment

- Windows, PowerShell, Python 3.14.5 in a virtual environment; scripts are started as `py "<script>.py"` from the repository root
- NVIDIA GeForce RTX 3060, CUDA 12.6
- torch 2.12.0+cu126, transformers 5.10.2, scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6 (full list: `requirements.txt`)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Scripts write to `outputs/` (and the generators to `dataset/`) relative to the repository root; both folders are created on first use.
To evaluate on the released benchmark without regenerating it, unzip `shared_nominal_v2_benchmark.zip` to `dataset/shared_nominal_v2/`
and `benchmark_source_csv.zip` to `dataset/`.

## 3. Benchmark generation and audit

| Step | Script | Output |
|---|---|---|
| 1–6 | `1.normal_observable_maker_revised.py` … `6.GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE_observable_maker_revised.py` | One CSV per scenario. All six generators draw the benign windows from `shared_nominal.py` (per-scenario disjoint random streams, same nominal law) and apply the scenario-specific effects on a copy of the benign window; the paired benign window is kept for the evidence audit. |
| 7 | `7.build_unified_dataset.py` | Unified record file, episode-level splits in `dataset/shared_nominal_v2/splits/`, build manifest with the SHA-256 of the six inputs and the derived `dataset_id`. |
| — | `benchmark_audit.py --splits dataset/shared_nominal_v2/splits` | Split-integrity, duplication, context-only and single-window probes. |
| — | `fingerprint_audit.py` | Evidence-free-window fingerprint audit (bag-of-tokens probe), per-class evidence availability and the evidence-availability reference; re-run with `--l10-predictions` after Step 11. |

Scripts that read benchmark outputs check the recorded `dataset_id` and stop if their inputs come from a different build.
The benchmark has 300,000 records in 5,016 episodes (train/validation/test 3,511/753/752 episodes); the common
L=10 evaluation population is 38,078 test endpoints.

## 4. Main pipeline (run in this order)

| Step | Script | What it does |
|---|---|---|
| 8 | `8.train_distilbert_window_baseline.py` | Fine-tunes DistilBERT on single serialized windows (L=1 baseline). |
| 9 | `9.extract_distilbert_embeddings.py` | Frozen per-window embeddings from the fine-tuned encoder. |
| 10 | `10.build_tst_sequences.py` | Episode-bounded, stride-1 window sequences for L ∈ {1, 2, 3, 5, 10} and the common endpoint population. |
| 11 | `11.train_tst_classifier.py` | Time-Series Transformer aggregation head per L (main seed) plus test-time variants (history zeroed / last window repeated / reversed / non-overlapping). |
| 12 | `12.multiseed_tst_runner.py` | Step 11 for the ten head seeds 20260610–20260619. |
| 13 | `13.pooling_mlp_temporal_aggregation_baseline.py` | Mean / Max / Mean+Max pooling + MLP aggregation heads, ten seeds (output folder keeps its historical name `39_pooling_mlp_finetuned_main`). |
| 14 | `14.lstm_finetuned_distilbert_l10_baseline.py` | LSTM aggregation head at L=10, ten seeds. |
| 15 | `15.paired_episode_bootstrap.py` | Class-stratified paired episode bootstrap (20,000 resamples) for the aggregator contrasts and the ten-seed mean confusion matrix. |
| 16 | `16.observable_rule_baseline.py` | Semantic-field rule detectors on the same 22-field records (Rule-L1, Rule-L10; thresholds set on validation only). |
| 17 | `17.rule_paired_episode_bootstrap.py` | Paired bootstrap for TST vs. rule contrasts. |

## 5. Experiment scripts by manuscript section

Section numbers refer to the revised manuscript; the supplementary material is abbreviated S.

| Manuscript part | Steps / scripts | Summary results |
|---|---|---|
| VI-A Overall performance, conventional baselines, error structure | 8, 11–17, `59.raw_semantic_representation_baseline.py` (one-hot 22-field HGB/MLP rows), `70.rf_statistical_baselines.py` (RF threshold / CUSUM reference) | `results/tst_training_finetuned_main`, `results/tst_training_finetuned_multiseed`, `results/39_pooling_mlp_finetuned_main`, `results/lstm_finetuned_distilbert_l10`, `results/15_paired_episode_bootstrap`, `results/16_observable_rule_baseline`, `results/17_rule_paired_episode_bootstrap`, `results/59_raw_semantic`, `results/70_rf_statistical_baselines` |
| VI-B Sources of the cross-window gain | `62.temporal_information_control.py` (paired heads: actual history vs. repeated current window), `63.temporal_evidence_analysis.py` (evidence location inside the L=10 context), `71.cross_window_coverage_control.py` (same number of observation slots, different masking policies), 12 (context-length sweep and test-time variants), `72.evidence_sparsity_sweep.py` (event-retention sweep of the generator) | `results/62_temporal_information_control`, `results/63_temporal_evidence_analysis`, `results/71_cross_window_coverage`, `results/72_evidence_sparsity` |
| VI-B / S Twenty-window extension (matched cohort) | `27.l20_tst_pilot.py`, `28.l20_evidence_analysis.py`, `29.matched_l10_tst_control.py`, `30.matched_l10_l20_meanmax.py`, `31.l20_population_lsweep.py`, `32.l20_groupB_breakdown.py`, `33.evidence_sparsity_L20.py`, `34.matched_2x2_bootstrap.py` (2×2 TST/Mean+Max × L10/L20 paired bootstrap, 3 seeds) | `results/27_L20_TST_VALIDATION_PILOT` … `results/34_matched_2x2_bootstrap` |
| VI-C Aggregators and encoders | 13, 14, `47.missing8_aggregation_evaluation.py` (EWMA, majority voting, probability averaging and the learned heads under identical masks), `52.temporal_statistics_baseline.py`, `42.encoder_comparison_audit.py`, `43.encoder_comparison_trainval_pilot.py`, `44.random_init_matched_pilot.py`, `45.encoder_comparison_formal_trainval.py` (categorical-embedding MLP, field-token Transformer, DistilBERT random-init, DistilBERT pretrained; 5 seeds), `46.encoder_comparison_frozen_test.py` | `results/47_missing8_aggregation`, `results/52_temporal_statistics`, `results/42_…` – `results/46_encoder_comparison_frozen_test` |
| VI-D Observable representation and field evidence | `51.field_group_train_ablation.py` (train-time field-group ablation, 5 seeds), `53.field_group_removal_evaluation.py` (test-time group removal), 59 (Raw / RawScoped / Semantic22 with HGB), `61.representation_shift_evaluation.py`, `67.threshold_recalibration_shift.py` (normalizer recalibration under the shifted regime), `41.benchmark_A_seed_duplicate_context_audit.py` (shortcut / duplication probes, S) | `results/51_field_group_train_ablation`, `results/53_field_group_removal`, `results/59_raw_semantic`, `results/61_representation_shift`, `results/67_threshold_recalibration`, `results/41A_benchmark_Aprime_audit` |
| VI-E Withheld evidence and generation conditions | 47 (uniform Missing-8, three mask seeds), `19.generate_realistic_missing_field_tests.py` → `18.evaluate_realistic_missing_field_robustness.py` (structural burst missingness, k = 2/5/8 fields), `48.fresh_realizations_evaluation.py` (three fresh realizations of the frozen protocol), `49.generator_b_transfer_evaluation.py` (parameter-stress regimes; the Generator-B transfer run is reported in S only) | `results/47_missing8_aggregation`, `results/realistic_missing_field_robustness`, `results/48_fresh_realizations`, `results/49_parameter_stress`, `results/49_generator_b_transfer` |
| VI-F Case study with records from an independent execution environment | `nos3_detector_evaluation.py` (frozen detectors on the mapped NOS3 records of `nos3_repeatability.zip`, extracted next to the script as `09_nos3_repeatability_v1/`; mapping v1), `nos3_mapping_sensitivity.py` (mapping v2 re-analysis) | `results/nos3_detector_evaluation`, `results/nos3_mapping_sensitivity` |
| VI-G Operational diagnostics | `54.inference_cost_benchmark.py` (per-endpoint latency and memory at L=10), `56.evidence_response_diagnostics.py` (first observable-change anchor), `64.alarm_delay_evaluation.py` (generator-activation anchor, alarm budgets) | `results/54_inference_cost`, `results/56_evidence_response_diagnostics`, `results/64_alarm_delay_evaluation` |
| Additional diagnostics (included for completeness; reported only where the manuscript or S refers to them) | `24.lstm_finetuned_distilbert_l1_baseline.py`, `60.paired_attack_response_audit.py`, `65.alarm_memory_controls.py`, `66.evidence_transition_alarm_evaluation.py`, `73.novelty_evidence_validation.py`, `nos3_missingness_adaptation.py`, `taes_shared_nominal_audit.py` | `results/lstm_finetuned_distilbert_l1`, `results/60_…`, `results/65_…`, `results/66_…`, `results/73_…`, `results/nos3_missingness_adaptation` |

## 6. Figures

| Folder | Script | Figure |
|---|---|---|
| `graph/01_confusion_matrix_L10` | `21_6-4_confusion_matrix_10seed.py` | Row-normalized L=10 confusion matrix, mean of the ten head seeds (counts are ten-seed means rounded to integers). |
| `graph/02_context_length_sweep` | `22. making_grah_6-2.py` | Macro-F1 vs. context length on the common endpoints, with the evidence-availability reference. |
| `graph/03_evidence_retention_sweep` | `23.fig03_evidence_retention_sweep.py` | Step 72 event-retention sweep (TST L=10, Mean+Max L=10, TST L=1, reference). |
| `graph/05_field_group_removal` | `25.group_removal_graph_step53.py` | Test-time field-group removal (Step 53). |
| `graph/05_class_f1_by_group` | `26.6-5_class_f1_by_group_step53.py` | Per-class F1 under test-time group removal (Step 53). |
| `graph/S6_context_length_sweep_L20` | `31.l20_population_lsweep.py` | Supplementary: context-length curve on the L=20 matched population. |

The scripts of figures 01–05 verify their inputs against the run manifests before plotting and stop if the plotted
values differ from the values reported in the manuscript; the plotted numbers are stored next to the figure in `*_values.csv`
and the input files in `*_sources.txt`. Re-rendering the figures needs the full `outputs/` tree (predictions), which is
regenerated by the pipeline above.

## 7. Seeds and statistical procedure

- Aggregation heads (TST, pooling + MLP, LSTM): ten head seeds 20260610–20260619 on the same frozen embeddings; `20260610` is the main run.
- Encoder comparison (Step 45/46): five encoder seeds; TST reference with a fixed seed. Field-group ablation (Step 51): five seeds. Uniform Missing-8 (Step 47): three mask seeds, identical masks for all models. Fresh realizations (Step 48): three realizations of the frozen generation protocol. Twenty-window extension (Steps 27–34): three seeds on the matched cohort.
- Generation seeds, mask seeds and bootstrap seeds are written to the run manifest / `sources.txt` of each run; the matched 2×2 bootstrap (Step 34) uses RNG seed 20261013 with 20,000 whole-episode resamples.
- Main benchmark contrasts: class-stratified paired episode bootstrap, 20,000 resamples; paired-regeneration experiments: paired block bootstrap. Bonferroni-adjusted intervals are reported per analysis family.
- Training is deterministic up to GPU non-determinism (fp16/AMP); reported seed SDs are the sample SDs over the seeds listed above.

## 8. Provenance checks

- `data/manifests/`: build manifest (`dataset_id`, input hashes), split manifest and benchmark audit reports.
- Every run folder in `results/` contains the gate report and/or `sources.txt` that lists the input files (with hashes) and the code version used.
- Re-running the generators with the shipped `shared_nominal.py` must reproduce the six input hashes recorded in the build manifest (line endings excepted).

## 9. Citation

```bibtex
@article{park2026temporal,
  title   = {Temporal Threat Detection in Satellite Networks Using Semantic Observable Representations},
  author  = {Park, Junbeom and Abbasi, Ali and Yoon, Zizung},
  journal = {IEEE Transactions on Aerospace and Electronic Systems},
  year    = {2026},
  note    = {manuscript TAES-2026-2877, under review}
}
```

Contact: Junbeom Park, Korea Aerospace University — <email>

License: <fill in before release>
