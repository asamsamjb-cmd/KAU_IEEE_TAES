#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TAES shared-nominal / Step 15: paired, class-stratified episode bootstrap.

사용법: 이 파일을 10~14번 .py 파일과 같은 프로젝트 폴더에 두고 실행합니다.
필수 패키지: numpy, pandas. GPU / torch / 재학습 / 추가 데이터 생성 불필요.
기본 실행: 5 architectures x 10 saved head seeds, L10, 20,000 resamples.

Statistical contract (unchanged; shared nominal input contract v3.0):
* Estimand = arithmetic mean of the 10 per-seed, sequence-level Macro-F1s.
  Never pool seeds' confusion matrices before F1, or average probabilities.
* Resampling unit = whole episode. Within each true class, sample the same
  number of episodes with replacement. Retain ALL sequences of each selected
  episode, including its multiplicity. Longer episodes retain their original
  larger contribution to the sequence-level metric (not episode-mean F1).
* Apply one common episode draw to all models AND all fixed head seeds.
* Primary CI = two-sided percentile 95% interval, conditional on the observed
  trained runs, fixed encoder, training data, generator realization/protocol,
  and class-specific episode counts. Seeds are NOT resampled. Their sample SD
  is reported separately and is NOT a confidence interval.
* Four predeclared contrasts: Mean/Max/MeanMax/LSTM minus TST. Also report
  Bonferroni marginal 98.75% percentile intervals (nominal family coverage 95%,
  subject to bootstrap approximation). Do not infer equivalence from a CI
  containing zero. No post-hoc practical-equivalence margin or p-values.
* Episode independence/exchangeability within class is an assumption. This
  analysis does not establish generalization to a new encoder or generator.
* Original predictions are read-only. Each execution creates a new run folder.
  Missing/invalid inputs STOP the run; no partial-seed 'formal' result is made.

CLI examples (optional; PyCharm Run needs no arguments):
  python "15.paired_episode_bootstrap.py"
  python "15.paired_episode_bootstrap.py" --audit-only
  python "15.paired_episode_bootstrap.py" --project-root "D:/research/project"

Requires COMPLETE revised Step 12, 13 and 14 runs. Step 12's recorded seed
locations are authoritative, including a reused Step 11 main run. No legacy
directory fallback or partial-seed analysis is performed.

Outputs in outputs/shared_nominal_v2/15_paired_episode_bootstrap/run_<UTC timestamp>/:
  protocol.json, input_audit.csv, metrics_by_seed.csv, model_summary.csv,
  paired_comparisons.csv, bootstrap_distribution.npz,
  seed_average_row_normalized_confusion.csv, episode_inventory.csv,
  analysis_summary.md, run_manifest.json (status must be COMPLETE).

Implementation uses NumPy episode-confusion sufficient statistics to avoid
materializing resampled prediction rows. No SciPy dependency is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ======================== USER CONFIG ========================
BASE_DIR = Path(__file__).resolve().parent
SEEDS = tuple(range(20260610, 20260620))
N_BOOTSTRAP = 20_000
BOOTSTRAP_RANDOM_SEED = 20260926
BOOTSTRAP_BATCH_SIZE = 100
EXPECTED_TEST_L10_COUNT = None  # Read from the verified current Step 10 report.
MODELS = ('TST', 'Mean_MLP', 'Max_MLP', 'MeanMax_MLP', 'LSTM')
LABELS = (
    'NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
    'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
    'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE',
)
REVISED_OUTPUT_REL = 'outputs/shared_nominal_v2'
METADATA_REL = REVISED_OUTPUT_REL + '/tst_sequences_finetuned_main/test_tst_sequence_metadata_L10.csv'
OUTPUT_REL = REVISED_OUTPUT_REL + '/15_paired_episode_bootstrap'
TST_ROOT_REL = REVISED_OUTPUT_REL + '/tst_training_finetuned_multiseed'
POOLING_ROOT_REL = REVISED_OUTPUT_REL + '/39_pooling_mlp_finetuned_main'
LSTM_ROOT_REL = REVISED_OUTPUT_REL + '/lstm_finetuned_distilbert_l10'
# Source directories retain the names actually used by Steps 12, 13 and 14.
# Do not rename the existing '39_pooling...' directory.
PATH_TEMPLATES = {
    'TST': TST_ROOT_REL + '/seed_{seed}/L10/predictions/test_predictions_L10.csv',  # documentation only; resolved from Step 12
    'Mean_MLP': POOLING_ROOT_REL + '/predictions/mean_seed{seed}_test_predictions.csv',
    'Max_MLP': POOLING_ROOT_REL + '/predictions/max_seed{seed}_test_predictions.csv',
    'MeanMax_MLP': POOLING_ROOT_REL + '/predictions/meanmax_seed{seed}_test_predictions.csv',
    'LSTM': LSTM_ROOT_REL + '/predictions/seed{seed}_test_predictions.csv',
}
POOLING_MODES = {'Mean_MLP': 'mean', 'Max_MLP': 'max', 'MeanMax_MLP': 'meanmax'}
PROVENANCE_KEYS = ('dataset_id', 'step8_run_id', 'step9_run_id', 'step10_run_id')
LENGTHS = [1, 2, 3, 5, 10]
KEY_COLUMNS = ('episode_id', 'sequence_start_row', 'sequence_end_row',
               'sequence_start_step', 'sequence_end_step')

FIELDS = ('observation_stage', 'path_type', 'direction', 'source_role',
          'destination_role', 'protocol_context', 'operational_phase',
          'message_type', 'service_context', 'command_intent',
          'payload_or_parameter_context', 'sequence_freshness_status',
          'control_security_state', 'signal_quality_state', 'tracking_stability',
          'navigation_consistency', 'timing_consistency', 'decoder_status',
          'command_execution_result', 'retry_timeout_state', 'session_state',
          'state_transition_consistency')
META = ['scenario_id', 'episode_id', 'episode_type', 'episode_step', 'window_id', 'timestamp', 'label']

def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def integers(series, context):
    x = pd.to_numeric(series, errors='raise').to_numpy(dtype=np.float64)
    require(np.isfinite(x).all() and np.equal(x, np.floor(x)).all()
            and (np.abs(x) < 2**53).all(), f'{context}: invalid integer values')
    return x.astype(np.int64)


def read_csv(path):
    return pd.read_csv(path, encoding='utf-8-sig', dtype={'episode_id': str})


def load_metadata(path, expected_count=EXPECTED_TEST_L10_COUNT):
    require(isinstance(expected_count, int) and expected_count > 0,
            'A positive expected count from the verified Step 10 report is required.')
    d = read_csv(path)
    needed = {'sequence_id', 'label_id', 'label', 'sequence_length',
              'source_split', *KEY_COLUMNS}
    require(needed <= set(d), f'{path}: missing metadata columns {sorted(needed-set(d))}')
    require(len(d) == expected_count, f'{path}: expected {expected_count} rows, got {len(d)}')
    require(not d[list(needed)].isna().any().any(), f'{path}: missing metadata values')
    for col in ('sequence_id', 'label_id', 'sequence_length', *KEY_COLUMNS[1:]):
        d[col] = integers(d[col], f'{path}:{col}')
    require(d.sequence_id.is_unique, f'{path}: duplicate sequence_id')
    d = d.sort_values('sequence_id').reset_index(drop=True)
    require(np.array_equal(d.sequence_id, np.arange(len(d))), f'{path}: sequence_id must be 0..N-1')
    require(set(d.label_id) == set(range(len(LABELS))), f'{path}: expected all six classes')
    require(np.array_equal(d.label, np.array(LABELS)[d.label_id]), f'{path}: label mapping mismatch')
    require((d.source_split == 'test').all(), f'{path}: non-test metadata')
    require((d.sequence_length == 10).all(), f'{path}: non-L10 sequences')
    require((d.sequence_end_step - d.sequence_start_step == 9).all(), f'{path}: non-L10 steps')
    require((d[list(KEY_COLUMNS[1:])] >= 0).all().all(), f'{path}: negative positions')
    require((d.sequence_end_row >= d.sequence_start_row).all(), f'{path}: invalid row interval')
    require(d.episode_id.str.strip().ne('').all(), f'{path}: empty episode_id')
    require(not d.duplicated(['episode_id', 'sequence_end_step']).any(), f'{path}: duplicate endpoint')
    require(d.sequence_end_row.is_unique, f'{path}: duplicate global endpoint row')
    require(d.groupby('episode_id').label_id.nunique().eq(1).all(),
            f'{path}: mixed-label episodes: class-stratified protocol is not applicable')
    if 'sequence_num_label_changes' in d:
        require((integers(d.sequence_num_label_changes, str(path)) == 0).all(),
                f'{path}: label transitions present; review protocol')
    return d



EFFECT_COLUMNS = ['audit_current_observable_effect', 'audit_history_effect_window_count',
                  'audit_context_effect_window_count', 'audit_effect_stratum']
EFFECT_STRATA = ['current_changed', 'history_only_changed', 'no_observable_change']


def validate_nominal_contract(build, source9, run8, source_audit, digest, dataset):
    """Authenticate shared nominal semantics; audit flags never become predictors."""
    policy = build.get('policy', {})
    require(build.get('status') == 'complete' and build == source9.get('step7_build')
            and policy.get('nominal_substrate') == 'shared_nominal_v2'
            and policy.get('columns') == list(META) + list(FIELDS)
            and policy.get('label_order') == list(LABELS) and policy.get('window_seconds') == 1,
            'Use the completed shared_nominal_v2 Step 7 build with the agreed 22 observables.')
    require(source9.get('step7_build_manifest_sha256') == digest
            and run8.get('step7_build_manifest_sha256') == digest
            and Path(source9.get('source_dataset_dir', '')).resolve() == dataset.resolve(),
            'Steps 7/8/9 refer to different shared nominal builds.')
    spec = build.get('observability_audit', {})
    files = {s: f'audit/{s}_window_observable_effects.csv' for s in ('train', 'val', 'test')}
    hashes = {name: build.get('artifacts', {}).get(name) for name in files.values()}
    expected = {'schema_version': 'counterfactual_observable_change_v1', 'dataset_id': build.get('dataset_id'),
        'source_dataset_dir': str(dataset.resolve()), 'step7_build_manifest_sha256': digest,
        'split_files': files, 'artifacts': hashes, 'scope': spec.get('scope')}
    require(spec.get('schema_version') == expected['schema_version'] and spec.get('split_files') == files
            and bool(spec.get('scope')) and all(isinstance(v, str) and len(v) == 64 for v in hashes.values())
            and source_audit == expected, 'Observable-effect audit is not bound to the same Step 7 build.')


def validate_effect_frame(effects, metadata, length):
    """Exact row identity plus causal current/history counts; no population filtering."""
    require(list(effects.columns) == list(metadata.columns) + EFFECT_COLUMNS,
            'Observable-effect sidecar must be exact metadata plus four diagnostic columns.')
    require(len(effects) == len(metadata) and not effects.isna().any().any(),
            'Missing or extra observable-effect rows/values.')
    for column in metadata:
        require(np.array_equal(effects[column].astype(str).to_numpy(), metadata[column].astype(str).to_numpy()),
                f'Observable-effect sidecar row identity mismatch: {column}')
    current, history, whole = [integers(effects[c], c) for c in EFFECT_COLUMNS[:3]]
    require(np.isin(current, [0, 1]).all() and ((history >= 0) & (history < length)).all()
            and np.array_equal(whole, current + history), 'Invalid observable-effect context counts.')
    strata = np.where(current == 1, 'current_changed',
                      np.where(history > 0, 'history_only_changed', 'no_observable_change'))
    require(np.array_equal(effects[EFFECT_COLUMNS[3]].to_numpy(), strata),
            'Observable-effect stratum disagrees with context counts.')
    result = effects[EFFECT_COLUMNS].copy().reset_index(drop=True)
    for column, values in zip(EFFECT_COLUMNS[:3], (current, history, whole)): result[column] = values
    return result


def load_prediction(path, ref, expected=None):
    before = sha256_file(path)
    d = read_csv(path)
    needed = {'sequence_id', 'true_label_id', 'pred_label_id', 'true_label',
              'pred_label', 'common_population', *KEY_COLUMNS, *EFFECT_COLUMNS}
    require(needed <= set(d), f'{path}: missing columns {sorted(needed-set(d))}')
    require(len(d) == len(ref), f'{path}: row count {len(d)} != {len(ref)}')
    require(not d[list(needed)].isna().any().any(), f'{path}: missing prediction values')
    require((integers(d.common_population, str(path)) == 1).all(),
            f'{path}: L10 predictions must all belong to the verified common population')
    if expected is not None:
        require(before == expected['sha256'], f'{path}: prediction differs from completed run')
        for key, value in expected['prediction_fields'].items():
            require(key in d and d[key].eq(value).all(), f'{path}: mismatched {key}')
    if 'sequence_length' in d:
        require((integers(d.sequence_length, str(path)) == 10).all(), f'{path}: non-L10 prediction')
    for col in ('sequence_id', 'true_label_id', 'pred_label_id', *KEY_COLUMNS[1:]):
        d[col] = integers(d[col], f'{path}:{col}')
    require(d.sequence_id.is_unique, f'{path}: duplicate sequence_id')
    reordered = not np.array_equal(d.sequence_id, ref.sequence_id)
    d = d.sort_values('sequence_id').reset_index(drop=True)
    for col in ('sequence_id', *KEY_COLUMNS):
        require(np.array_equal(d[col], ref[col]), f'{path}: {col} mismatch against Step 10')
    for col in ('sequence_start_timestamp', 'sequence_end_timestamp'):
        if col in ref and col in d:
            require(np.array_equal(d[col], ref[col]), f'{path}: {col} mismatch')
    require(set(EFFECT_COLUMNS) <= set(ref), 'Verified Step 10 observable-effect reference is required.')
    for column in EFFECT_COLUMNS:
        require(np.array_equal(d[column].astype(str), ref[column].astype(str)),
                f'{path}: {column} mismatch against Step 10 diagnostics')
    y = d.true_label_id.to_numpy()
    p = d.pred_label_id.to_numpy()
    require(np.array_equal(y, ref.label_id), f'{path}: true_label_id mismatch')
    require(((p >= 0) & (p < len(LABELS))).all(), f'{path}: predicted label outside 0..5')
    require(np.array_equal(d.true_label, np.array(LABELS)[y]), f'{path}: true_label mismatch')
    require(np.array_equal(d.pred_label, np.array(LABELS)[p]), f'{path}: pred_label mismatch')
    if 'correct' in d:
        require(np.array_equal(integers(d.correct, str(path)), (y == p).astype(int)),
                f'{path}: correct column mismatch')
    probcols = ['prob_' + x.lower() for x in LABELS]
    require(set(probcols) <= set(d), f'{path}: missing class probability columns')
    probs = d[probcols].to_numpy(dtype=float)
    require(np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all(),
            f'{path}: invalid probabilities')
    require(np.allclose(probs.sum(1), 1, rtol=0, atol=2e-6), f'{path}: probability sums != 1')
    require(np.array_equal(probs.argmax(1), p), f'{path}: probability argmax != prediction')
    if 'confidence' in d:
        require(np.allclose(d.confidence.to_numpy(float), probs.max(1), rtol=0, atol=2e-6),
                f'{path}: confidence mismatch')
    require(before == sha256_file(path), f'{path}: changed while being read')
    return p, {'path': str(Path(path).resolve()), 'sha256': before,
               'rows': len(d), 'episodes': int(d.episode_id.nunique()),
               'rows_reordered': bool(reordered), 'alignment': 'PASS'}


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def track_file(path, observed, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected is not None:
        require(digest == expected, f'Missing/changed completed artifact: {path}')
    if str(path) in observed:
        require(observed[str(path)] == digest, f'Input changed during analysis: {path}')
    observed[str(path)] = digest
    return digest


def load_json(path, observed):
    before = track_file(path, observed)
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    track_file(path, observed, before)
    require(isinstance(value, dict), f'{path}: expected a JSON object')
    return value


def verify_selected(root, hashes, names, observed):
    """Hash only the artifacts used here, including the relevant saved checkpoint."""
    root = Path(root).resolve()
    for name in names:
        path = (root/name).resolve()
        digest = hashes.get(name)
        require(path.is_relative_to(root) and isinstance(digest, str) and len(digest) == 64,
                f'{root}: missing/invalid completed artifact record for {name}')
        track_file(path, observed, digest)


def resolve_root(root, value, default):
    path = Path(default if value is None else value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def exact_seed_records(records, seeds, context):
    require(isinstance(records, list) and len(records) == len(seeds), f'{context}: incomplete seed grid')
    require(all(isinstance(item, dict) and type(item.get('seed')) is int for item in records),
            f'{context}: invalid seed record')
    require({item['seed'] for item in records} == set(seeds), f'{context}: duplicate/missing/foreign seed')
    return {item['seed']: item for item in records}



def validate_saved_diagnostic(path, observed, provenance, source, recorded_path, seed=None, mode=None):
    require(isinstance(recorded_path, str) and Path(recorded_path).resolve() == path.resolve(),
            f'{path}: metrics report refers to another diagnostic file')
    diagnostic = load_json(path, observed)
    require(diagnostic.get('provenance') == provenance and diagnostic.get('source_audit') == source.get('observability_audit')
            and diagnostic.get('split') == 'test' and diagnostic.get('sequence_length') == 10,
            f'{path}: diagnostic provenance/context mismatch')
    if seed is not None:
        require(diagnostic.get('seed') == seed and diagnostic.get('pooling_mode') == mode,
                f'{path}: diagnostic seed/mode mismatch')


def check_inputs(root, seeds, tst_root=None, pooling_root=None, lstm_root=None, return_context=False):
    """Resolve the completed 12/13/14 runs without importing or executing training code."""
    root = Path(root).resolve()
    observed, expected = {}, {}
    roots = {'TST': resolve_root(root, tst_root, TST_ROOT_REL),
             'pooling': resolve_root(root, pooling_root, POOLING_ROOT_REL),
             'LSTM': resolve_root(root, lstm_root, LSTM_ROOT_REL)}
    runner = load_json(roots['TST']/'multiseed_run_manifest.json', observed)
    require(runner.get('status') == 'COMPLETE', 'Step 12 is not COMPLETE; finish all requested seeds first.')
    identity = runner.get('identity', {})
    require(runner.get('job_id') == object_hash(identity), 'Step 12 run identity mismatch')
    require(identity.get('seeds') == list(seeds) and identity.get('ordered_lengths') == LENGTHS
            and identity.get('missing_aware') is False, 'Step 12 must use the agreed ten seeds, all L, and clean training.')
    require(Path(identity.get('output_root', '')).resolve() == roots['TST'], 'Step 12 output was moved/mislabelled')
    verify_selected(roots['TST'], runner.get('artifact_sha256', {}),
                    ['multiseed_validation.json', 'multiseed_raw_results.csv'], observed)
    validation = load_json(roots['TST']/'multiseed_validation.json', observed)
    require(validation.get('status') == 'VALIDATED' and validation.get('complete_grid') is True
            and validation.get('seeds') == list(seeds) and validation.get('ordered_lengths') == LENGTHS,
            'Step 12 validation is incomplete or uses another protocol')
    source = validation.get('source_contract', {})
    require(all(isinstance(source.get(k), str) and source[k] for k in PROVENANCE_KEYS), 'Missing upstream provenance')
    provenance = {k: source[k] for k in PROVENANCE_KEYS}
    runs12 = exact_seed_records(runner.get('runs'), seeds, 'Step 12')
    validated12 = exact_seed_records(validation.get('runs'), seeds, 'Step 12 validation')
    raw = read_csv(roots['TST']/'multiseed_raw_results.csv')
    needed = {'seed', 'sequence_length', 'seed_output_root', 'seed_manifest_sha256',
              'step11_run_id', *PROVENANCE_KEYS, 'test_common_count', 'test_common_accuracy', 'test_common_macro_f1'}
    require(needed <= set(raw), 'Step 12 aggregate lacks revised provenance/metrics')
    pairs = list(zip(integers(raw.seed, 'Step 12 seed'), integers(raw.sequence_length, 'Step 12 length')))
    require(len(pairs) == len(seeds)*len(LENGTHS) and set(pairs) == {(s, l) for s in seeds for l in LENGTHS},
            'Step 12 aggregate has duplicate/missing seed x L rows')
    paths, baseline = [], None
    for seed in seeds:
        item, verified = runs12[seed], validated12[seed]
        seed_root = resolve_root(root, item['output_root'], '')
        require(seed_root == resolve_root(root, verified['output_root'], ''), 'Step 12 seed location mismatch')
        run_path = seed_root/'run_manifest.json'
        require(isinstance(item.get('manifest_sha256'), str) and len(item['manifest_sha256']) == 64,
                f'Step 12 seed={seed}: missing completed manifest hash')
        track_file(run_path, observed, item.get('manifest_sha256'))
        require(item.get('manifest_sha256') == verified.get('manifest_sha256'), 'Step 12 seed manifest mismatch')
        run = load_json(run_path, observed)
        require(run.get('status') == 'COMPLETE' and run.get('run_id') == item.get('step11_run_id') == verified.get('run_id'),
                f'Step 11 seed={seed}: incomplete or different run')
        settings = run.get('settings', {})
        require(run.get('source_contract') == source and run.get('dataset_id') == source['dataset_id'],
                f'Step 11 seed={seed}: different dataset/encoder/sequence artifacts')
        require(settings.get('DEFAULT_RANDOM_SEED') == seed and settings.get('USE_MISSING_AWARE_TRAINING') is False
                and run.get('selected_sequence_lengths') == LENGTHS
                and settings.get('SEQUENCE_FIT_SUBSET') == 'full' and settings.get('PRIMARY_TEST_SUBSET') == 'full'
                and settings.get('PRIMARY_SELECTION_METRIC') == 'val_macro_f1'
                and settings.get('USE_ROBUST_EVAL_AS_PRIMARY_FOR_LONG_CONTEXTS') is False,
                f'Step 11 seed={seed}: different seed/evaluation protocol')
        require(Path(run.get('output_root', '')).resolve() == seed_root, 'Step 11 output was moved/mislabelled')
        signature = {k: run.get(k) for k in ('script_sha256', 'runtime', 'selected_sequence_lengths', 'seed_policy', 'comparison_policy')}
        signature['settings'] = {k: v for k, v in settings.items() if k != 'DEFAULT_RANDOM_SEED'}
        require(baseline is None or signature == baseline, 'Step 11 mixes head-training protocols across seeds')
        baseline = signature
        name = 'L10/predictions/test_predictions_L10.csv'
        report_name = 'L10/reports/evaluation_metrics_L10.json'
        diagnostic_name = 'L10/reports/observable_effect_diagnostics_test_L10.json'
        verify_selected(seed_root, run.get('artifacts', {}),
                        [name, report_name, diagnostic_name, 'L10/checkpoints/best_tst_model_L10.pt'], observed)
        report = load_json(seed_root/report_name, observed)
        fields = {**provenance, 'step11_run_id': run['run_id']}
        validate_saved_diagnostic(seed_root/diagnostic_name, observed, fields, source,
            report.get('observable_effect_diagnostics', {}).get('test'))
        require(report.get('provenance') == fields and report.get('train_config', {}).get('random_seed') == seed
                and report.get('sequence_length') == 10 and report.get('label_order') == list(LABELS)
                and report.get('test_metric_protocol') == 'standard_unperturbed'
                and report.get('common_endpoint_report') == source.get('common_endpoint_report'),
                f'Step 11 seed={seed}: evaluation report mismatch')
        row = raw.loc[(raw.seed == seed) & (raw.sequence_length == 10)].iloc[0].to_dict()
        require(all(row.get(k) == v for k, v in fields.items())
                and Path(row['seed_output_root']).resolve() == seed_root
                and row['seed_manifest_sha256'] == observed[str(run_path)], 'Step 12 aggregate/source mismatch')
        expected[('TST', seed)] = {'sha256': run['artifacts'][name], 'prediction_fields': {**fields, 'source_split': 'test'},
            'metric_records': [report['test_summary'], report['test_subset_summaries']['common_population']],
            'aggregate': row, 'source_manifest': str(run_path), 'source_manifest_sha256': observed[str(run_path)]}
        paths.append(('TST', seed, seed_root/name))

    for family, step, modes in [('pooling', 13, POOLING_MODES), ('LSTM', 14, {'LSTM': None})]:
        folder = roots[family]
        run_path = folder/'run_manifest.json'
        run = load_json(run_path, observed)
        identity = run.get('identity', {})
        require(run.get('status') == 'COMPLETE' and run.get('run_id') == object_hash(identity),
                f'Step {step} is incomplete or has an invalid run identity')
        require(identity.get('source_contract') == source and run.get('dataset_id') == source['dataset_id'],
                f'Step {step}: different dataset/encoder/sequence artifacts')
        require(Path(identity.get('output_root', '')).resolve() == folder, f'Step {step} output was moved/mislabelled')
        settings = identity.get('settings', {})
        require(settings.get('SEEDS') == list(seeds) and settings.get('SEQUENCE_LENGTH') == 10,
                f'Step {step}: different seed list or context length')
        keys = {f'{mode}:{seed}' if mode is not None else str(seed) for mode in modes.values() for seed in seeds}
        require(set(run.get('completed_runs', {})) == keys, f'Step {step}: incomplete/extra trial grid')
        for model, mode in modes.items():
            for seed in seeds:
                prefix = f'{mode}_seed{seed}' if mode is not None else f'seed{seed}'
                key = f'{mode}:{seed}' if mode is not None else str(seed)
                trial = run['completed_runs'][key]
                name = f'predictions/{prefix}_test_predictions.csv'
                report_name = f'reports/{prefix}_test_metrics.json'
                result_name = f'reports/{prefix}_result_summary.json'
                checkpoint = (f'models/best_pooling_mlp_L10_{mode}_seed{seed}.pt' if mode is not None
                              else f'models/best_lstm_finetuned_distilbert_L10_seed{seed}.pt')
                diagnostic_name = f'reports/{prefix}_test_observable_effect_diagnostics.json'
                names = [name, report_name, result_name, checkpoint, diagnostic_name]
                require(all(trial.get('artifacts', {}).get(n) == run.get('artifacts', {}).get(n) for n in names),
                        f'Step {step}: trial/complete artifact mismatch')
                verify_selected(folder, run.get('artifacts', {}), names, observed)
                report, result = load_json(folder/report_name, observed), load_json(folder/result_name, observed)
                fields = {**provenance, f'step{step}_run_id': run['run_id'], 'seed': seed}
                if mode is not None:
                    fields['pooling_mode'] = mode
                require(result == trial.get('result') and all(record.get(k) == v for record in (report, result) for k, v in fields.items()),
                        f'Step {step}: trial provenance/result mismatch')
                validate_saved_diagnostic(folder/diagnostic_name, observed,
                    {**provenance, f'step{step}_run_id': run['run_id']}, source,
                    report.get('observable_effect_diagnostics'), seed, mode)
                require(result.get('sequence_length') == 10, f'Step {step}: non-L10 result')
                expected[(model, seed)] = {'sha256': run['artifacts'][name], 'prediction_fields': fields,
                    'metric_records': [report, result], 'aggregate': result,
                    'source_manifest': str(run_path), 'source_manifest_sha256': observed[str(run_path)]}
                paths.append((model, seed, folder/name))

    # The heads already verified their full training inputs. Match their saved
    # contract to the current completed 8/9/10 manifests; hash the metadata used
    # here. No multi-gigabyte embedding reload or model inference is needed.
    upstream = {'step8': root/REVISED_OUTPUT_REL/'distilbert_window_baseline',
                'step9': root/REVISED_OUTPUT_REL/'tst_embeddings_finetuned_main',
                'step10': root/REVISED_OUTPUT_REL/'tst_sequences_finetuned_main'}
    upstream_runs = {}
    for step, folder in upstream.items():
        path = folder/'run_manifest.json'
        digest = source.get('manifest_sha256', {}).get(step)
        require(isinstance(digest, str) and len(digest) == 64, f'Missing {step} manifest hash')
        track_file(path, observed, digest)
        run = load_json(path, observed)
        require(run.get('status') == 'COMPLETE' and run.get('dataset_id') == source['dataset_id']
                and run.get('run_id') == source[f'{step}_run_id'], f'{step}: incomplete/different upstream run')
        upstream_runs[step] = run
    dataset = root/'dataset/shared_nominal_v2'
    build = load_json(dataset/'build_manifest.json', observed)
    validate_nominal_contract(build, upstream_runs['step9'].get('source_contract', {}),
        upstream_runs['step8'], source.get('observability_audit'),
        observed[str((dataset/'build_manifest.json').resolve())], dataset)
    require(upstream_runs['step10'].get('source', {}).get('observability_audit') == source.get('observability_audit'),
            'Step 10/shared nominal audit mismatch')
    seq = upstream['step10']
    meta = root/METADATA_REL
    effect_path = seq/'test_tst_sequence_observable_effects_L10.csv'
    names = ['sequence_build_report_index.json', 'sequence_build_report_L10.json', meta.name, effect_path.name]
    require(all(upstream_runs['step10'].get('artifacts', {}).get(n) == source.get('step10_artifacts', {}).get(n) for n in names),
            'Step 10 current/training artifact inventories differ')
    verify_selected(seq, source.get('step10_artifacts', {}), names, observed)
    index = load_json(seq/'sequence_build_report_index.json', observed)
    report = load_json(seq/'sequence_build_report_L10.json', observed)
    for record in (index, report):
        require(all(record.get(k) == source[k] for k in PROVENANCE_KEYS[:-1])
                and record.get('run_id') == source['step10_run_id'], 'Step 10 report provenance mismatch')
    for record in (index, report):
        require(record.get('observability_audit') == source.get('observability_audit'),
                'Step 10 report observable-effect source mismatch')
    detail = report.get('splits', {}).get('test', {})
    require(detail.get('sequence_observable_effects_sha256') == source['step10_artifacts'][effect_path.name]
            and Path(detail.get('sequence_observable_effects_csv_path', '')).resolve() == effect_path.resolve(),
            'Step 10 test observable-effect file record mismatch')
    common = source.get('common_endpoint_report', {})
    require(common.get('reference_length') == 10 and common == index.get('common_endpoint_report'),
            'Step 10 common endpoint contract mismatch')
    population = common.get('splits', {}).get('test', {})
    count = report.get('splits', {}).get('test', {}).get('num_sequences')
    require(type(count) is int and count > 0 and count == population.get('common_endpoint_count')
            and count == population.get('available_sequences_by_L', {}).get('10'), 'Step 10 L10/common count mismatch')
    if EXPECTED_TEST_L10_COUNT is not None:
        require(count == EXPECTED_TEST_L10_COUNT, 'Optional configured L10 count differs from Step 10')
    context = {'source_contract': source, 'expected_count': count, 'population': population,
               'step10_test_report': report['splits']['test'], 'observed': observed,
               'observable_effect_path': str(effect_path), 'expected': expected, 'roots': {k: str(v) for k, v in roots.items()}}
    return (meta, paths, context) if return_context else (meta, paths)


def validate_metadata_population(ref, context):
    ends = ref.sequence_end_row.to_numpy(dtype='<i8')
    population, report = context['population'], context['step10_test_report']
    require(hashlib.sha256(np.sort(ends).tobytes()).hexdigest() == population['sorted_end_row_indices_sha256']
            and hashlib.sha256(ends.tobytes()).hexdigest() == report['end_row_indices_sha256'],
            'Metadata endpoints differ from the current Step 10 L10/common population')
    require(ref.label.value_counts().reindex(LABELS, fill_value=0).to_dict() == report['sequence_label_distribution'],
            'Metadata label distribution differs from Step 10')
    require(ref.episode_id.nunique() == report['num_episodes'] - report['skipped_short_episodes'],
            'Metadata eligible episode count differs from Step 10')


def verify_prediction_metrics(y, pred, expected):
    cm = np.bincount(y*len(LABELS)+pred, minlength=len(LABELS)**2).reshape(len(LABELS), len(LABELS))
    values = {'macro_f1': float(macro_f1(cm)), 'accuracy': float((y == pred).mean()), 'total': len(y)}
    for record in expected['metric_records']:
        for key, value in values.items():
            require(isinstance(record.get(key), (int, float)) and np.isfinite(record[key])
                    and np.isclose(record[key], value, rtol=1e-9, atol=1e-10), f'Saved {key} differs from prediction rows')
    aggregate = expected['aggregate']
    for key, value in [('test_common_count', len(y)), ('test_common_macro_f1', values['macro_f1']),
                       ('test_common_accuracy', values['accuracy'])]:
        require(isinstance(aggregate.get(key), (int, float)) and np.isfinite(aggregate[key])
                and np.isclose(aggregate[key], value, rtol=1e-9, atol=1e-10), f'Aggregate {key} differs from prediction rows')
    return values


def recheck_inputs(observed):
    for path, digest in observed.items():
        require(sha256_file(path) == digest, f'Input changed during analysis: {path}')


def macro_f1(cm):
    """Macro-F1 over last two confusion axes (true, predicted)."""
    cm = np.asarray(cm)
    den = cm.sum(axis=-1) + cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    return np.divide(2.0 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(axis=-1)


def episode_confusions(y, predictions, episode_codes, n_episodes, n_classes):
    """Return [episode, run, true, predicted]; run order is model-major."""
    predictions = np.asarray(predictions)
    counts = np.zeros((n_episodes, len(predictions), n_classes, n_classes), dtype=np.int64)
    for r, p in enumerate(predictions):
        codes = (episode_codes * n_classes + y) * n_classes + p
        counts[:, r] = np.bincount(codes, minlength=n_episodes*n_classes*n_classes).reshape(
            n_episodes, n_classes, n_classes)
    return counts


def draw_episode_weights(strata, batch, rng):
    """Multinomial multiplicities exactly represent within-class cluster draws."""
    weights = np.zeros((batch, len(strata)), dtype=np.int64)
    for c in np.unique(strata):
        ix = np.flatnonzero(strata == c)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=batch)
    return weights


def bootstrap_scores(counts, strata, n_seeds, n_models, n_resamples,
                     random_seed, batch_size, progress=True):
    require(counts.shape[1] == n_seeds*n_models, 'run/seed dimension mismatch')
    require(len(strata) == counts.shape[0], 'episode stratum dimension mismatch')
    require(n_resamples > 0 and batch_size > 0, 'positive resample and batch counts required')
    rng = np.random.default_rng(random_seed)
    # F1 requires TP, true support and predicted support, not all K*K entries.
    sufficient = np.stack((np.diagonal(counts, axis1=-2, axis2=-1),
                           counts.sum(-1), counts.sum(-2)), axis=-2).astype(float)
    flat = sufficient.reshape(len(strata), -1)
    scores = np.empty((n_resamples, n_models), dtype=float)
    start = time.perf_counter()
    for lo in range(0, n_resamples, batch_size):
        hi = min(lo+batch_size, n_resamples)
        w = draw_episode_weights(strata, hi-lo, rng)
        sampled = (w.astype(float) @ flat).reshape(hi-lo, n_models, n_seeds, 3, counts.shape[-1])
        den = sampled[..., 1, :] + sampled[..., 2, :]
        per_class = np.divide(2*sampled[..., 0, :], den,
                              out=np.zeros_like(den), where=den > 0)
        # First class macro-average, THEN seed average. Never an ensemble.
        scores[lo:hi] = per_class.mean(-1).mean(-1)
        if progress and (lo == 0 or hi == n_resamples or hi % 1000 == 0):
            print(f'  Bootstrap {hi:,}/{n_resamples:,} | {time.perf_counter()-start:.1f}s', flush=True)
    return scores


def interval_direction(lo, hi):
    if lo > 0:
        return 'comparator_higher'
    if hi < 0:
        return 'TST_higher'
    return 'zero_included_not_equivalence'


def save_results(out, metrics, counts, scores, seeds):
    n_seeds = len(seeds)
    cm = counts.sum(0).reshape(len(MODELS), n_seeds, len(LABELS), len(LABELS))
    f1 = macro_f1(cm)
    pd.DataFrame(metrics).to_csv(out/'metrics_by_seed.csv', index=False, encoding='utf-8-sig')
    rows = []
    for i, name in enumerate(MODELS):
        lo, hi = np.quantile(scores[:, i], [.025, .975])
        rows.append({'model': name, 'n_seeds': n_seeds, 'macro_f1_mean': float(f1[i].mean()),
                     'macro_f1_seed_sd_ddof1': float(f1[i].std(ddof=1)),
                     'episode_ci95_low': float(lo), 'episode_ci95_high': float(hi)})
    pd.DataFrame(rows).to_csv(out/'model_summary.csv', index=False, encoding='utf-8-sig')
    comparisons = []
    contrasts = scores[:, 1:] - scores[:, [0]]
    for j, name in enumerate(MODELS[1:]):
        delta = f1[j+1].mean()-f1[0].mean()
        lo, hi = np.quantile(contrasts[:, j], [.025, .975])
        # 4 contrasts: alpha/(2*4) per tail => marginal coverage 98.75%.
        blo, bhi = np.quantile(contrasts[:, j], [.00625, .99375])
        comparisons.append({'comparator': name, 'reference': 'TST',
            'delta_macro_f1': float(delta), 'delta_percentage_points': float(100*delta),
            'ci95_low': float(lo), 'ci95_high': float(hi),
            'bonferroni_ci98_75_low': float(blo), 'bonferroni_ci98_75_high': float(bhi),
            'ci95_interpretation': interval_direction(lo, hi),
            'familywise_interpretation': interval_direction(blo, bhi)})
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv', index=False, encoding='utf-8-sig')
    norm = cm / cm.sum(-1, keepdims=True)
    norm = norm.mean(1)  # normalize each seed's true row, then average seeds
    confusion_rows = [{'model': model, 'true_label': LABELS[t], 'pred_label': LABELS[p],
                       'seed_mean_row_normalized': float(norm[i,t,p])}
                      for i, model in enumerate(MODELS)
                      for t in range(len(LABELS)) for p in range(len(LABELS))]
    pd.DataFrame(confusion_rows).to_csv(out/'seed_average_row_normalized_confusion.csv',
                                       index=False, encoding='utf-8-sig')
    np.savez_compressed(out/'bootstrap_distribution.npz', model_names=np.array(MODELS),
                        head_seeds=np.array(seeds), seed_mean_macro_f1=scores,
                        comparator_minus_tst=contrasts)
    text = ['# Step 15: paired episode bootstrap', '',
            'Delta = comparator minus TST. Values are Macro-F1 on the 0–1 scale.', '',
            '| Comparator | Delta | 95% percentile CI | Bonferroni 98.75% CI |',
            '|---|---:|---|---|']
    for r in comparisons:
        text.append(f"| {r['comparator']} | {r['delta_macro_f1']:+.8f} | "
                    f"[{r['ci95_low']:+.8f}, {r['ci95_high']:+.8f}] | "
                    f"[{r['bonferroni_ci98_75_low']:+.8f}, {r['bonferroni_ci98_75_high']:+.8f}] |")
    text += ['', 'Intervals condition on the ten observed trained heads and fixed encoder/data.',
             'Head seeds are averaged, not resampled. Seed SD is reported separately.',
             'An interval containing zero does not establish equivalence.',
             'Statistical direction does not by itself establish practical importance.',
             'The four adjusted intervals target nominal 95% family coverage under the bootstrap approximation.',
             'Completed run records, checkpoint/prediction hashes and upstream identities were checked.',
             'No checkpoint inference was rerun; these checks verify the saved artifact chain.',
             'No operational satellite deployment or generator-shift claim follows from this analysis.']
    (out/'analysis_summary.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print('\n'+pd.DataFrame(comparisons)[['comparator','delta_macro_f1','ci95_low','ci95_high']].to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE_DIR)
    parser.add_argument('--tst-root', type=Path, help='Step 12 result directory; relative paths use project-root')
    parser.add_argument('--pooling-root', type=Path, help='Step 13 result directory; relative paths use project-root')
    parser.add_argument('--lstm-root', type=Path, help='Step 14 result directory; relative paths use project-root')
    parser.add_argument('--audit-only', action='store_true', help='Validate all 50 CSVs without bootstrap')
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    require(len(SEEDS) == 10 and len(set(SEEDS)) == 10, 'Formal protocol requires ten unique seeds')
    require(N_BOOTSTRAP >= 20_000, 'Protocol requires at least 20,000 bootstrap replicates')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = root/OUTPUT_REL/f'run_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        'version': '3.0', 'statistical_contract': 'unchanged from v1.0', 'created_utc': stamp, 'project_root': str(root),
        'models': list(MODELS), 'head_seeds': list(SEEDS), 'L': 10,
        'expected_sequences': EXPECTED_TEST_L10_COUNT, 'bootstrap_replicates': N_BOOTSTRAP,
        'bootstrap_rng_seed': BOOTSTRAP_RANDOM_SEED, 'batch_size': BOOTSTRAP_BATCH_SIZE,
        'resampling': 'class-stratified whole episodes with replacement; shared across all runs',
        'estimand': 'mean of per-head-seed sequence-level six-class Macro-F1',
        'seed_treatment': 'fixed trained runs; arithmetic mean, not ensemble; no seed resampling',
        'ci_method': 'percentile; numpy quantile default linear interpolation',
        'primary_ci': .95, 'contrast_sign': 'comparator minus TST',
        'contrast_count': 4, 'bonferroni_marginal_ci': .9875,
        'seed_sd_ddof': 1, 'practical_equivalence_margin': None,
        'conditioning': 'fixed training data, encoder, heads, generation regime and class episode counts',
        'assumption': 'episodes independent/exchangeable within class',
        'numpy': np.__version__, 'pandas': pd.__version__, 'python': platform.python_version(),
        'script_sha256': sha256_file(Path(__file__)),
        'source_templates': PATH_TEMPLATES, 'tst_location_policy': 'Step 12 recorded output_root per seed, including reused main run',
        'audit_only': args.audit_only,
        'observable_effect_use': 'Verify diagnostic alignment only; all common L10 endpoints remain in every comparison.',
    }
    write_json(out/'protocol.json', protocol)
    manifest = {'status': 'RUNNING', 'output_dir': str(out), 'protocol_sha256': sha256_file(out/'protocol.json')}
    write_json(out/'run_manifest.json', manifest)
    audits = []
    try:
        print('Checking completed Steps 12/13/14, recorded seed locations and artifact hashes...', flush=True)
        metadata_path, paths, context = check_inputs(root, SEEDS, args.tst_root, args.pooling_root,
                                                    args.lstm_root, return_context=True)
        track_file(Path(__file__), context['observed'], protocol['script_sha256'])
        metadata_hash = sha256_file(metadata_path)
        ref = load_metadata(metadata_path, context['expected_count'])
        require(metadata_hash == sha256_file(metadata_path), 'Metadata changed during read')
        validate_metadata_population(ref, context)
        effect_path = Path(context['observable_effect_path'])
        effects = read_csv(effect_path)
        effects['sequence_id'] = integers(effects.sequence_id, 'audit sequence_id')
        effects = effects.sort_values('sequence_id').reset_index(drop=True)
        audit_fields = validate_effect_frame(effects, ref, 10)
        track_file(effect_path, context['observed'])
        ref = pd.concat([ref, audit_fields], axis=1)
        protocol.update(expected_sequences=len(ref), source_contract=context['source_contract'],
                        source_roots=context['roots'], common_test_population=context['population'])
        write_json(out/'protocol.json', protocol)
        manifest.update(protocol_sha256=sha256_file(out/'protocol.json'),
                        source_contract=context['source_contract'], source_roots=context['roots'])
        codes, episode_ids = pd.factorize(ref.episode_id, sort=True)
        inventory = ref.groupby('episode_id', sort=True).agg(
            label_id=('label_id','first'), label=('label','first'), sequences=('sequence_id','size'))
        require(np.array_equal(inventory.index.to_numpy(), episode_ids.to_numpy()), 'Episode ordering mismatch')
        inventory.to_csv(out/'episode_inventory.csv', encoding='utf-8-sig')
        manifest.update({'metadata_sha256': metadata_hash, 'metadata_path': str(metadata_path),
                         'n_sequences': len(ref), 'n_episodes': len(inventory),
                         'class_episode_counts': inventory.groupby('label').size().to_dict()})
        predictions, metrics = [], []
        print(f'Project: {root}\nOutput: {out}\nValidating all {len(paths)} prediction CSVs...', flush=True)
        y = ref.label_id.to_numpy(dtype=np.int64)
        for i, (model, seed, path) in enumerate(paths, 1):
            expected = context['expected'][(model, seed)]
            pred, audit = load_prediction(path, ref, expected)
            audit.update(model=model, seed=seed, source_manifest=expected['source_manifest'],
                         source_manifest_sha256=expected['source_manifest_sha256'],
                         **{k: context['source_contract'][k] for k in PROVENANCE_KEYS})
            audits.append(audit)
            predictions.append(pred)
            values = verify_prediction_metrics(y, pred, expected)
            metrics.append({'model': model, 'seed': seed, 'macro_f1': values['macro_f1'],
                            'accuracy': values['accuracy'], 'errors': int((y!=pred).sum())})
            print(f'  [{i:02d}/{len(paths)}] {model} seed={seed}: PASS | F1={metrics[-1]["macro_f1"]:.8f}', flush=True)
        pd.DataFrame(audits).to_csv(out/'input_audit.csv', index=False, encoding='utf-8-sig')
        if args.audit_only:
            recheck_inputs(context['observed'])
            pd.DataFrame(metrics).to_csv(out/'metrics_by_seed.csv', index=False, encoding='utf-8-sig')
            manifest['status'] = 'AUDIT_ONLY_COMPLETE'
        else:
            counts = episode_confusions(y, np.stack(predictions), codes, len(inventory), len(LABELS))
            require(np.array_equal(counts.sum((0,2,3)), np.full(len(paths),len(ref))), 'Confusion count mismatch')
            scores = bootstrap_scores(counts, inventory.label_id.to_numpy(), len(SEEDS), len(MODELS),
                                      N_BOOTSTRAP, BOOTSTRAP_RANDOM_SEED, BOOTSTRAP_BATCH_SIZE)
            recheck_inputs(context['observed'])
            save_results(out, metrics, counts, scores, SEEDS)
            manifest['status'] = 'COMPLETE'
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['input_sha256'] = context['observed']
        manifest['output_sha256'] = {p.name: sha256_file(p) for p in sorted(out.iterdir())
                                     if p.is_file() and p.name != 'run_manifest.json'}
        write_json(out/'run_manifest.json', manifest)
        print(f'\n{manifest["status"]}\nResults: {out}', flush=True)
        return 0
    except Exception as exc:
        manifest.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        write_json(out/'run_manifest.json', manifest)
        if audits:
            pd.DataFrame(audits).to_csv(out/'partial_input_audit.csv', index=False, encoding='utf-8-sig')
        print(f'\nFAILED: {exc}\nDiagnostic folder: {out}\nFinish incomplete upstream runs or correct the reported input; existing results are preserved.',
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
