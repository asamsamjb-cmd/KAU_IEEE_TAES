#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TAES shared-nominal Step 17: fixed Rule baselines versus TST, paired episode CI.

Save beside Steps 11-16 and Run. Requires numpy/pandas only; no GPU/training.
Uses existing local Step 15/16 run folders, not the uploaded ZIP.
Contrasts: Rule_L10 - Rule_L1; TST_L10 - Rule_L10.
20,000 whole-episode stratified draws; shared across 1+1+10 prediction runs.
TST = mean of ten per-seed sequence Macro-F1s, NOT a probability ensemble.
95% percentile CI + Bonferroni 97.5% marginal CI for these two contrasts only.
Revision contrasts are carried forward from the earlier benchmark; no prospective preregistration claim.
Conditional on frozen rules/thresholds, encoder, trained heads and benchmark.
No equivalence, optimal-rule, encoder-necessity or operational-validity claim.

Default: python 17.rule_paired_episode_bootstrap.py
Optional: --project-root PATH --rule-run PATH --step15-run PATH --audit-only
Output: outputs/shared_nominal_v2/17_rule_paired_episode_bootstrap/run_<UTC>/.
All source predictions/configs are read-only. Missing input => FAILED, no CI.
Accepts the existing Step 15 v1.1 / shared_nominal_v2 Step 16 v1 outputs as
well as the expanded v3 records. Existing rules and predictions are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RULE_RUN_REL = 'outputs/shared_nominal_v2/16_observable_rule_baseline'
STEP15_RUN_REL = 'outputs/shared_nominal_v2/15_paired_episode_bootstrap'
METADATA_REL = 'outputs/shared_nominal_v2/tst_sequences_finetuned_main/test_tst_sequence_metadata_L10.csv'
OUTPUT_REL = 'outputs/shared_nominal_v2/17_rule_paired_episode_bootstrap'
IDENTITY_KEYS = ('dataset_id', 'step8_run_id', 'step9_run_id', 'step10_run_id')
SEEDS = tuple(range(20260610, 20260620))
N_BOOTSTRAP = 20_000
RNG_SEED = 20260926
BATCH_SIZE = 100
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
MODELS = ('Rule_L1', 'Rule_L10', 'TST_L10')
KEYS = ('episode_id', 'sequence_start_row', 'sequence_end_row',
        'sequence_start_step', 'sequence_end_step',
        'sequence_start_timestamp', 'sequence_end_timestamp')
INT_META = ('sequence_id', 'label_id', 'sequence_length', 'observable_window_seconds',
            'temporal_context_seconds', 'sequence_start_row', 'sequence_end_row',
            'sequence_start_step', 'sequence_end_step', 'sequence_start_window_id',
            'sequence_end_window_id', 'sequence_num_label_changes', 'sequence_has_label_transition')
META_COLUMNS = (*INT_META, 'source_split', 'episode_id', 'episode_type', 'scenario_id',
                'sequence_start_timestamp', 'sequence_end_timestamp', 'start_label', 'label')


FIELDS = ('observation_stage', 'path_type', 'direction', 'source_role',
          'destination_role', 'protocol_context', 'operational_phase',
          'message_type', 'service_context', 'command_intent',
          'payload_or_parameter_context', 'sequence_freshness_status',
          'control_security_state', 'signal_quality_state', 'tracking_stability',
          'navigation_consistency', 'timing_consistency', 'decoder_status',
          'command_execution_result', 'retry_timeout_state', 'session_state',
          'state_transition_consistency')
META = ['scenario_id', 'episode_id', 'episode_type', 'episode_step',
        'window_id', 'timestamp', 'label']
EFFECT_COLUMNS = ['audit_current_observable_effect', 'audit_history_effect_window_count',
                  'audit_context_effect_window_count', 'audit_effect_stratum']

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

def effects_for_context(effects, length):
    require(length in (1, 10), 'Unexpected diagnostic context length')
    result = effects[EFFECT_COLUMNS].copy()
    if length == 1:
        result[EFFECT_COLUMNS[1]] = 0
        result[EFFECT_COLUMNS[2]] = result[EFFECT_COLUMNS[0]].to_numpy()
        result[EFFECT_COLUMNS[3]] = np.where(result[EFFECT_COLUMNS[0]].eq(1), 'current_changed', 'no_observable_change')
    return result


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_input(path, audit, kind='csv'):
    path = Path(path)
    require(path.is_file(), f'Required input missing: {path}')
    before = sha256(path)
    if kind == 'csv':
        value = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                            dtype={'episode_id': str})
    elif kind == 'json':
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    elif kind == 'npz':
        with np.load(path, allow_pickle=False) as data:
            value = {key: data[key] for key in data.files}
    else:
        raise ValueError(f'Unsupported input format: {kind}')
    require(before == sha256(path), f'Input changed while being read: {path}')
    audit.append({'path': str(path.resolve()), 'sha256': before,
                  'size_bytes': path.stat().st_size})
    return value


def integers(values, context):
    a = pd.to_numeric(np.asarray(values).reshape(-1), errors='raise')
    require(np.isfinite(a).all() and (a == np.floor(a)).all() and
            ((a >= 0) & (a < 2**53)).all(), f'{context}: invalid integer values')
    return np.asarray(a, dtype=np.int64).reshape(np.shape(values))


def macro_f1(cm):
    cm = np.asarray(cm)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(-1) + cm.sum(-2)
    return np.divide(2. * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(-1)


def episode_confusions(y, predictions, episode_codes, n_episodes, n_classes=6):
    counts = np.zeros((n_episodes, len(predictions), n_classes, n_classes), dtype=np.int64)
    for r, pred in enumerate(predictions):
        code = (episode_codes * n_classes + y) * n_classes + pred
        counts[:, r] = np.bincount(code, minlength=n_episodes*n_classes*n_classes).reshape(
            n_episodes, n_classes, n_classes)
    return counts


def draw_weights(strata, batch, rng):
    weights = np.zeros((batch, len(strata)), dtype=np.int64)
    for label in np.unique(strata):
        ix = np.flatnonzero(strata == label)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=batch)
    return weights


def group_scores(per_run):
    require(per_run.shape[-1] == 12, 'Require exactly 1 Rule L1 + 1 Rule L10 + 10 TST runs')
    return np.stack((per_run[..., 0], per_run[..., 1], per_run[..., 2:].mean(-1)), axis=-1)


def bootstrap(counts, strata, n_resamples=N_BOOTSTRAP, seed=RNG_SEED,
              batch_size=BATCH_SIZE, progress=True):
    require(counts.shape[1] == 12 and len(strata) == counts.shape[0], 'Invalid run/episode dimensions')
    require(n_resamples > 0 and batch_size > 0, 'Positive resampling parameters required')
    sufficient = np.stack((np.diagonal(counts, axis1=-2, axis2=-1),
                           counts.sum(-1), counts.sum(-2)), axis=-2).astype(float)
    flat = sufficient.reshape(len(strata), -1)
    scores = np.empty((n_resamples, 3), dtype=float)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for lo in range(0, n_resamples, batch_size):
        hi = min(lo + batch_size, n_resamples)
        weights = draw_weights(strata, hi-lo, rng)
        sampled = (weights.astype(float) @ flat).reshape(hi-lo, 12, 3, counts.shape[-1])
        den = sampled[..., 1, :] + sampled[..., 2, :]
        f1 = np.divide(2 * sampled[..., 0, :], den, out=np.zeros_like(den), where=den > 0).mean(-1)
        scores[lo:hi] = group_scores(f1)
        if progress and (lo == 0 or hi % 1000 == 0 or hi == n_resamples):
            print(f'  Bootstrap {hi:,}/{n_resamples:,} | {time.perf_counter()-started:.1f}s', flush=True)
    return scores


def interval_direction(lo, hi):
    return 'comparator_higher' if lo > 0 else 'reference_higher' if hi < 0 else 'zero_included_not_equivalence'


def compare(point, scores):
    pairs = ((1, 0), (2, 1))
    differences = np.column_stack([scores[:, a] - scores[:, b] for a, b in pairs])
    rows = []
    for j, (a, b) in enumerate(pairs):
        lo, hi = np.quantile(differences[:, j], [.025, .975])
        adjusted_lo, adjusted_hi = np.quantile(differences[:, j], [.0125, .9875])
        delta = float(point[a] - point[b])
        rows.append({'comparator': MODELS[a], 'reference': MODELS[b],
            'delta_macro_f1': delta, 'delta_percentage_points': 100 * delta,
            'ci95_low': float(lo), 'ci95_high': float(hi),
            'bonferroni_ci97_5_low': float(adjusted_lo), 'bonferroni_ci97_5_high': float(adjusted_hi),
            'ci95_interpretation': interval_direction(lo, hi),
            'two_contrast_family_interpretation': interval_direction(adjusted_lo, adjusted_hi)})
    return rows, differences


def columns_present(frame, columns, context):
    require(set(columns) <= set(frame), f'{context}: missing columns {sorted(set(columns)-set(frame))}')
    require(not frame[list(columns)].isna().any().any(), f'{context}: null values')
    for col in columns:
        value = frame[col].astype(str)
        require((value.str.len() > 0).all() and (value == value.str.strip()).all(),
                f'{context}: empty/whitespace values in {col}')


def load_metadata(path, audit, contract):
    frame = read_input(path, audit)
    columns_present(frame, META_COLUMNS, 'Step 10 metadata')
    common = contract['common_endpoint_report']['splits']['test']
    require(len(frame) == common['common_endpoint_count'], 'Step 10 common population count mismatch')
    for col in INT_META:
        frame[col] = integers(frame[col], col)
    require(frame.sequence_id.is_unique, 'Duplicate metadata sequence_id')
    frame = frame.sort_values('sequence_id').reset_index(drop=True)
    require(np.array_equal(frame.sequence_id, np.arange(len(frame))), 'sequence_id must cover 0..N-1')
    require(set(frame.label_id) == set(range(6)), 'Expected six classes')
    require(np.array_equal(frame.label, np.array(LABELS)[frame.label_id]), 'Canonical label mapping mismatch')
    require((frame.source_split == 'test').all() and (frame.sequence_length == 10).all(), 'Expected test L10 metadata')
    for col, value in [('observable_window_seconds', 1), ('temporal_context_seconds', 10),
                       ('sequence_num_label_changes', 0), ('sequence_has_label_transition', 0)]:
        require((frame[col] == value).all(), f'Unexpected {col}')
    require((frame.sequence_end_step - frame.sequence_start_step == 9).all(), 'Invalid L10 step span')
    require((frame.sequence_end_window_id - frame.sequence_start_window_id == 9).all(), 'Invalid L10 window span')
    require((frame.sequence_end_row >= frame.sequence_start_row).all(), 'Invalid row interval')
    start = pd.to_datetime(frame.sequence_start_timestamp, errors='raise', utc=True)
    end = pd.to_datetime(frame.sequence_end_timestamp, errors='raise', utc=True)
    require(((end-start).dt.total_seconds() == 9).all(), 'Invalid L10 timestamp span')
    require((frame.start_label == frame.label).all(), 'Start/terminal labels differ')
    require(not frame.duplicated(['episode_id', 'sequence_end_step']).any(), 'Duplicate test endpoint')
    require(frame.groupby('episode_id').label_id.nunique().eq(1).all(), 'Mixed-label episode')
    ends = np.sort(frame.sequence_end_row.to_numpy(dtype=np.int64))
    require(hashlib.sha256(ends.astype('<i8', copy=False).tobytes()).hexdigest() == common['sorted_end_row_indices_sha256'],
            'Step 10 common endpoint identity mismatch')
    return frame


def load_prediction(path, ref, kind, audit, thresholds=None, expected=None, legacy_rule=False):
    frame = read_input(path, audit)
    require(kind in ('TST', 'Rule_L1', 'Rule_L10'), 'Unknown prediction format')
    true_col, pred_col = ('true_label_id', 'pred_label_id') if kind == 'TST' else ('y_true', 'y_pred')
    require(not legacy_rule or kind in ('Rule_L1', 'Rule_L10'), 'Legacy handling is for saved Rule outputs only')
    required = ('sequence_id', true_col, pred_col, 'true_label', 'pred_label', *KEYS)
    columns_present(frame, required, kind)
    if not legacy_rule:
        columns_present(frame, EFFECT_COLUMNS, kind)
    for key, value in (expected or {}).items():
        columns_present(frame, [key], kind)
        require(frame[key].astype(str).eq(str(value)).all(), f'{kind}: provenance mismatch: {key}')
    if 'common_population' in frame:
        require(integers(frame.common_population, 'common_population').tolist() == [1] * len(frame),
                f'{kind}: common L10 evaluation population is required')
    else:
        require(legacy_rule, f'{kind}: common L10 evaluation population is required')
    require(len(frame) == len(ref), f'{kind}: prediction row count mismatch')
    for col in set(INT_META) & set(frame):
        frame[col] = integers(frame[col], col)
    for col in (true_col, pred_col):
        frame[col] = integers(frame[col], col)
    require(frame.sequence_id.is_unique, f'{kind}: duplicate sequence_id')
    reordered = not np.array_equal(frame.sequence_id, ref.sequence_id)
    frame = frame.sort_values('sequence_id').reset_index(drop=True)
    for col in set(META_COLUMNS) & set(frame):
        require(np.array_equal(frame[col].astype(str), ref[col].astype(str)), f'{kind}: {col} mismatch')
    columns_present(ref, EFFECT_COLUMNS, 'Verified observable-effect reference')
    effect_reference = effects_for_context(ref, 1 if kind == 'Rule_L1' else 10)
    present_effects = set(EFFECT_COLUMNS) & set(frame)
    require(not present_effects or present_effects == set(EFFECT_COLUMNS), f'{kind}: partial observable diagnostics')
    if present_effects:
        for column in EFFECT_COLUMNS:
            require(np.array_equal(frame[column].astype(str), effect_reference[column].astype(str)),
                    f'{kind}: observable diagnostic mismatch: {column}')
    if kind != 'TST':
        columns_present(frame, META_COLUMNS, kind)
        require('model' in frame and (frame.model == kind).all(), f'{kind}: model name mismatch')
    y, pred = frame[true_col].to_numpy(), frame[pred_col].to_numpy()
    require(np.array_equal(y, ref.label_id), f'{kind}: true-label ID mismatch')
    require(np.isin(pred, range(6)).all(), f'{kind}: predicted label outside 0..5')
    require(np.array_equal(frame.true_label, np.array(LABELS)[y]), f'{kind}: true_label mismatch')
    require(np.array_equal(frame.pred_label, np.array(LABELS)[pred]), f'{kind}: pred_label mismatch')
    if 'correct' in frame:
        require(np.array_equal(integers(frame.correct, 'correct'), (y == pred).astype(int)), f'{kind}: correct mismatch')
    if kind == 'TST':
        prob_cols = ['prob_' + label.lower() for label in LABELS]
        columns_present(frame, prob_cols, 'TST probabilities')
        probabilities = frame[prob_cols].to_numpy(float)
        require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
                'TST: invalid probabilities')
        require(np.allclose(probabilities.sum(1), 1, atol=2e-6, rtol=0), 'TST: probability sums differ from 1')
        require(np.array_equal(probabilities.argmax(1), pred), 'TST: probability argmax mismatch')
        if 'confidence' in frame:
            require(np.allclose(frame.confidence.to_numpy(float), probabilities.max(1), atol=2e-6, rtol=0),
                    'TST confidence mismatch')
    else:
        columns_present(frame, ['eligible_attack_count', 'max_score_threshold_ratio'], kind)
        score_cols = ['evidence_' + label for label in LABELS[1:]]
        columns_present(frame, score_cols, kind)
        scores = frame[score_cols].to_numpy(float)
        require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 3)).all(), f'{kind}: invalid evidence scores')
        thresholds = np.asarray(thresholds, dtype=float)
        require(thresholds.shape == (5,) and np.isfinite(thresholds).all() and (thresholds > 0).all(),
                f'{kind}: invalid frozen thresholds')
        ratios = scores / thresholds
        top = ratios.max(1)
        expected = np.where(top >= 1, ratios.argmax(1) + 1, 0)
        require(np.array_equal(pred, expected), f'{kind}: decisions disagree with frozen thresholds/scores')
        require(np.array_equal(integers(frame.eligible_attack_count, 'eligible_attack_count'), (ratios >= 1).sum(1)),
                f'{kind}: eligible attack count mismatch')
        require(np.allclose(frame.max_score_threshold_ratio.to_numpy(float), top, atol=1e-12, rtol=1e-12),
                f'{kind}: score/threshold ratio mismatch')
    audit[-1].update(model=kind, rows=len(frame), rows_reordered=bool(reordered), alignment='PASS',
        common_population_verification='complete Step 10 row/endpoint/label identity',
        saved_observable_diagnostics='verified' if present_effects else 'not emitted by existing Step 16')
    return pred


def source_artifact(directory, manifest, step, name, audit, kind='csv'):
    if step == 15:
        hashes = manifest.get('output_sha256', {})
    else:
        entries = manifest.get('outputs', [])
        require(len({x['file'] for x in entries}) == len(entries), 'Duplicate Step 16 output inventory entries')
        hashes = {x['file']: x['sha256'] for x in entries}
    require(name in hashes, f'Step {step} manifest does not bind {name}')
    value = read_input(directory / name, audit, kind)
    require(audit[-1]['sha256'] == hashes[name], f'Step {step} artifact hash mismatch: {name}')
    return value


def basename(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def validate_existing_step16(root, record, contract, audit):
    """Bind the older producer's recorded inputs without inventing v3 metadata."""
    source = record.get('dataset_contract', {})
    require(source.get('dataset_version') == 'shared_nominal_v2'
            and all(source.get(k) == contract[k] for k in IDENTITY_KEYS),
            'Step 16: existing dataset/encoder/sequence identity differs')
    dataset = root/'dataset/shared_nominal_v2'
    base = root/'outputs/shared_nominal_v2'
    build = read_input(dataset/'build_manifest.json', audit, 'json')
    r9 = read_input(base/'tst_embeddings_finetuned_main/run_manifest.json', audit, 'json')
    r10 = read_input(base/'tst_sequences_finetuned_main/run_manifest.json', audit, 'json')
    needed = {dataset/'build_manifest.json': sha256(dataset/'build_manifest.json'),
        base/'tst_sequences_finetuned_main/sequence_build_report_index.json':
            r10['artifacts']['sequence_build_report_index.json'],
        dataset/'splits/label_mapping.json': build['artifacts']['splits/label_mapping.json']}
    for split in ('train', 'val', 'test'):
        counts = build['counts'][split]
        require(source.get('expected_counts', {}).get(split) ==
                [counts['rows'], counts['episodes'], contract['common_endpoint_report']['splits'][split]['common_endpoint_count']],
                f'Step 16: {split} population differs from the current build')
        name = f'splits/{split}_observable.csv'
        needed[dataset/name] = build['artifacts'][name]
    for split in ('val', 'test'):
        name = f'{split}_window_metadata.csv'
        needed[base/'tst_embeddings_finetuned_main'/name] = r9['artifacts'][name]
        for suffix in ('sequence_metadata_L10.csv', 'sequences_L10.npz'):
            name = f'{split}_tst_{suffix}'
            needed[base/'tst_sequences_finetuned_main'/name] = r10['artifacts'][name]
    entries = record.get('inputs', [])
    for path, expected_hash in needed.items():
        matches = [item for item in entries if Path(item.get('path', '')).resolve() == path.resolve()]
        require(len(matches) == 1 and matches[0].get('sha256') == expected_hash,
                f'Step 16: missing/different recorded input: {path}')
        require(sha256(path) == expected_hash, f'Step 16: input changed: {path}')
        audit.append({'path': str(path.resolve()), 'sha256': expected_hash,
                      'size_bytes': path.stat().st_size})
    return source


def load_sources(root, step15_run, rule_run, ref, metadata_hash, audit, contract):
    s15 = read_input(step15_run / 'run_manifest.json', audit, 'json')
    s16 = read_input(rule_run / 'run_manifest.json', audit, 'json')
    require(s15.get('status') == 'COMPLETE' and s16.get('status') == 'COMPLETE',
            'Both Step 15 and Step 16 source runs must be COMPLETE')
    p15 = source_artifact(step15_run, s15, 15, 'protocol.json', audit, 'json')
    require(audit[-1]['sha256'] == s15.get('protocol_sha256'), 'Step 15 protocol hash mismatch')
    require(p15.get('version') in ('1.1', '3.0'), 'Unsupported Step 15 saved output format')
    legacy15 = p15['version'] == '1.1'
    legacy16 = 'source_contract' not in s16
    for step, record in ((15, s15), (16, s16)):
        if step == 15 and legacy15:
            source = p15.get('dataset_contract', {})
            origin_keys = {'step10_sequence_index', 'step12_multiseed_validation',
                           'step13_pooling_summary', 'step14_lstm_summary'}
            require(source.get('dataset_version') == 'shared_nominal_v2'
                    and source.get('dataset_id') == contract['dataset_id']
                    and source.get('expected_test_l10_count') == len(ref)
                    and set(source.get('sources', {})) == origin_keys
                    and all(value == contract['dataset_id'] for value in source['sources'].values()),
                    'Step 15: existing dataset/cohort/source summaries differ')
            continue
        if step == 16 and legacy16:
            validate_existing_step16(root, record, contract, audit)
            continue
        source = record.get('source_contract', {})
        for key, value in contract.items():
            require(source.get(key) == value, f'Step {step}: different current source contract: {key}')
        for upstream, folder in ((8, 'distilbert_window_baseline'), (9, 'tst_embeddings_finetuned_main'),
                                 (10, 'tst_sequences_finetuned_main')):
            path = root / 'outputs/shared_nominal_v2' / folder / 'run_manifest.json'
            require(source.get('manifest_sha256', {}).get(f'step{upstream}') == sha256(path),
                    f'Step {step}: upstream Step {upstream} completion record changed')
        require(source.get('step10_artifacts', {}).get(Path(METADATA_REL).name) == metadata_hash,
                f'Step {step}: different Step 10 metadata binding')
    require(s15.get('metadata_sha256') == metadata_hash, 'Step 15/current metadata hash mismatch')
    require(s15.get('n_sequences') == len(ref) and s15.get('n_episodes') == ref.episode_id.nunique(),
            'Step 15/current population mismatch')
    for key, value in [('head_seeds', list(SEEDS)), ('bootstrap_replicates', N_BOOTSTRAP),
                       ('bootstrap_rng_seed', RNG_SEED), ('batch_size', BATCH_SIZE), ('L', 10)]:
        require(p15.get(key) == value, f'Step 15 statistical setting differs: {key}')
    old_audit = source_artifact(step15_run, s15, 15, 'input_audit.csv', audit)
    old_metrics = source_artifact(step15_run, s15, 15, 'metrics_by_seed.csv', audit)
    old_summary = source_artifact(step15_run, s15, 15, 'model_summary.csv', audit)
    old_distribution = source_artifact(step15_run, s15, 15, 'bootstrap_distribution.npz', audit, 'npz')
    names = old_distribution['model_names'].astype(str)
    require(np.sum(names == 'TST') == 1, 'Step 15 distribution must identify one TST column')
    require(np.array_equal(old_distribution['head_seeds'], SEEDS), 'Step 15 distribution seeds mismatch')
    samples = old_distribution['seed_mean_macro_f1']
    require(samples.shape == (N_BOOTSTRAP, len(names)) and np.isfinite(samples).all(),
            'Invalid Step 15 bootstrap distribution')
    old_tst = samples[:, int(np.flatnonzero(names == 'TST')[0])]
    require(((old_tst >= 0) & (old_tst <= 1)).all(), 'Invalid Step 15 TST bootstrap scores')

    config = source_artifact(rule_run, s16, 16, 'frozen_rule_config.json', audit, 'json')
    require(audit[-1]['sha256'] == s16.get('frozen_config_sha256'), 'Step 16 frozen config hash mismatch')
    p16 = source_artifact(rule_run, s16, 16, 'protocol.json', audit, 'json')
    spec16 = source_artifact(rule_run, s16, 16, 'rule_spec.json', audit, 'json')
    require(object_digest(p16) == config.get('protocol_sha256') == s16.get('protocol_sha256'),
            'Step 16 canonical protocol digest mismatch')
    require(object_digest(spec16) == config.get('rule_spec_sha256') == p16.get('rule_spec_sha256'),
            'Step 16 rule-spec digest mismatch')
    version16 = 'shared_nominal_v2_semantic_rules_v1' if legacy16 else 'shared_nominal_semantic_rules_v3'
    require(config.get('protocol_version') == version16 == p16.get('version'),
            'Unsupported Step 16 protocol version')
    require(config.get('script_sha256') == s16.get('script_sha256'), 'Step 16 script identity mismatch')
    if legacy16:
        require(p16.get('dataset_contract') == s16.get('dataset_contract')
                and p16.get('expected_counts') == s16['dataset_contract']['expected_counts'],
                'Step 16: rule protocol/dataset contract mismatch')
        require(p16.get('metadata_feature_use') is False
                and p16.get('available_observable_fields') == list(FIELDS),
                'Step 16: unexpected rule inputs')
    else:
        require(config.get('source_contract') == s16.get('source_contract'), 'Rule config/source identity mismatch')
    require(config.get('label_to_id') == {name: i for i, name in enumerate(LABELS)}, 'Rule label mapping mismatch')
    require(set(config.get('models', {})) == {'Rule_L1', 'Rule_L10'}, 'Rule model set mismatch')
    require(config.get('test_inputs_read_before_freeze') is False and s16.get('test_opened_after_freeze') is True,
            'Step 16 freeze-before-test record missing')
    frozen_at = datetime.fromisoformat(config['frozen_at_utc'])
    test_inputs = [x for x in s16['inputs'] if basename(x['path']).startswith('test_')]
    require(test_inputs and all(datetime.fromisoformat(x['read_at_utc']) > frozen_at for x in test_inputs),
            'Step 16 recorded test input precedes config freeze')
    meta_inputs = [x for x in test_inputs if basename(x['path']) == Path(METADATA_REL).name]
    require(len(meta_inputs) == 1 and meta_inputs[0]['sha256'] == metadata_hash,
            'Step 16/current metadata hash mismatch')
    rule_summary = source_artifact(rule_run, s16, 16, 'results_summary.csv', audit)
    for model, length in [('Rule_L1', 1), ('Rule_L10', 10)]:
        if legacy16:
            selected = rule_summary.loc[(rule_summary.split == 'test') & (rule_summary.model == model)]
            require(len(selected) == 1 and selected.iloc[0]['population'] == 'common_L10_endpoints'
                    and int(selected.iloc[0]['n_sequences']) == len(ref)
                    and int(selected.iloc[0]['n_episodes']) == ref.episode_id.nunique(),
                    f'{model}: existing Rule population mismatch')
            continue
        name = f'test_observable_effect_diagnostics_{model}.json'
        diagnostic = source_artifact(rule_run, s16, 16, name, audit, 'json')
        expected = {key: contract[key] for key in IDENTITY_KEYS}
        expected.update(step16_run_id=s16['run_id'], rule_config_sha256=s16['frozen_config_sha256'],
                        rule_spec_sha256=config['rule_spec_sha256'])
        require(diagnostic.get('provenance') == expected
                and diagnostic.get('source_audit') == contract['observability_audit']
                and diagnostic.get('split') == 'test' and diagnostic.get('sequence_length') == length
                and diagnostic.get('rule_context_length') == length
                and diagnostic.get('evaluation_population') == 'common_L10_endpoints',
                f'{model}: Step 16 diagnostic context/source mismatch')
        selected = rule_summary.loc[(rule_summary.split == 'test') & (rule_summary.model == model)]
        require(len(selected) == 1 and Path(selected.iloc[0]['observable_effect_diagnostics']).resolve()
                == (rule_run/name).resolve(), f'{model}: diagnostic file differs from Step 16 summary')
    data_audit = source_artifact(rule_run, s16, 16, 'data_audit.json', audit, 'json')
    require(data_audit['test'].get('alignment_passed') is True and
            data_audit['test'].get('sequences') == len(ref) and
            data_audit['test'].get('episodes') == (s16['dataset_contract']['expected_counts']['test'][1]
                if legacy16 else s16['source_contract']['counts']['test']['episodes']),
            'Step 16 data audit mismatch (raw episodes include short episodes)')
    return {'s16_manifest': s16, 'config': config, 'rule_summary': rule_summary,
            'old_audit': old_audit, 'old_metrics': old_metrics, 'old_summary': old_summary,
            'old_tst_distribution': old_tst, 'source_contract': contract, 's15_manifest': s15,
            'legacy15': legacy15, 'legacy16': legacy16,
            'source_formats': {'step15': p15['version'], 'step16': p16['version'],
                'rule_diagnostics': 'not saved by this producer' if legacy16 else 'verified saved diagnostics',
                'step16_access_scope': ('Existing v1 reads test episode IDs for count validation before calibration; '
                    'the recorded full test evidence reads follow saved thresholds. No stronger no-test-read claim.'
                    if legacy16 else 'recorded test reads after saved thresholds')}}


def unique_tst_rows(frame, context):
    columns_present(frame, ['model', 'seed'], context)
    rows = frame.loc[frame.model == 'TST'].copy()
    rows['seed'] = integers(rows.seed, context + ':seed')
    require(len(rows) == 10 and rows.seed.is_unique and set(rows.seed) == set(SEEDS),
            f'{context}: require exactly the original ten TST seeds')
    return rows.set_index('seed')


def collect_predictions(root, rule_run, ref, sources, audit):
    old_audit = unique_tst_rows(sources['old_audit'], 'Step 15 input audit')
    paths = [Path(old_audit.loc[seed, 'path']) for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    require(not missing, 'All ten TST prediction files are required. Missing:\n' + '\n'.join(missing))
    old_metrics = unique_tst_rows(sources['old_metrics'], 'Step 15 per-seed metrics')
    predictions, rows = [], []
    y = ref.label_id.to_numpy(dtype=np.int64)
    items = [(name, None, rule_run / f'test_predictions_{name}.csv') for name in MODELS[:2]]
    items += [('TST', seed, path) for seed, path in zip(SEEDS, paths)]
    rule_hashes = {x['file']: x['sha256'] for x in sources['s16_manifest']['outputs']}
    for i, (name, seed, path) in enumerate(items, 1):
        threshold = None if name == 'TST' else sources['config']['models'][name]['thresholds']
        expected = {key: sources['source_contract'][key] for key in IDENTITY_KEYS}
        if name == 'TST':
            seed_manifest = (path.parents[2] / 'run_manifest.json' if sources['legacy15']
                             else Path(old_audit.loc[seed, 'source_manifest']))
            record = read_input(seed_manifest, audit, 'json')
            require(record.get('status') == 'COMPLETE', f'Seed {seed}: incomplete training run')
            if not sources['legacy15']:
                require(audit[-1]['sha256'] == old_audit.loc[seed, 'source_manifest_sha256'],
                        f'Seed {seed}: changed training run')
            audit[-1]['binding'] = ('derived from the saved prediction path; verified against current upstream inputs'
                                    if sources['legacy15'] else 'recorded in Step 15 input audit')
            require(record.get('settings', {}).get('DEFAULT_RANDOM_SEED') == seed,
                    f'Seed {seed}: training seed mismatch')
            require(path.resolve() == (seed_manifest.parent / 'L10/predictions/test_predictions_L10.csv').resolve(),
                    f'Seed {seed}: prediction is outside its training run')
            require(record.get('artifacts', {}).get('L10/predictions/test_predictions_L10.csv')
                    == old_audit.loc[seed, 'sha256'], f'Seed {seed}: unbound predictions')
            source = record.get('source_contract', {})
            for key, value in sources['source_contract'].items():
                require(source.get(key) == value, f'Seed {seed}: different current input contract: {key}')
            for upstream, folder in ((8, 'distilbert_window_baseline'), (9, 'tst_embeddings_finetuned_main'),
                                     (10, 'tst_sequences_finetuned_main')):
                manifest_path = root / 'outputs/shared_nominal_v2' / folder / 'run_manifest.json'
                require(source.get('manifest_sha256', {}).get(f'step{upstream}') == sha256(manifest_path),
                        f'Seed {seed}: upstream Step {upstream} completion record changed')
            require(source.get('step10_artifacts', {}).get(Path(METADATA_REL).name)
                    == sources['s15_manifest']['metadata_sha256'], f'Seed {seed}: different Step 10 metadata')
            if not sources['legacy15']:
                require(source == sources['s15_manifest']['source_contract'],
                        f'Seed {seed}: different Step 15 training input contract')
            expected['step11_run_id'] = record['run_id']
            diagnostic_name = 'L10/reports/observable_effect_diagnostics_test_L10.json'
            diagnostic = read_input(seed_manifest.parent/diagnostic_name, audit, 'json')
            require(audit[-1]['sha256'] == record.get('artifacts', {}).get(diagnostic_name)
                    and diagnostic.get('provenance') == expected
                    and diagnostic.get('source_audit') == sources['source_contract']['observability_audit']
                    and diagnostic.get('split') == 'test' and diagnostic.get('sequence_length') == 10,
                    f'Seed {seed}: TST diagnostic binding/context mismatch')
        elif sources['legacy16']:
            # v1 binds predictions to its dataset and frozen rules through its
            # manifest/config/input inventory; it did not emit provenance columns.
            expected = {}
        else:
            expected.update(step16_run_id=sources['s16_manifest']['run_id'],
                            rule_config_sha256=sources['s16_manifest']['frozen_config_sha256'],
                            rule_spec_sha256=sources['config']['rule_spec_sha256'],
                            rule_context_length=1 if name == 'Rule_L1' else 10)
        pred = load_prediction(path, ref, name, audit, threshold, expected,
                               legacy_rule=sources['legacy16'] and name != 'TST')
        expected_hash = old_audit.loc[seed, 'sha256'] if name == 'TST' else rule_hashes[path.name]
        require(audit[-1]['sha256'] == expected_hash, f'Prediction differs from completed source run: {path}')
        audit[-1]['seed'] = seed
        cm = np.bincount(y*6+pred, minlength=36).reshape(6, 6)
        f1, accuracy = float(macro_f1(cm)), float((y == pred).mean())
        if name == 'TST':
            require(abs(f1-float(old_metrics.loc[seed, 'macro_f1'])) < 1e-12
                    and abs(accuracy-float(old_metrics.loc[seed, 'accuracy'])) < 1e-12,
                    f'Step 15 TST seed {seed} point metric differs')
        else:
            summary = sources['rule_summary']
            summary = summary.loc[(summary.split == 'test') & (summary.model == name)]
            require(len(summary) == 1 and abs(f1-float(summary.iloc[0].macro_f1)) < 1e-12 and
                    abs(accuracy-float(summary.iloc[0].accuracy)) < 1e-12, f'{name}: Step 16 point metrics differ')
        predictions.append(pred)
        rows.append({'model': 'TST_L10' if name == 'TST' else name, 'seed': seed,
                     'macro_f1': f1, 'accuracy': accuracy, 'errors': int((y != pred).sum())})
        print(f'  [{i:02d}/12] {name}' + (f' seed={seed}' if seed is not None else '') +
              f': PASS | Macro-F1={f1:.8f}', flush=True)
    old_summary = sources['old_summary'].loc[sources['old_summary'].model == 'TST']
    require(len(old_summary) == 1, 'Step 15 TST model summary missing/duplicated')
    tst_f1 = np.array([row['macro_f1'] for row in rows[2:]])
    require(abs(tst_f1.mean()-float(old_summary.iloc[0].macro_f1_mean)) < 1e-12 and
            abs(tst_f1.std(ddof=1)-float(old_summary.iloc[0].macro_f1_seed_sd_ddof1)) < 1e-12,
            'Step 15 TST mean/SD mismatch')
    return np.stack(predictions), rows


def save_results(out, counts, scores, run_metrics):
    cm = counts.sum(0)
    per_run_f1 = macro_f1(cm)
    point = group_scores(per_run_f1)
    summaries = []
    for j, name in enumerate(MODELS):
        lo, hi = np.quantile(scores[:, j], [.025, .975])
        summaries.append({'model': name, 'n_prediction_runs': 10 if j == 2 else 1,
            'macro_f1': float(point[j]),
            'head_seed_sd_ddof1': float(per_run_f1[2:].std(ddof=1)) if j == 2 else '',
            'episode_ci95_low': float(lo), 'episode_ci95_high': float(hi),
            'estimator': 'mean_of_ten_per_seed_F1' if j == 2 else 'single_deterministic_rule_F1'})
    pd.DataFrame(summaries).to_csv(out / 'model_summary.csv', index=False, encoding='utf-8-sig')
    comparisons, deltas = compare(point, scores)
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparisons.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(out / 'bootstrap_distribution.npz', model_names=np.array(MODELS),
                        macro_f1=scores, head_seeds=np.array(SEEDS),
                        contrast_names=np.array(['Rule_L10_minus_Rule_L1', 'TST_L10_minus_Rule_L10']),
                        delta_macro_f1=deltas)
    norm = np.divide(cm, cm.sum(-1, keepdims=True), out=np.zeros_like(cm, dtype=float),
                     where=cm.sum(-1, keepdims=True) != 0)
    grouped = np.stack((norm[0], norm[1], norm[2:].mean(0)))
    pd.DataFrame([{'model': model, 'true_label': LABELS[t], 'pred_label': LABELS[p],
                   'run_mean_row_normalized': float(grouped[j, t, p])}
                  for j, model in enumerate(MODELS) for t in range(6) for p in range(6)]).to_csv(
                      out / 'row_normalized_confusion.csv', index=False, encoding='utf-8-sig')
    lines = ['# Step 17: fixed Rule comparisons', '',
             'Delta = comparator minus reference. F1 and intervals use the 0-1 scale.', '',
             '| Comparator | Reference | Delta | 95% CI | Bonferroni 97.5% CI |',
             '|---|---|---:|---|---|']
    for row in comparisons:
        lines.append(f"| {row['comparator']} | {row['reference']} | {row['delta_macro_f1']:+.8f} | "
                     f"[{row['ci95_low']:+.8f}, {row['ci95_high']:+.8f}] | "
                     f"[{row['bonferroni_ci97_5_low']:+.8f}, {row['bonferroni_ci97_5_high']:+.8f}] |")
    lines += ['', 'The contrast definitions are carried forward from the earlier revision analysis; no prospective preregistration is claimed.',
              'Rules/thresholds and the trained encoder/heads remain fixed; no selection is repeated in bootstrap.',
              'Resampling is shared whole episodes within class. Head seeds are fixed and averaged after F1.',
              'Rule L10 includes mean evidence, recurrence support and separate validation calibration.',
              'It is not a pure mean-pooling ablation. One rule baseline cannot represent every possible rule.',
              'Neither contrast establishes that DistilBERT or TST is essential.',
              'The Bonferroni family contains only these two contrasts, not all revision analyses.',
              'A CI containing zero is not evidence of equivalence; no p-values are computed.',
              'Intervals require approximately independent/exchangeable episodes within each class.',
              'No encoder-training, validation-selection or generator uncertainty is included.',
              'Prediction hashes/alignment do not independently certify checkpoint provenance or operational validity.',
              '',
              'Do not retune Rule thresholds or select a new aggregator because of these test scores.']
    (out / 'analysis_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n' + pd.DataFrame(comparisons)[['comparator', 'reference', 'delta_macro_f1',
                                            'ci95_low', 'ci95_high']].to_string(index=False), flush=True)


def current_contract(root, audit):
    """Bind the paired analysis to the current complete 8/9/10 handoff."""
    base = root / 'outputs/shared_nominal_v2'
    runs = {}
    for step, name in ((8, 'distilbert_window_baseline'), (9, 'tst_embeddings_finetuned_main'),
                       (10, 'tst_sequences_finetuned_main')):
        runs[step] = read_input(base / name / 'run_manifest.json', audit, 'json')
        require(runs[step].get('status') == 'COMPLETE', f'Step {step} must be COMPLETE')
    r8, r9, r10 = (runs[x] for x in (8, 9, 10))
    require(r8.get('dataset_id') and r8['dataset_id'] == r9.get('dataset_id') == r10.get('dataset_id'),
            'Mixed dataset IDs')
    require(r8['run_id'] == r9.get('step8_run_id') == r10.get('step8_run_id')
            and r9['run_id'] == r10.get('step9_run_id'), 'Mixed encoder/embedding runs')
    dataset = root/'dataset/shared_nominal_v2'
    build = read_input(dataset/'build_manifest.json', audit, 'json')
    build_hash = audit[-1]['sha256']
    source9, source10 = r9.get('source_contract', {}), r10.get('source', {})
    observable_audit = source10.get('observability_audit', {})
    validate_nominal_contract(build, source9, r8, observable_audit, build_hash, dataset)
    require(build.get('dataset_id') == r8['dataset_id']
            and build['dataset_id'] == object_digest({'policy': build['policy'], 'sources': build.get('sources')}),
            'Step 7 shared nominal dataset identity mismatch')
    require(source9.get('input_sha256') == r8.get('input_sha256')
            and source9.get('step8_manifest_sha256') == sha256(base/'distilbert_window_baseline/run_manifest.json')
            and source10.get('step9_manifest_sha256') == sha256(base/'tst_embeddings_finetuned_main/run_manifest.json'),
            'Step 8/9/10 source chain mismatch')
    index_path = (root / METADATA_REL).parent / 'sequence_build_report_index.json'
    index = read_input(index_path, audit, 'json')
    require(audit[-1]['sha256'] == r10.get('artifacts', {}).get(index_path.name)
            and index.get('status') == 'COMPLETE' and index.get('run_id') == r10['run_id'],
            'Step 10 index identity/hash mismatch')
    require(index.get('dataset_id') == r8['dataset_id'] and index.get('step8_run_id') == r8['run_id']
            and index.get('step9_run_id') == r9['run_id']
            and index.get('observability_audit') == observable_audit, 'Step 10 index source mismatch')
    report_path = index_path.with_name('sequence_build_report_L10.json')
    report = read_input(report_path, audit, 'json')
    require(audit[-1]['sha256'] == r10.get('artifacts', {}).get(report_path.name), 'Step 10 L10 report mismatch')
    for key in ('dataset_id', 'step8_run_id', 'step9_run_id', 'run_id', 'observability_audit'):
        require(report.get(key) == index.get(key), f'Step 10 L10 report differs: {key}')
    effect_path = index_path.with_name('test_tst_sequence_observable_effects_L10.csv')
    detail = report.get('splits', {}).get('test', {})
    require(detail.get('sequence_observable_effects_sha256') == r10.get('artifacts', {}).get(effect_path.name)
            and isinstance(detail.get('sequence_observable_effects_sha256'), str)
            and Path(detail.get('sequence_observable_effects_csv_path', '')).resolve() == effect_path.resolve(),
            'Step 10 L10 diagnostic artifact record mismatch')
    require(index.get('common_endpoint_report', {}).get('reference_length') == 10,
            'The common evaluation reference must be L10')
    require(sha256(root / METADATA_REL) == r10.get('artifacts', {}).get(Path(METADATA_REL).name),
            'Step 10 metadata hash mismatch')
    return {'dataset_id': r8['dataset_id'], 'step8_run_id': r8['run_id'],
            'step9_run_id': r9['run_id'], 'step10_run_id': r10['run_id'],
            'common_endpoint_report': index['common_endpoint_report'], 'observability_audit': observable_audit}



def attach_observable_effect_reference(root, ref, audit):
    directory = (root/METADATA_REL).parent
    run = read_input(directory/'run_manifest.json', audit, 'json')
    path = directory/'test_tst_sequence_observable_effects_L10.csv'
    effects = read_input(path, audit)
    require(audit[-1]['sha256'] == run.get('artifacts', {}).get(path.name),
            'Step 10 observable-effect sidecar differs from the completed run')
    columns_present(effects, ['sequence_id'], 'Step 10 observable-effect sidecar')
    effects['sequence_id'] = integers(effects.sequence_id, 'audit sequence_id')
    effects = effects.sort_values('sequence_id').reset_index(drop=True)
    fields = validate_effect_frame(effects, ref, 10)
    return pd.concat([ref, fields], axis=1)


def resolve_completed_run(root, value, default, contract):
    if value is not None:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    candidates = []
    for path in sorted((root / default).glob('run_*/run_manifest.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8-sig'))
            if record.get('status') != 'COMPLETE':
                continue
            source = record.get('source_contract', {})
            matches = all(source.get(k) == contract[k] for k in IDENTITY_KEYS)
            if not source and default == RULE_RUN_REL:
                source = record.get('dataset_contract', {})
                matches = (source.get('dataset_version') == 'shared_nominal_v2'
                           and all(source.get(k) == contract[k] for k in IDENTITY_KEYS))
            elif not source and default == STEP15_RUN_REL:
                protocol_path = path.with_name('protocol.json')
                digest = sha256(protocol_path)
                protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
                source = protocol.get('dataset_contract', {})
                matches = (protocol.get('version') == '1.1'
                           and digest == record.get('protocol_sha256')
                           == record.get('output_sha256', {}).get('protocol.json')
                           and source.get('dataset_version') == 'shared_nominal_v2'
                           and source.get('dataset_id') == contract['dataset_id'])
        except (OSError, ValueError, TypeError, AttributeError):
            # Incomplete/unreadable folders are not eligible completed sources.
            continue
        if matches:
            candidates.append(path.parent)
    require(len(candidates) == 1,
            f'Expected one matching COMPLETE source under {root / default}, found {len(candidates)}. '
            'Specify --step15-run and/or --rule-run; results are never used to choose a run.')
    return candidates[0]


def run(root, rule_run=None, step15_run=None, audit_only=False):
    root = Path(root).expanduser().resolve()
    print(f'Step 17: checking shared-nominal inputs and completed Step 15/16 results.\n'
          f'Project root: {root}', flush=True)
    audit = []
    contract = current_contract(root, audit)
    rule_run = resolve_completed_run(root, rule_run, RULE_RUN_REL, contract)
    step15_run = resolve_completed_run(root, step15_run, STEP15_RUN_REL, contract)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = root / OUTPUT_REL / ('run_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = {'version': 'step17_rule_paired_shared_nominal_v4', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'models': list(MODELS), 'contrasts': ['Rule_L10_minus_Rule_L1', 'TST_L10_minus_Rule_L10'],
        'source_step15': str(step15_run), 'source_step16': str(rule_run),
        'source_contract': contract, 'class_order': list(LABELS),
        'head_seeds': list(SEEDS), 'n_bootstrap': N_BOOTSTRAP, 'rng_seed': RNG_SEED,
        'batch_size': BATCH_SIZE, 'ci_method': 'percentile, numpy linear quantile',
        'ci95_tails': [.025, .975], 'bonferroni_two_contrast_tails': [.0125, .9875],
        'family_size': 2, 'nominal_family_coverage': .95, 'bonferroni_marginal_coverage': .975,
        'sampling': 'whole episode with replacement within class; shared across all 12 runs',
        'statistic': 'sequence-level 6-class Macro-F1; TST arithmetic mean of 10 per-seed F1s',
        'seed_treatment': 'fixed; no seed resampling, no deterministic-rule seed duplication',
        'condition_on': 'frozen rules/thresholds, training data, encoder, heads, generator realization and class counts',
        'study_timing': 'carried-forward revision contrasts; no prospective preregistration claim',
        'assumption': 'episodes approximately independent/exchangeable within class',
        'equivalence_margin': None, 'p_values': False, 'audit_only': bool(audit_only),
        'observable_effect_use': 'diagnostic alignment only; every common L10 endpoint remains in both contrasts',
        'script_sha256': sha256(Path(__file__)), 'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__}
    write_json(out / 'protocol.json', protocol)
    manifest = {'status': 'RUNNING', 'started_at_utc': protocol['created_at_utc'],
                'source_contract': contract,
                'output_dir': str(out), 'protocol_sha256': sha256(out / 'protocol.json'),
                'source_step15': str(step15_run), 'source_step16': str(rule_run)}
    write_json(out / 'run_manifest.json', manifest)
    try:
        require(len(SEEDS) == 10 and len(set(SEEDS)) == 10, 'Ten unique TST seeds required')
        require(N_BOOTSTRAP >= 20_000 and BATCH_SIZE > 0, 'Formal protocol requires >=20000 draws')
        print(f'Step 17: source integrity and endpoint audit.\nOutput: {out}', flush=True)
        ref = load_metadata(root / METADATA_REL, audit, contract)
        metadata_hash = audit[-1]['sha256']
        ref = attach_observable_effect_reference(root, ref, audit)
        sources = load_sources(root, step15_run, rule_run, ref, metadata_hash, audit, contract)
        manifest['source_formats'] = sources['source_formats']
        print(f"Saved output formats: Step 15 {sources['source_formats']['step15']}; "
              f"Step 16 {sources['source_formats']['step16']}", flush=True)
        predictions, run_metrics = collect_predictions(root, rule_run, ref, sources, audit)
        pd.DataFrame(run_metrics).to_csv(out / 'metrics_by_run.csv', index=False, encoding='utf-8-sig')
        codes, episode_ids = pd.factorize(ref.episode_id, sort=True)
        inventory = ref.groupby('episode_id', sort=True).agg(label_id=('label_id', 'first'),
                    label=('label', 'first'), sequences=('sequence_id', 'size'))
        require(np.array_equal(episode_ids, inventory.index), 'Episode ordering mismatch')
        inventory.to_csv(out / 'episode_inventory.csv', encoding='utf-8-sig')
        manifest.update(metadata_sha256=metadata_hash, n_sequences=len(ref), n_episodes=len(inventory),
                        class_episode_counts=inventory.groupby('label').size().to_dict())
        if audit_only:
            manifest['status'] = 'AUDIT_ONLY_COMPLETE'
        else:
            counts = episode_confusions(ref.label_id.to_numpy(dtype=np.int64), predictions,
                                         codes, len(inventory), 6)
            require(np.array_equal(counts.sum((0, 2, 3)), np.full(12, len(ref))), 'Confusion count mismatch')
            scores = bootstrap(counts, inventory.label_id.to_numpy(dtype=np.int64),
                               N_BOOTSTRAP, RNG_SEED, BATCH_SIZE)
            error = float(np.max(np.abs(scores[:, 2] - sources['old_tst_distribution'])))
            manifest['step15_tst_bootstrap_max_abs_difference'] = error
            require(error <= 1e-12, f'Step 15 TST bootstrap reproduction failed: max error={error}. '
                    'Check source run, cohort, RNG/batch settings and NumPy version; do not tune model/rules.')
            print(f'Step 15 TST bootstrap reproduction: PASS (max abs difference {error:.3g})', flush=True)
            save_results(out, counts, scores, run_metrics)
            manifest['status'] = 'COMPLETE'
        for item in audit:
            require(sha256(item['path']) == item['sha256'], f"Input changed during analysis: {item['path']}")
        require(sha256(Path(__file__)) == protocol['script_sha256'], 'Step 17 code changed during analysis')
        write_json(out / 'input_audit.json', audit)
        manifest['elapsed_seconds'] = time.perf_counter() - started
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['output_sha256'] = {p.name: sha256(p) for p in sorted(out.iterdir())
                                     if p.is_file() and p.name != 'run_manifest.json'}
        archive = out.with_suffix('.zip')
        manifest['review_zip'] = str(archive)
        manifest['archive_note'] = 'Derived transfer container; output inventory covers its member artifacts, not the ZIP itself.'
        write_json(out / 'run_manifest.json', manifest)
        # The completed manifest is included; archive contains this run only.
        with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as z:
            for path in sorted(out.iterdir()):
                if path.is_file():
                    z.write(path, arcname=path.name)
        print(f"\n{manifest['status']}\nResults: {out}\nReview ZIP: {archive}", flush=True)
        return out
    except Exception as error:
        manifest.update(status='FAILED', error_type=type(error).__name__, error=str(error),
                        failed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter()-started)
        write_json(out / 'input_audit.json', audit)
        (out / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(out / 'run_manifest.json', manifest)
        print(f'FAILED: {error}\nDiagnostic folder: {out}\nDo not rerun training or alter frozen Rule thresholds.',
              file=sys.stderr, flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE_DIR)
    parser.add_argument('--rule-run', type=Path, default=None, help='Explicit Step16 run directory; never selects latest')
    parser.add_argument('--step15-run', type=Path, default=None, help='Explicit Step15 run directory')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        run(args.project_root, args.rule_run, args.step15_run, args.audit_only)
    except Exception as error:
        # Input/run discovery can fail before run() creates its diagnostic folder.
        # Always expose that failure to the CLI instead of silently returning 1.
        print(f'Step 17 FAILED: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
