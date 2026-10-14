#!/usr/bin/env python3
"""Step 16: validation-calibrated semantic rules on the frozen benchmark (DATASET_VERSION tree).

Put this file beside Steps 11-15 and Run in PyCharm. Requires numpy, pandas.
Existing datasets/checkpoints/results are read-only. No GPU is used.
Default: calibrate on common L10 validation endpoints, save a frozen config,
THEN open test inputs and evaluate Rule_L1 and Rule_L10 at identical endpoints.
No actual A-prime rule result was available when this script was developed.

Outputs: outputs/16_observable_rule_baseline/run_<UTC>/
Optional: --project-root PATH | --calibrate-only | --frozen-config JSON_PATH
Rule scores are heuristic evidence, NOT probabilities. No CI is computed here.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_VERSION = 'shared_nominal_v2'          # benchmark version; every input must come from this tree
DATASET_DIR = f'dataset/{DATASET_VERSION}'
OUT_ROOT = f'outputs/{DATASET_VERSION}'
PROTOCOL_VERSION = f'{DATASET_VERSION}_semantic_rules_v1'
LENGTH = 10
# Expected counts are resolved at run time from the Step-7 build manifest and the
# Step-10 report index of the same tree (dataset-id guard); never hard-coded.
EXPECTED = {}
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
ATTACKS = LABELS[1:]
FIELDS = ('observation_stage', 'path_type', 'direction', 'source_role',
          'destination_role', 'protocol_context', 'operational_phase',
          'message_type', 'service_context', 'command_intent',
          'payload_or_parameter_context', 'sequence_freshness_status',
          'control_security_state', 'signal_quality_state', 'tracking_stability',
          'navigation_consistency', 'timing_consistency', 'decoder_status',
          'command_execution_result', 'retry_timeout_state', 'session_state',
          'state_transition_consistency')
KEYS = ['episode_id', 'episode_step', 'window_id']
META = ['scenario_id', 'episode_id', 'episode_type', 'episode_step',
        'window_id', 'timestamp', 'label']
GRID = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.1)
STARTS = (0.5, 1.5)
MAX_SWEEPS = 3

# Every membership set is prespecified from observable meaning, not test counts.
TOKENS = {
 'command_intent': ['STATUS_REQUEST', 'TIME_SYNC', 'CONFIG_UPDATE', 'NOMINAL_CONTROL',
   'HOUSEKEEPING', 'ORBIT_CONTROL', 'POLICY_UPDATE', 'ROUTING_UPDATE', 'ROUTE_UPDATE',
   'SESSION_REAUTH', 'PAYLOAD_CONFIG', 'MODE_CHANGE', 'ATTITUDE_ADJUST', 'CONFIG_CHANGE',
   'LINK_RECONFIGURATION', 'PAYLOAD_DISABLE', 'RESET', 'GCS_GATEWAY_SYNC'],
 'command_message': ['TELECOMMAND', 'COMMAND_UPLINK', 'COMMAND_VALIDATION_EVENT',
   'PARTIAL_COMMAND_FRAME', 'UNKNOWN_COMMAND'],
 'bad_signal': ['DEGRADED', 'SEVERELY_DEGRADED', 'LOST'],
 'severe_signal': ['SEVERELY_DEGRADED', 'LOST'],
 'bad_decoder': ['CRC_FAIL', 'DECODE_FAIL', 'FRAME_LOSS', 'PARTIAL_DECODE', 'SYNC_LOSS'],
 'usable_decoder': ['DECODE_OK', 'DECODE_WARN'],
 'usable_signal': ['NOMINAL', 'FLUCTUATING', 'DEGRADED'],
 'bad_tracking': ['UNSTABLE', 'CYCLE_SLIP_LIKE', 'LOCK_LOSS'],
 'bad_retry': ['RETRY_INCREASED', 'TIMEOUT', 'BURST_TIMEOUT', 'LINK_OUTAGE'],
 'bad_navigation': ['DRIFTING', 'INCONSISTENT', 'JUMP_DETECTED'],
 'bad_timing': ['DRIFTING', 'INCONSISTENT', 'OFFSET_JUMP'],
 'failed_security': ['AUTH_FAILED', 'AUTH_MISSING', 'INTEGRITY_FAILED', 'UNAUTHORIZED'],
 'weak_security': ['AUTH_FAILED', 'AUTH_MISSING', 'INTEGRITY_FAILED', 'UNAUTHORIZED',
   'AUTH_UNVERIFIED', 'INTEGRITY_UNVERIFIED', 'SOURCE_UNVERIFIED', 'VALID_LOOKING_ANOMALOUS'],
 'valid_looking': ['AUTH_VALID', 'VALID_LOOKING_ANOMALOUS'],
 'bad_payload': ['CONTEXT_MISMATCH_PARAMETER', 'OUT_OF_POLICY_PARAMETER',
   'PARAMETER_RANGE_VIOLATION', 'UNEXPECTED_TARGET_PARAMETER', 'MALFORMED_PARAMETER',
   'UNAUTHORIZED_PARAMETER'],
 'bad_session': ['ABNORMAL', 'DEGRADED', 'EXPIRED', 'RESET', 'NO_SESSION'],
 'bad_transition': ['DELAYED_TRANSITION', 'INCOMPLETE_TRANSITION', 'INCONSISTENT',
   'FORBIDDEN_TRANSITION', 'UNEXPECTED_TRANSITION'],
 'severe_transition': ['FORBIDDEN_TRANSITION', 'UNEXPECTED_TRANSITION', 'INCONSISTENT'],
 'old_freshness': ['DUPLICATE', 'STALE', 'REPLAYED'],
 'uncertain_freshness': ['OUT_OF_ORDER', 'UNKNOWN', 'MISSING_SEQUENCE'],
 'held_execution': ['HELD', 'BLOCKED', 'REJECTED', 'FAILED', 'TIMEOUT'],
 'accepted_execution': ['ACCEPTED', 'EXECUTED', 'QUEUED'],
 'sensitive_intent': ['CONFIG_UPDATE', 'CONFIG_CHANGE', 'PAYLOAD_CONFIG', 'MODE_CHANGE',
   'ATTITUDE_ADJUST', 'ORBIT_CONTROL', 'POLICY_UPDATE', 'ROUTING_UPDATE', 'ROUTE_UPDATE',
   'LINK_RECONFIGURATION', 'PAYLOAD_DISABLE', 'RESET'],
 'recurrence_excluded_intent': ['STATUS_REQUEST', 'HOUSEKEEPING', 'TIME_SYNC', 'NO_COMMAND'],
 'command_phase': ['COMMAND_SESSION', 'CONTACT_WINDOW'],
 'command_path': ['GATEWAY_LEO', 'LEO_LEO'],
 'command_protocol': ['CCSDS_TTC', 'CCSDS_ISL'],
 'command_direction': ['GROUND_TO_SPACE', 'SPACE_TO_SPACE'],
 'ground_source': ['GROUND_GATEWAY', 'GROUND_CONTROL_SERVICE', 'GCS_OPERATOR',
                   'MAINTENANCE_OPERATOR'],
}

CLAUSES = {
 'RF_LINK_JAMMING': ['bad_signal OR bad_decoder',
   '(bad_signal OR bad_tracking) AND (bad_decoder OR bad_retry OR link_loss)',
   '(severe_signal AND bad_decoder) OR (bad_signal AND bad_decoder AND bad_retry)'],
 'GNSS_SPOOFING': ['bad_navigation OR bad_timing',
   '(bad_navigation OR bad_timing) AND usable_signal AND usable_decoder',
   'bad_navigation AND bad_timing'],
 'UNAUTHORIZED_TELECOMMAND_INJECTION': [
   'command AND (weak_security OR bad_payload OR (context_off AND bad_session))',
   'command AND ((weak_security AND (bad_payload OR bad_transition OR bad_session)) OR (context_off AND bad_payload))',
   'command AND failed_security'],
 'TELECOMMAND_REPLAY': ['command AND (old_freshness OR (uncertain_freshness AND corroboration))',
   'command AND old_freshness AND corroboration', 'command AND explicit_replayed'],
 'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE': [
   'command AND valid_looking AND (bad_payload OR bad_transition)',
   'command AND valid_looking AND ((bad_payload AND bad_transition) OR (sensitive_intent AND bad_transition AND bad_session))',
   'command AND valid_looking AND accepted_execution AND ((severe_transition AND (bad_payload OR sensitive_intent)) OR (bad_payload AND bad_transition AND ground_source))'],
}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def window_evidence(frame):
    """Only explicitly allowlisted observable columns reach the rule engine."""
    require(set(FIELDS) <= set(frame.columns), 'Missing observable fields')
    x = {k: frame[k].to_numpy(dtype=str) for k in FIELDS}

    def isin(field, group):
        return np.isin(x[field], TOKENS[group])

    command = (isin('command_intent', 'command_intent') |
               isin('message_type', 'command_message'))
    signal = isin('signal_quality_state', 'bad_signal')
    severe = isin('signal_quality_state', 'severe_signal')
    decoder = isin('decoder_status', 'bad_decoder')
    tracking = isin('tracking_stability', 'bad_tracking')
    retry = isin('retry_timeout_state', 'bad_retry')
    nav = isin('navigation_consistency', 'bad_navigation')
    timing = isin('timing_consistency', 'bad_timing')
    weak = isin('control_security_state', 'weak_security')
    failed = isin('control_security_state', 'failed_security')
    valid = isin('control_security_state', 'valid_looking')
    payload = isin('payload_or_parameter_context', 'bad_payload')
    session = isin('session_state', 'bad_session')
    transition = isin('state_transition_consistency', 'bad_transition')
    transition_severe = isin('state_transition_consistency', 'severe_transition')
    old = isin('sequence_freshness_status', 'old_freshness')
    uncertain = isin('sequence_freshness_status', 'uncertain_freshness')
    held = isin('command_execution_result', 'held_execution')
    accepted = isin('command_execution_result', 'accepted_execution')
    sensitive = isin('command_intent', 'sensitive_intent')
    ground = isin('source_role', 'ground_source')
    # Corroboration only. A shared context is not necessarily an exclusive sensor origin.
    context_off = ~(isin('operational_phase', 'command_phase') &
                    isin('path_type', 'command_path') &
                    isin('protocol_context', 'command_protocol') &
                    isin('direction', 'command_direction'))
    corroboration = held | session | transition
    clauses = [
      [signal | decoder, (signal | tracking) &
       (decoder | retry | (x['session_state'] == 'LINK_LOSS')),
       (severe & decoder) | (signal & decoder & retry)],
      [nav | timing, (nav | timing) & isin('signal_quality_state', 'usable_signal') &
       isin('decoder_status', 'usable_decoder'), nav & timing],
      [command & (weak | payload | (context_off & session)),
       command & ((weak & (payload | transition | session)) | (context_off & payload)),
       command & failed],
      [command & (old | (uncertain & corroboration)), command & old & corroboration,
       command & (x['sequence_freshness_status'] == 'REPLAYED')],
      [command & valid & (payload | transition),
       command & valid & ((payload & transition) | (sensitive & transition & session)),
       command & valid & accepted & ((transition_severe & (payload | sensitive)) |
                                    (payload & transition & ground))],
    ]
    scores = np.column_stack([np.column_stack(c).sum(axis=1) for c in clauses]).astype(float)
    aux = {'repeat_eligible': command & isin('command_intent', 'command_intent') &
           ~isin('command_intent', 'recurrence_excluded_intent'),
           'corroboration': corroboration | old | uncertain,
           'intent': x['command_intent'], 'parameter': x['payload_or_parameter_context']}
    return scores, aux


def aggregate_evidence(frame, indices, length):
    require(length in (1, LENGTH), 'Only L1 and L10 are prespecified')
    scores, aux = window_evidence(frame)
    if length == 1:
        return scores[indices[:, -1]].copy()
    values = scores[indices].copy()
    intent = aux['intent'][indices]
    parameter = aux['parameter'][indices]
    eligible = aux['repeat_eligible'][indices]
    for t in range(1, indices.shape[1]):
        earlier = ((intent[:, :t] == intent[:, t, None]) &
                   (parameter[:, :t] == parameter[:, t, None]) & eligible[:, :t]).any(axis=1)
        bonus = earlier & eligible[:, t] & aux['corroboration'][indices[:, t]]
        values[:, t, 3] = np.minimum(3, values[:, t, 3] + bonus)
    return values.mean(axis=1)


def classify(scores, thresholds):
    thresholds = np.asarray(thresholds, dtype=float)
    require(thresholds.shape == (5,) and np.all(np.isfinite(thresholds)) and
            np.all(thresholds > 0), 'Invalid five-class threshold vector')
    ratios = scores / thresholds
    eligible = ratios >= 1.
    top = ratios.max(axis=1)
    pred = np.where(top >= 1., ratios.argmax(axis=1) + 1, 0)
    return pred, {'eligible_attack_count': eligible.sum(axis=1),
                  'tie_at_max': ((ratios == top[:, None]).sum(axis=1) > 1) & (top >= 1),
                  'max_score_threshold_ratio': top}


def confusion(y, pred):
    y = np.asarray(y, dtype=np.int64); pred = np.asarray(pred, dtype=np.int64)
    require(y.shape == pred.shape and y.ndim == 1 and len(y) > 0, 'Invalid prediction shape')
    require(np.isin(y, range(6)).all() and np.isin(pred, range(6)).all(), 'Invalid label ID')
    return np.bincount(y * 6 + pred, minlength=36).reshape(6, 6)


def metrics(y, pred):
    cm = confusion(y, pred)
    tp = np.diag(cm).astype(float); denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(6), where=denom != 0)
    return {'macro_f1': float(f1.mean()), 'accuracy': float(tp.sum() / cm.sum()),
            'normal_false_alarm_rate': float((cm[0].sum() - cm[0, 0]) / cm[0].sum())
            if cm[0].sum() else None}


def calibrate(scores, y, model_name):
    """Finite, deterministic coordinate search. Never accepts test inputs."""
    require(set(np.unique(y)) == set(range(6)), 'Validation must contain all six classes')
    trace = []

    def evaluate(thresholds, start, sweep, coordinate):
        pred, _ = classify(scores, thresholds)
        result = metrics(y, pred)
        trace.append({'model': model_name, 'start': start, 'sweep': sweep,
                      'coordinate': coordinate, 'thresholds': json.dumps(list(thresholds)), **result})
        rank = (result['macro_f1'], -result['normal_false_alarm_rate'],
                sum(thresholds), tuple(thresholds))
        return rank, result

    global_rank = None; selected = None
    for start in STARTS:
        current = [start] * 5
        rank, result = evaluate(current, start, -1, 'initial')
        for sweep in range(MAX_SWEEPS):
            old = list(current)
            for j, label in enumerate(ATTACKS):
                best_rank, best_result, best = rank, result, list(current)
                for threshold in GRID:
                    candidate = list(current); candidate[j] = threshold
                    candidate_rank, candidate_result = evaluate(candidate, start, sweep, label)
                    if candidate_rank > best_rank:
                        best_rank, best_result, best = candidate_rank, candidate_result, candidate
                rank, result, current = best_rank, best_result, best
            if current == old:
                break
        if global_rank is None or rank > global_rank:
            global_rank = rank
            selected = {'thresholds': list(current), 'validation_macro_f1': result['macro_f1'],
                        'validation_normal_false_alarm_rate': result['normal_false_alarm_rate']}
    return selected, trace


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_input(path, audit, kind='csv'):
    path = Path(path)
    require(path.is_file(), f'Required input missing: {path}')
    before = sha256(path)
    if kind == 'csv':
        value = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    elif kind == 'json':
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    else:
        raise ValueError(f'Unknown input kind: {kind}')
    require(sha256(path) == before, f'Input changed during reading: {path}')
    audit.append({'path': str(path.resolve()), 'sha256': before,
                  'size_bytes': path.stat().st_size, 'read_at_utc': utc_now()})
    return value


def integers(values, description):
    a = pd.to_numeric(np.asarray(values).reshape(-1), errors='raise')
    require(np.isfinite(a).all() and (a == np.floor(a)).all(),
            f'{description}: expected finite integer values')
    require(((a >= 0) & (a < 2**53)).all(), f'{description}: integer values out of range')
    return np.asarray(a, dtype=np.int64).reshape(np.shape(values))


def require_columns(frame, required, description):
    missing = sorted(set(required) - set(frame.columns))
    require(not missing, f'{description}: missing columns {missing}')
    require(not frame[list(required)].isna().any().any(), f'{description}: null values')
    for column in required:
        s = frame[column].astype(str)
        require((s.str.len() > 0).all() and (s == s.str.strip()).all(),
                f'{description}: empty or surrounding-whitespace values in {column}')


def parse_times(values):
    try:
        return pd.to_datetime(values, utc=True, errors='raise')
    except (ValueError, TypeError) as error:
        raise ValueError(f'Invalid/inconsistent timestamp representation: {error}') from error


def validate_window_metadata(frame, mapping, description):
    require_columns(frame, META, description)
    for column in ('episode_step', 'window_id'):
        frame[column] = integers(frame[column], description + ':' + column)
    require(not frame.duplicated(KEYS).any(), f'{description}: duplicate window identity')
    require(not frame.duplicated(['episode_id', 'episode_step']).any(),
            f'{description}: duplicate episode step')
    require(set(frame.label) <= set(mapping), f'{description}: unknown labels')
    if 'label_id' in frame:
        require(np.array_equal(integers(frame.label_id, 'label_id'), frame.label.map(mapping)),
                f'{description}: label_id mismatch')
    require((frame.groupby('episode_id').label.nunique() == 1).all(),
            f'{description}: non-homogeneous episode labels; unsupported A-prime data')
    ordered = frame.sort_values(['episode_id', 'episode_step'])
    within = ordered.episode_id.eq(ordered.episode_id.shift())
    require((ordered.episode_step.diff()[within] == 1).all(),
            f'{description}: non-consecutive episode steps')
    require((ordered.window_id.diff()[within] == 1).all(),
            f'{description}: non-consecutive window_id values')
    timestamps = parse_times(ordered.timestamp)
    require((timestamps.diff()[within].dt.total_seconds() == 1).all(),
            f'{description}: timestamps do not follow 1-second windows')


def resolve_dataset_contract(root, audit):
    """Dataset-id guard: Step-7 build manifest and Step-10 report index must agree.

    Also fills EXPECTED (rows, episodes, L10 common endpoints) for each split from
    those two artifacts, replacing the hard-coded A-prime counts.
    """
    build = read_input(root / f'{DATASET_DIR}/build_manifest.json', audit, 'json')
    index = read_input(root / f'{OUT_ROOT}/tst_sequences_finetuned_main/sequence_build_report_index.json', audit, 'json')
    b_id = build.get('dataset_id') or build.get('contract', {}).get('dataset_id')
    i_id = index.get('dataset_id')
    require(isinstance(b_id, str) and isinstance(i_id, str) and b_id == i_id,
            f'dataset_id mismatch between Step 7 ({str(b_id)[:12]}) and Step 10 ({str(i_id)[:12]}); do not mix artifacts')
    counts = index.get('common_endpoint_report', {}).get('splits', {})
    for split in ('train', 'val', 'test'):
        obs = pd.read_csv(root / f'{DATASET_DIR}/splits/{split}_observable.csv', encoding='utf-8-sig',
                          dtype=str, keep_default_na=False, usecols=['episode_id'])
        l10 = counts.get(split, {}).get('common_endpoint_count')
        require(isinstance(l10, int) and l10 > 0, f'Step 10 index: {split} common_endpoint_count missing')
        EXPECTED[split] = (int(len(obs)), int(obs.episode_id.nunique()), int(l10))
    return {'dataset_version': DATASET_VERSION, 'dataset_id': b_id, 'expected_counts': dict(EXPECTED),
            'step8_run_id': index.get('step8_run_id'), 'step9_run_id': index.get('step9_run_id'),
            'step10_run_id': index.get('run_id')}


def load_mapping(root, audit):
    data = read_input(root / f'{DATASET_DIR}/splits/label_mapping.json', audit, 'json')
    require('label_to_id' in data and isinstance(data['label_to_id'], dict),
            'label_mapping.json must contain label_to_id; no default is assumed')
    raw = data['label_to_id']
    require(set(raw) == set(LABELS), 'Expected exactly six A-prime class names')
    ids = integers(list(raw.values()), 'label_to_id')
    mapping = dict(zip(raw, map(int, ids)))
    require(set(ids) == set(range(6)), 'label_to_id must be a bijection onto 0..5')
    if 'label_order' in data:
        require(data['label_order'] == sorted(mapping, key=mapping.get),
                'label_order disagrees with label_to_id')
    if 'id_to_label' in data:
        require({str(k): str(v) for k, v in data['id_to_label'].items()} ==
                {str(v): k for k, v in mapping.items()}, 'id_to_label disagrees with label_to_id')
    return mapping


def load_observables(root, split, mapping, audit):
    frame = read_input(root / f'{DATASET_DIR}/splits/{split}_observable.csv', audit)
    require_columns(frame, list(FIELDS) + META, split)
    validate_window_metadata(frame, mapping, split)
    n_rows, n_episodes, _ = EXPECTED[split]
    require(len(frame) == n_rows and frame.episode_id.nunique() == n_episodes,
            f'{split}: {DATASET_VERSION} count mismatch; expected {n_rows} rows/{n_episodes} episodes, '
            f'got {len(frame)}/{frame.episode_id.nunique()}. Do not disable this gate to mix artifacts.')
    require(set(frame.label) == set(LABELS), f'{split}: missing a class')
    return frame


def check_disjoint(splits):
    names = list(splits)
    for i, a in enumerate(names):
        ea = set(splits[a].episode_id)
        for b in names[i + 1:]:
            overlap = ea & set(splits[b].episode_id)
            require(not overlap, f'Episode leakage/overlap between {a} and {b}: {sorted(overlap)[:5]}')


def align_sequences(root, split, obs, mapping, audit):
    """Use actual Step 10 indices in Step 9 order after an explicit identity join."""
    wm = read_input(root / f'{OUT_ROOT}/tst_embeddings_finetuned_main/{split}_window_metadata.csv', audit)
    require_columns(wm, META + ['label_id', 'embedding_row_index'], f'{split} Step 9')
    validate_window_metadata(wm, mapping, f'{split} Step 9')
    require(np.array_equal(integers(wm.embedding_row_index, 'embedding_row_index'),
                           np.arange(len(wm))), 'Step 9 embedding_row_index must equal matrix row order')
    require(len(wm) == len(obs), 'Step 9/observable row count mismatch')
    indexed = obs.set_index(KEYS, drop=False)
    key_index = pd.MultiIndex.from_frame(wm[KEYS])
    require(indexed.index.is_unique and key_index.is_unique and
            set(indexed.index) == set(key_index), 'Step 9/observable window identities differ')
    ordered = indexed.reindex(key_index).reset_index(drop=True)
    for column in META:
        if column == 'timestamp':
            left = parse_times(ordered[column]).to_numpy()
            right = parse_times(wm[column]).to_numpy()
        else:
            left = ordered[column].astype(str).to_numpy()
            right = wm[column].astype(str).to_numpy()
        require(np.array_equal(left, right), f'Step 9/observable mismatch: {column}')
    directory = root / f'{OUT_ROOT}/tst_sequences_finetuned_main'
    sm = read_input(directory / f'{split}_tst_sequence_metadata_L10.csv', audit)
    required = ['sequence_id', 'source_split', 'episode_id', 'episode_type', 'scenario_id',
        'sequence_length', 'observable_window_seconds', 'temporal_context_seconds',
        'sequence_start_row', 'sequence_end_row', 'sequence_start_step', 'sequence_end_step',
        'sequence_start_window_id', 'sequence_end_window_id', 'sequence_start_timestamp',
        'sequence_end_timestamp', 'start_label', 'label', 'label_id',
        'sequence_num_label_changes', 'sequence_has_label_transition']
    require_columns(sm, required, 'Step 10 sequence metadata')
    for col in ['sequence_id', 'sequence_length', 'observable_window_seconds',
                'temporal_context_seconds', 'sequence_start_row', 'sequence_end_row',
                'sequence_start_step', 'sequence_end_step', 'sequence_start_window_id',
                'sequence_end_window_id', 'label_id', 'sequence_num_label_changes',
                'sequence_has_label_transition']:
        sm[col] = integers(sm[col], col)
    require(np.array_equal(sm.sequence_id, np.arange(len(sm))),
            'sequence_id must be unique and in original Step 10 order')
    require(len(sm) == EXPECTED[split][2], f'{split}: {DATASET_VERSION} L10 sequence count mismatch')
    for col, expected in [('source_split', split), ('sequence_length', LENGTH),
                          ('observable_window_seconds', 1), ('temporal_context_seconds', LENGTH),
                          ('sequence_num_label_changes', 0), ('sequence_has_label_transition', 0)]:
        require((sm[col] == expected).all(), f'Step 10 unexpected {col}')
    npz_path = directory / f'{split}_tst_sequences_L10.npz'
    require(npz_path.is_file(), f'Required input missing: {npz_path}')
    digest = sha256(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        require({'sequence_indices', 'y'} <= set(data.files), 'NPZ missing sequence_indices/y')
        indices = integers(data['sequence_indices'], 'sequence_indices')
        y = integers(data['y'], 'y')
        require(indices.shape == (len(sm), LENGTH), 'Invalid sequence_indices shape')
        require(y.shape == (len(sm),), 'Invalid NPZ y shape')
        for alias, expected in [('label_ids', y), ('start_row_indices', indices[:, 0]),
                                ('end_row_indices', indices[:, -1])]:
            if alias in data.files:
                require(np.array_equal(integers(data[alias], alias), expected), f'NPZ {alias} alias mismatch')
    require(sha256(npz_path) == digest, f'Input changed during reading: {npz_path}')
    audit.append({'path': str(npz_path.resolve()), 'sha256': digest,
                  'size_bytes': npz_path.stat().st_size, 'read_at_utc': utc_now()})
    require(indices.min() >= 0 and indices.max() < len(ordered), 'Sequence index out of range')
    ep = ordered.episode_id.to_numpy()[indices]
    require((ep == ep[:, -1, None]).all(), 'Sequence crosses an episode boundary')
    steps = ordered.episode_step.to_numpy()[indices]
    require((np.diff(steps, axis=1) == 1).all(), 'Sequence steps are not consecutive')
    require(np.array_equal(indices[:, 0], sm.sequence_start_row) and
            np.array_equal(indices[:, -1], sm.sequence_end_row), 'Step 10 sequence row mismatch')
    # Complete stride-1 population, not just a plausible subset of matching endpoints.
    endpoints = indices[:, -1]
    valid_ends = np.flatnonzero(ordered.groupby('episode_id').episode_step.rank(method='first').to_numpy() >= LENGTH)
    require(len(np.unique(endpoints)) == len(endpoints) and
            np.array_equal(np.sort(endpoints), valid_ends), 'Duplicate or missing L10 endpoints')
    for side, index in [('start', indices[:, 0]), ('end', endpoints)]:
        rows = ordered.iloc[index]
        for field, target in [('episode_step', f'sequence_{side}_step'),
                              ('window_id', f'sequence_{side}_window_id')]:
            require(np.array_equal(rows[field].to_numpy(), sm[target]), f'Step 10 {target} mismatch')
        expected_time = parse_times(rows.timestamp).to_numpy()
        actual_time = parse_times(sm[f'sequence_{side}_timestamp']).to_numpy()
        require(np.array_equal(expected_time, actual_time), f'Step 10 {side} timestamp mismatch')
        target_label = 'start_label' if side == 'start' else 'label'
        require(np.array_equal(rows.label.to_numpy(), sm[target_label]), f'Step 10 {target_label} mismatch')
    for field in ['episode_id', 'episode_type', 'scenario_id']:
        require(np.array_equal(ordered.iloc[endpoints][field].to_numpy(), sm[field]),
                f'Step 10 {field} mismatch')
    expected_y = ordered.iloc[endpoints].label.map(mapping).to_numpy(dtype=np.int64)
    require(np.array_equal(y, expected_y) and np.array_equal(y, sm.label_id), 'Step 10 terminal label mismatch')
    return ordered, indices, sm


def rule_specification():
    return {'version': PROTOCOL_VERSION, 'available_fields': list(FIELDS),
            'used_fields': [f for f in FIELDS if f not in
                            ('observation_stage', 'destination_role', 'service_context')],
            'canonical_labels': list(LABELS), 'membership_sets': TOKENS, 'clauses': CLAUSES,
            'command': 'recognized command intent OR recognized command message type',
            'context_off': 'NOT (command_phase AND command_path AND command_protocol AND command_direction)',
            'corroboration': 'held_execution OR bad_session OR bad_transition',
            'recurrence': 'same recognized non-status intent and parameter as any prior eligible '
                          'window inside this sequence; current corroboration OR old/uncertain freshness; '
                          'add at most one to replay score and cap at three',
            'L1': 'last window only', 'L10': 'arithmetic mean of ten window scores with recurrence support',
            'threshold_grid': list(GRID), 'starts': list(STARTS), 'max_sweeps': MAX_SWEEPS,
            'objective': 'validation common-endpoint six-class sequence macro-F1',
            'tie_selection': 'lower normal FAR, then larger threshold sum, then lexicographically larger thresholds',
            'conflict': 'largest score/threshold among eligible attacks; exact ties use canonical order',
            'normal_fallback': 'no attack reaches its threshold',
            'implementation': {fn.__name__: inspect.getsource(fn) for fn in
                               (window_evidence, aggregate_evidence, classify, metrics, confusion, calibrate)}}


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def input_signature(root, audit):
    return {str(Path(item['path']).relative_to(root)).replace('\\', '/'): item['sha256']
            for item in audit if Path(item['path']).is_relative_to(root) and
            '16_observable_rule_baseline' not in Path(item['path']).parts}


def canonical_y(frame):
    return frame.label.map({label: i for i, label in enumerate(LABELS)}).to_numpy(dtype=np.int64)


def export_evaluation(out, split, model, meta, scores, config, mapping):
    y = canonical_y(meta)
    pred, diagnostics = classify(scores, config['thresholds'])
    result = metrics(y, pred)
    cm = confusion(y, pred)
    table = meta.copy()
    table['model'] = model
    table['y_true'] = meta.label.map(mapping)
    table['y_pred'] = [mapping[LABELS[i]] for i in pred]
    table['true_label'] = meta.label.to_numpy()
    table['pred_label'] = [LABELS[i] for i in pred]
    table['correct'] = (y == pred).astype(int)
    for j, label in enumerate(ATTACKS):
        table['evidence_' + label] = scores[:, j]
    for name, value in diagnostics.items():
        table[name] = value
    table.to_csv(out / f'{split}_predictions_{model}.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cm, index=LABELS, columns=LABELS).rename_axis('true_label').to_csv(
        out / f'{split}_confusion_counts_{model}.csv', encoding='utf-8-sig')
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros((6, 6)),
                           where=cm.sum(axis=1, keepdims=True) != 0)
    pd.DataFrame(normalized, index=LABELS, columns=LABELS).rename_axis('true_label').to_csv(
        out / f'{split}_confusion_normalized_{model}.csv', encoding='utf-8-sig')
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(6), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(6), where=cm.sum(1) != 0)
    f1 = np.divide(2*tp, cm.sum(0)+cm.sum(1), out=np.zeros(6), where=cm.sum(0)+cm.sum(1) != 0)
    pd.DataFrame({'label': LABELS, 'precision': precision, 'recall': recall, 'f1': f1,
                  'support': cm.sum(1), 'predicted_count': cm.sum(0)}).to_csv(
        out / f'{split}_per_class_{model}.csv', index=False, encoding='utf-8-sig')
    return {'split': split, 'model': model, 'n_sequences': len(meta),
            'n_episodes': meta.episode_id.nunique(), 'population': 'common_L10_endpoints',
            **result, 'conflict_rate': float((diagnostics['eligible_attack_count'] > 1).mean()),
            'exact_tie_rate': float(diagnostics['tie_at_max'].mean()),
            'normal_prediction_rate': float((pred == 0).mean())}


def coverage(train, frames):
    vocab = {field: set(train[field]) for field in FIELDS}
    report = {'train_vocabulary': {f: sorted(v) for f, v in vocab.items()}, 'unseen_from_train': {}}
    for name, frame in frames.items():
        report['unseen_from_train'][name] = {
            f: {'unseen_values': sorted(set(frame[f]) - vocab[f]),
                'n_rows_with_unseen_value': int((~frame[f].isin(vocab[f])).sum())} for f in FIELDS}
    return report


def dataset_audit(frames, sequence_meta):
    return {split: {'records': len(frame), 'episodes': int(frame.episode_id.nunique()),
                    'class_records': {str(k): int(v) for k, v in frame.label.value_counts().items()},
                    'class_episodes': {str(k): int(v) for k, v in
                        frame.groupby('label').episode_id.nunique().items()},
                    'extra_columns_not_used_as_features': sorted(set(frame) - set(FIELDS) - set(META)),
                    'sequences': len(sequence_meta[split]) if split in sequence_meta else None,
                    'alignment_passed': split in sequence_meta}
            for split, frame in frames.items()}


def run(root, calibrate_only=False, frozen_config=None):
    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = root / f'{OUT_ROOT}/16_observable_rule_baseline' / ('run_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    audit = []
    manifest = {'status': 'RUNNING', 'started_at_utc': utc_now(), 'root': str(root),
                'output_dir': str(out), 'script_sha256': sha256(Path(__file__)),
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'calibrate_only': bool(calibrate_only), 'inputs': audit,
                'test_opened_after_freeze': False}
    write_json(out / 'run_manifest.json', manifest)
    started = time.perf_counter()
    try:
        contract = resolve_dataset_contract(root, audit)
        manifest['dataset_contract'] = contract
        rule_spec = rule_specification()
        rule_digest = object_digest(rule_spec)
        protocol = {'version': PROTOCOL_VERSION, 'dataset_contract': contract, 'expected_counts': EXPECTED,
                    'rule_spec_sha256': rule_digest, 'available_observable_fields': list(FIELDS),
                    'metadata_feature_use': False, 'test_access': 'after_config_freeze_only',
                    'primary_population': 'common L10 endpoints; stride 1; terminal-window labels',
                    'models': ['Rule_L1', 'Rule_L10'], 'calibration_seed': None,
                    'uncertainty': 'point estimates only; no bootstrap or seed duplication',
                    'design_provenance': 'generator-informed semantic rules; latent generator values excluded',
                    'limitations': ['heuristic baseline, not exhaustive rule optimization',
                                    'L10 combines averaging and recurrence support',
                                    'coarse recurrence is not packet identity',
                                    'episode classification is not onset detection',
                                    'synthetic performance is not operational validation']}
        write_json(out / 'protocol.json', protocol)
        write_json(out / 'rule_spec.json', rule_spec)
        protocol_digest = object_digest(protocol)
        manifest['protocol_sha256'] = protocol_digest
        print('Step 16: read train/validation and audit common L10 endpoints.', flush=True)
        mapping = load_mapping(root, audit)
        frames = {split: load_observables(root, split, mapping, audit) for split in ('train', 'val')}
        check_disjoint(frames)
        aligned_val, val_indices, val_meta = align_sequences(root, 'val', frames['val'], mapping, audit)
        validation_signature = input_signature(root, audit)
        val_scores = {f'Rule_L{length}': aggregate_evidence(aligned_val, val_indices, length)
                      for length in (1, LENGTH)}
        models = {}; trace = []
        if frozen_config is None:
            for model, scores in val_scores.items():
                print(f'Calibrating {model} on validation only ...', flush=True)
                models[model], evaluations = calibrate(scores, canonical_y(val_meta), model)
                trace.extend(evaluations)
            pd.DataFrame(trace).to_csv(out / 'validation_search_trace.csv', index=False, encoding='utf-8-sig')
            calibration_origin = 'new_validation_calibration'
        else:
            saved = read_input(Path(frozen_config).resolve(), audit, 'json')
            require(saved.get('rule_spec_sha256') == rule_digest and
                    saved.get('protocol_version') == PROTOCOL_VERSION, 'Frozen rule/protocol digest mismatch')
            require(saved.get('protocol_sha256') == protocol_digest, 'Frozen protocol digest mismatch')
            require(saved.get('label_to_id') == mapping, 'Frozen label mapping mismatch')
            require(saved.get('calibration_input_sha256') == validation_signature,
                    'Frozen calibration train/validation input hashes differ from current A-prime inputs')
            require(set(saved.get('models', {})) == set(val_scores), 'Frozen model names mismatch')
            models = saved['models']
            for model in models:
                thresholds = models[model].get('thresholds', [])
                require(len(thresholds) == 5 and all(type(t) in (int, float) and t in GRID for t in thresholds),
                        'Frozen thresholds outside prespecified grid')
            calibration_origin = 'reused_frozen_config'
        frozen = {'protocol_version': PROTOCOL_VERSION, 'frozen_at_utc': utc_now(),
                  'protocol_sha256': protocol_digest,
                  'rule_spec_sha256': rule_digest, 'script_sha256': manifest['script_sha256'],
                  'label_to_id': mapping, 'models': models, 'calibration_origin': calibration_origin,
                  'calibration_input_sha256': validation_signature, 'test_inputs_read_before_freeze': False}
        # This write is intentionally before ANY test input read/hash/metric calculation.
        frozen_path = out / 'frozen_rule_config.json'
        write_json(frozen_path, frozen)
        frozen_digest = sha256(frozen_path)
        manifest['frozen_config_sha256'] = frozen_digest
        manifest['frozen_at_utc'] = frozen['frozen_at_utc']
        write_json(out / 'run_manifest.json', manifest)
        print('Validation settings frozen. No test-driven calibration follows.', flush=True)
        summaries = [export_evaluation(out, 'val', model, val_meta, scores, models[model], mapping)
                     for model, scores in val_scores.items()]
        sequence_meta = {'val': val_meta}
        if not calibrate_only:
            require(sha256(frozen_path) == frozen_digest, 'Frozen config changed before test evaluation')
            manifest['test_opened_after_freeze'] = True
            frames['test'] = load_observables(root, 'test', mapping, audit)
            check_disjoint(frames)
            aligned_test, test_indices, test_meta = align_sequences(root, 'test', frames['test'], mapping, audit)
            sequence_meta['test'] = test_meta
            for length in (1, LENGTH):
                model = f'Rule_L{length}'
                scores = aggregate_evidence(aligned_test, test_indices, length)
                summaries.append(export_evaluation(out, 'test', model, test_meta, scores, models[model], mapping))
        require(sha256(frozen_path) == frozen_digest, 'Frozen config changed during evaluation')
        summary = pd.DataFrame(summaries)
        summary.to_csv(out / 'results_summary.csv', index=False, encoding='utf-8-sig')
        write_json(out / 'data_audit.json', dataset_audit(frames, sequence_meta))
        write_json(out / 'token_coverage.json', coverage(frames['train'], frames))
        write_json(out / 'input_audit.json', audit)
        text = ['Step 16 semantic-rule evaluation', '',
                'Rule thresholds were fixed using common-endpoint validation only.',
                'Rule_L1 and Rule_L10 are evaluated at the same L10 endpoints.',
                'L10 additionally uses corroborated coarse command-category recurrence.',
                'No CI or statistical superiority/equivalence conclusion is produced.',
                'Existing Step 15 confidence intervals do not include these rule baselines.', '',
                summary.to_string(index=False), '',
                'Next review: data_audit, selected thresholds, per-class errors, conflict/tie rates.',
                'Do not revise rules or thresholds in response to these test results.',
                'After the audit, continue the prespecified encoder comparison.']
        (out / 'analysis_summary.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
        manifest['status'] = 'CALIBRATED' if calibrate_only else 'COMPLETE'
        manifest['elapsed_seconds'] = time.perf_counter() - started
        manifest['completed_at_utc'] = utc_now()
        manifest['outputs'] = [{'file': p.name, 'sha256': sha256(p), 'size_bytes': p.stat().st_size}
                               for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run_manifest.json']
        write_json(out / 'run_manifest.json', manifest)
        print(summary[['split', 'model', 'n_sequences', 'macro_f1', 'accuracy']].to_string(index=False))
        print(f"\n{manifest['status']}\nResults: {out}", flush=True)
        return out
    except Exception as error:
        manifest.update(status='FAILED', failed_at_utc=utc_now(), error_type=type(error).__name__,
                        error=str(error), elapsed_seconds=time.perf_counter() - started)
        write_json(out / 'input_audit.json', audit)
        (out / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(out / 'run_manifest.json', manifest)
        print(f'FAILED: {error}\nDiagnostic folder: {out}', file=sys.stderr, flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--frozen-config', type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        run(args.project_root, args.calibrate_only, args.frozen_config)
    except Exception:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
