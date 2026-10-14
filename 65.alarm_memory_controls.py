#!/usr/bin/env python3
"""Step65: finite alarm-memory controls for the completed, reviewed Step62 run.

Place beside Steps62/63/64 in the existing project and run:
    python "65.alarm_memory_controls.py" --preflight-only
    python "65.alarm_memory_controls.py"
    python "65.alarm_memory_controls.py" --self-test

First additional experiment ONLY. No continuous-stream generator or recovery labels.
ActualHistory and RepeatCurrent retain the original Step62 endpoint probabilities.
Three controls apply to RepeatCurrent: max(any-attack)-vector hold, probability mean,
and finite normalized EWMA(alpha=.25), all using exactly [t-9,...,t]. Earliest nine
RepeatCurrent scores are inferred from the SELECTED saved head and existing FP32
Step9 embeddings; no encoder loading/training and no head training. Fixed canaries
must replay saved scores. This avoids silently adding nine seconds of warm-up.

All five methods are evaluated on the original common L10 endpoints. Thresholds
are locked using NORMAL validation scores AFTER postprocessing, before test scoring.
Primary family: 3 controls x 2 validation FPR budgets x 2 metrics =12 contrasts.
Metrics: history-only ALARM-GATED correct-type recall, and 10s any-attack event
probability from maker-active onset. These are NOT the ungated Step63 recall.
CIs condition on fixed ten paired heads/encoder and calibration. Post-hoc internal
analysis; neither external validation nor preregistered independent confirmation.

Requires original local Step62 outputs (including repeat_current/best.pt), Step9
embeddings and dataset, not just review ZIPs. Existing sources are read-only.
Default persistent output: outputs/shared_nominal_v2/65_alarm_memory_controls/
run_memory_controls_v1. Rerun identical command to reuse verified per-seed stages.
Only CPU FP32 prefix/canary inference is used. Runtime is hardware-dependent.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

VERSION = 'step65_finite_alarm_memory_v1'
PINS = {
    62: 'b5095475f0ff9aedd0b9797b24e39636041c0d20e914cf2d7a98fd9eab56cb81',
    63: '29a8d528634f30b355584e31d82dce0fc522b227862f0b6051bfeb0d0a4d7681',
    64: '0c698e566ab736fbec1a0143b0bc49b66273258eea0f064f10f7ec09658b7444',
    11: 'a2101c08cfecbd6af80975195a1bfde265462c0e7939c15ec06ca8d2e4d3967d',
}
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
METHODS = ('ActualHistory', 'RepeatCurrent', 'MaxHold10', 'ProbabilityMean10', 'FiniteEWMA10')
TARGETS = (.01, .05)
LENGTH = 10
ALPHA = .25
CANARY_ATOL, CANARY_RTOL = 5e-5, 1e-4
DEFAULT_OUTPUT = Path('outputs/shared_nominal_v2/65_alarm_memory_controls/run_memory_controls_v1')
DEFAULT62 = Path('outputs/shared_nominal_v2/62_temporal_information_control/run_information_control_v1')
DEFAULT9 = Path('outputs/shared_nominal_v2/tst_embeddings_finetuned_main')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_json(x):
    if isinstance(x, dict): return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [safe_json(v) for v in x]
    if isinstance(x, np.ndarray): return safe_json(x.tolist())
    if isinstance(x, np.generic): return safe_json(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def obj_hash(x):
    return hashlib.sha256(json.dumps(safe_json(x), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False,
                             allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    frame.to_csv(tmp, index=False, encoding='utf-8-sig', float_format='%.17g')
    os.replace(tmp, path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def resolve(root, path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def check_file(path, expected, audit=None):
    path = Path(path)
    require(path.is_file(), 'Required local input missing: ' + str(path))
    require(digest(path) == expected, 'SHA256 mismatch; restore the recorded input, do not edit its pin: ' + str(path))
    if audit is not None: audit[str(path.resolve())] = expected
    return path


def import_verified(path, expected, name):
    check_file(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def locate_code(root, run62, number):
    candidates = list(root.glob(f'{number}*.py')) + list((run62 / 'source').glob(f'{number}*.py'))
    # Identical source copies in prior result folders are allowed; never choose by score/mtime.
    if number in (63, 64):
        base = root / 'outputs/shared_nominal_v2'
        candidates += list(base.glob(f'{number}_*/run*/source/{number}*.py'))
        candidates += list(base.glob(f'{number}_*/run*/{number}*.py'))
    matches = sorted({p.resolve() for p in candidates if p.is_file() and digest(p) == PINS[number]})
    require(matches, f'Cannot find reviewed Step{number} source (SHA256 {PINS[number]}). Keep the original numbered scripts beside this file.')
    return matches[0]


@contextmanager
def exclusive_lock(out):
    """OS lock is released on close/crash; no stale-PID deletion or upstream locks."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / '.step65.lock'
    f = path.open('a+b')
    try:
        f.seek(0, 2)
        if f.tell() == 0: f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another Step65 process owns this output directory. Do not start concurrent copies.') from exc
        yield
    finally:
        f.close()


def validate_probabilities(p):
    p = np.asarray(p)
    require(p.ndim == 2 and p.shape[1] == 6 and len(p) > 0, 'Expected nonempty [windows,6] probabilities')
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probability values')
    require(np.max(np.abs(p.sum(-1) - 1)) < 5e-7, 'Probability rows do not sum to one')


def finite_controls(probabilities, ends, episode_ids):
    """Return [3,endpoints,6], max-hold origin lag. No labels/audit flags used.

    MaxHold10 preserves the entire probability vector from the most attack-like
    window; latest timestamp wins equal 1-pNORMAL scores. It is not independent
    per-class maxima (which would not be a probability vector). Binary thresholding
    after this max is equivalent to holding any threshold exceedance for ten samples.
    EWMA is explicitly normalized AND truncated to ten samples, not recursive.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    validate_probabilities(p)
    ends = np.asarray(ends); ep = np.asarray(episode_ids)
    require(ends.ndim == 1 and ends.dtype.kind in 'iu' and len(ends) > 0, 'Invalid endpoint indices')
    require(ep.shape == (len(p),) and (ends >= 9).all() and (ends < len(p)).all(), 'Missing complete ten-window history')
    ix = ends[:, None] - np.arange(9, -1, -1)
    require((ep[ix] == ep[ends, None]).all(), 'History crosses episode boundary')
    block = p[ix]
    score = 1. - block[:, :, 0]
    chosen = 9 - score[:, ::-1].argmax(1)
    maximum = block[np.arange(len(ends)), chosen]
    mean = block.mean(1)
    weights = (1. - ALPHA) ** np.arange(9, -1, -1, dtype=np.float64)
    weights /= weights.sum()
    ewma = np.einsum('ntc,t->nc', block, weights, optimize=False)
    return np.stack((maximum, mean, ewma)), (9 - chosen).astype(np.int8)


def assemble_windows(windows, endpoints, saved_current, prefix_rows, prefix_probs):
    """Retain saved endpoint probabilities verbatim; fill ONLY missing first nine."""
    n = len(windows); ends = endpoints.sequence_end_row.to_numpy(dtype=np.int64)
    required_prefix = np.flatnonzero(windows.episode_step.to_numpy(dtype=int) < 9)
    require(np.array_equal(prefix_rows, required_prefix), 'Prefix row set differs; no padding/imputation allowed')
    validate_probabilities(saved_current); validate_probabilities(prefix_probs)
    require(saved_current.shape == (len(ends), 6) and prefix_probs.shape == (len(prefix_rows), 6), 'Prefix/endpoint shape mismatch')
    require(len(ends) + len(prefix_rows) == n and not np.intersect1d(ends, prefix_rows).size,
            'Prefix and endpoints must partition the complete timeline')
    full = np.full((n, 6), np.nan, dtype=np.float64)
    full[ends] = saved_current; full[prefix_rows] = prefix_probs
    validate_probabilities(full)
    return full


def select_threshold(scores, target):
    # Same strict > order statistic as Step64; no interpolation/tie randomization.
    a = np.asarray(scores, dtype=np.float64)
    require(a.ndim == 1 and len(a) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'Invalid calibration scores')
    require(0 <= target < 1, 'Invalid FPR target')
    budget = int(np.floor(len(a) * target + 1e-12))
    threshold = float(np.partition(a, len(a) - budget - 1)[len(a) - budget - 1])
    count = int((a > threshold).sum())
    require(count <= budget, 'Calibration budget exceeded')
    return dict(threshold=threshold, validation_normal_endpoints=len(a),
                validation_allowed_alarm_count=budget, validation_alarm_count=count,
                validation_achieved_fpr=count / len(a), validation_ties_at_threshold=int((a == threshold).sum()))


def alarm_run_lengths(alarms, times):
    a = np.asarray(alarms, dtype=bool); t = np.asarray(times)
    require(a.ndim == t.ndim == 1 and len(a) == len(t), 'Run timeline mismatch')
    if not len(a): return np.empty(0, dtype=np.int64)
    adjacent = np.diff(t) == 1
    start = a & ~np.r_[False, a[:-1] & adjacent]
    end = a & ~np.r_[a[1:] & adjacent, False]
    lengths = np.flatnonzero(end) - np.flatnonzero(start) + 1
    require(int(lengths.sum()) == int(a.sum()), 'Alarm run count identity failed')
    return lengths


def stage_record(out, name, identity, artifacts, details=None):
    paths = {Path(p).relative_to(out).as_posix(): digest(p) for p in artifacts}
    write_json(out / 'state' / (name + '.json'), dict(identity=identity, files=paths, details=details or {}))


def reuse_stage(out, name, identity):
    marker = out / 'state' / (name + '.json')
    if not marker.exists(): return None
    record = read_json(marker)
    require(record['identity'] == identity, 'Stage protocol changed: ' + name)
    for rel, sha in record['files'].items():
        p = (out / rel).resolve()
        require(out.resolve() in p.parents, 'Unsafe cached artifact path')
        check_file(p, sha)
    return record


def prepare_sources(args, reader, ctx, code_paths):
    root = Path(args.project_root).resolve(); source = ctx['protocol']['source_contract']
    audit = ctx['source_audit']['verified_files']
    directory = resolve(root, args.embeddings_dir or DEFAULT9)
    check_file(directory / 'run_manifest.json', source['manifest_sha256']['step9'], audit)
    require(read_json(directory / 'run_manifest.json')['run_id'] == source['step9_run_id'], 'Wrong Step9 run')
    check_file(directory / 'embedding_extraction_report.json', source['step9_artifacts']['embedding_extraction_report.json'], audit)
    report = read_json(directory / 'embedding_extraction_report.json')
    require(report['embedding_encoder_source'] == 'finetuned' and report['embedding_dtype'] == 'float32'
            and report['use_fp16_inference'] is False, 'Wrong frozen embedding extraction')
    require(report['dataset_id'] == reader.DATASET_ID and report['run_id'] == source['step9_run_id'], 'Embedding lineage differs')
    for split in ('val', 'test'):
        for suffix in ('window_embeddings.npz', 'window_metadata.csv'):
            name = f'{split}_{suffix}'
            check_file(directory / name, source['step9_artifacts'][name], audit)
        meta = pd.read_csv(directory / f'{split}_window_metadata.csv', keep_default_na=False)
        w = ctx['windows'][split]
        require(len(meta) == len(w), 'Step9 window count differs')
        for key in reader.META: require(np.array_equal(meta[key], w[key]), 'Step9 metadata differs: ' + key)
    checkpoints = []
    for seed in ctx['seeds']:
        folder = ctx['run'] / f'seed_{seed}' / 'repeat_current'
        done = read_json(folder / 'trainval_complete.json')
        path = check_file(folder / 'best.pt', done['artifact_sha256']['best.pt'], audit)
        checkpoints.append(dict(seed=seed, path=str(path), sha256=done['artifact_sha256']['best.pt'],
                                contract=done['contract'], state_sha256=done['result']['best_state_sha256']))
    for n, p in code_paths.items(): check_file(p, PINS[n], audit)
    ctx['embedding_dir'] = directory; ctx['prefix_checkpoints'] = checkpoints


def load_embeddings(ctx, split, reader):
    w = ctx['windows'][split]; source = ctx['protocol']['source_contract']
    name = f'{split}_window_embeddings.npz'; path = ctx['embedding_dir'] / name
    check_file(path, source['step9_artifacts'][name])
    with np.load(path, allow_pickle=False) as z:
        x = z['embeddings']; rows = z['row_indices']; y = z['label_ids']
    require(x.shape == (len(w), 768) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid frozen embeddings')
    require(np.array_equal(rows, np.arange(len(w))) and np.array_equal(y, w.label.map(dict(zip(LABELS, range(6))))),
            'Frozen embedding row/label alignment differs')
    return x


def infer_rows(model, embeddings, rows, batch_size):
    import torch
    logits = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            values = embeddings[rows[start:start + batch_size]]
            x = np.repeat(values[:, None, :], 10, axis=1)
            out = model(torch.from_numpy(x))
            require(bool(torch.isfinite(out).all()), 'Nonfinite prefix/canary logits')
            logits.append(out.detach().cpu().numpy())
    require(logits, 'No prefix/canary inference rows')
    logits = np.concatenate(logits)
    # Exactly the Step62 NumPy FP32 softmax, not a different library softmax.
    shifted = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shifted); p /= p.sum(axis=1, keepdims=True)
    validate_probabilities(p)
    return p.astype(np.float64)


def prefix_predictions(ctx, split, si, saved_current, embeddings, model_module, step62, out, identity, batch_size):
    seed = ctx['seeds'][si]; name = f'prefix_{split}_{seed}'
    file = out / 'prefix_predictions' / (name + '.npz')
    w = ctx['windows'][split]; meta = ctx['endpoints'][split]
    rows = np.flatnonzero(w.episode_step.to_numpy(dtype=int) < 9)
    # Fixed first AND last available endpoint in EVERY episode. No score-based sampling.
    groups = meta.groupby('episode_id', sort=False).indices
    canary_ix = np.unique(np.concatenate([np.array([ix[0], ix[-1]]) for ix in groups.values()])).astype(np.int64)
    record = reuse_stage(out, name, identity)
    if record is None:
        import torch
        ck = ctx['prefix_checkpoints'][si]
        state = step62.load_checkpoint(Path(ck['path']), ck['sha256'])
        require(state['contract'] == ck['contract'], 'Selected prefix head contract differs')
        require(step62.state_hash(state['model']) == ck['state_sha256'], 'Selected tensor state differs')
        model = model_module.TimeSeriesTransformerClassifier(model_module.ModelConfig(**ctx['protocol']['architecture']))
        require(sum(p.numel() for p in model.parameters()) == 234758, 'Wrong original TST parameter count')
        model.load_state_dict(state['model'], strict=True)
        replay = infer_rows(model, embeddings, meta.sequence_end_row.to_numpy(dtype=int)[canary_ix], batch_size)
        expected = saved_current[canary_ix]
        require(np.allclose(replay, expected, atol=CANARY_ATOL, rtol=CANARY_RTOL),
                f'Prefix head canary mismatch: {split} seed {seed}, max_abs={np.max(np.abs(replay-expected))}. '
                'STOP; do not loosen tolerances or substitute predictions.')
        probs = infer_rows(model, embeddings, rows, batch_size)
        write_npz(file, row_indices=rows, probabilities=probs, canary_sequence_ids=canary_ix,
                  canary_probabilities=replay, canary_original_probabilities=expected)
        details = dict(prefix_rows=len(rows), canary_endpoints=len(canary_ix),
                       max_abs_canary_error=float(np.max(np.abs(replay - expected))),
                       canary_argmax_disagreements=int((replay.argmax(1) != expected.argmax(1)).sum()),
                       checkpoint_sha256=ck['sha256'], new_encoder_inference=False, training=False)
        stage_record(out, name, identity, [file], details)
        del model, state
    with np.load(file, allow_pickle=False) as z:
        got_rows = z['row_indices']; probs = z['probabilities']
        require(np.array_equal(got_rows, rows) and np.array_equal(z['canary_sequence_ids'], canary_ix), 'Cached prefix alignment differs')
        require(np.array_equal(z['canary_original_probabilities'], saved_current[canary_ix]), 'Cached canary source changed')
        require(np.allclose(z['canary_probabilities'], saved_current[canary_ix], atol=CANARY_ATOL, rtol=CANARY_RTOL), 'Cached canary failed')
    validate_probabilities(probs)
    return rows, probs


def method_probabilities(ctx, split, si, original, embeddings, modules, out, identity, batch_size):
    rows, pp = prefix_predictions(ctx, split, si, original[si, 1], embeddings, modules[11], modules[62], out, identity, batch_size)
    windows = ctx['windows'][split]; meta = ctx['endpoints'][split]
    full = assemble_windows(windows, meta, original[si, 1], rows, pp)
    controls, lag = finite_controls(full, meta.sequence_end_row.to_numpy(dtype=int), windows.episode_id.to_numpy())
    methods = np.concatenate((original[si], controls), axis=0)
    require(methods.shape == (5, len(meta), 6), 'Method shape differs')
    require(np.array_equal(methods[:2], original[si]), 'Original Step62 scores must stay unchanged')
    return methods, lag


def safe_div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def bootstrap_bank(num, den, labels, class_sets, draws, seed):
    """Mean fixed-head numerators; paired stratified episode multiplicities.

    num[K,E,M], den[K,E]. All methods share denominators. Missing/censored episodes
    stay in resampling with zero denominators. No undefined class/draw is dropped.
    Supports arbitrary method count, and all specifications share episode draws.
    """
    num = np.asarray(num, float); den = np.asarray(den, float); labels = np.asarray(labels)
    require(num.ndim == 3 and den.shape == num.shape[:2] and len(labels) == num.shape[1], 'Bootstrap shape mismatch')
    require(len(class_sets) == len(num) and draws > 0 and all(len(c) for c in class_sets), 'Invalid bootstrap specification')
    require(np.isfinite(num).all() and np.isfinite(den).all() and (den >= 0).all(), 'Nonfinite bootstrap components')
    k, _, m = num.shape
    classes = sorted({c for group in class_sets for c in group})
    point = np.zeros((k, m)); support = [[] for _ in range(k)]
    groups = {}
    for c in classes:
        ids = np.flatnonzero(labels == c); selected = np.array([c in group for group in class_sets])
        groups[c] = (ids, selected)
        values = safe_div(num[:, ids].sum(1), den[:, ids].sum(1)[:, None])
        for j in np.flatnonzero(selected):
            point[j] += values[j] / len(class_sets[j])
            support[j].append(int((den[j, ids] > 0).sum()))
    samples = np.zeros((draws, k, m)); rng = np.random.default_rng(seed)
    for start in range(0, draws, 256):
        nrows = min(256, draws - start)
        for c in classes:
            ids, selected = groups[c]; n = len(ids)
            if not n:
                samples[start:start+nrows, selected] = np.nan
                continue
            weights = rng.multinomial(n, np.full(n, 1. / n), size=nrows)
            a = (weights @ num[:, ids].transpose(1, 0, 2).reshape(n, k * m)).reshape(nrows, k, m)
            b = weights @ den[:, ids].T
            values = safe_div(a, b[:, :, None])
            for j in np.flatnonzero(selected):
                samples[start:start+nrows, j] += values[:, j] / len(class_sets[j])
    status = []
    for j in range(k):
        status.append('missing_target_class' if min(support[j]) == 0 else
                      'fewer_than_two_eligible_episodes_in_class' if min(support[j]) < 2 else
                      'undefined_bootstrap_draw' if not np.isfinite(samples[:, j]).all() else 'ok')
    return point, samples, status


class MetricBank:
    def __init__(self, episode_ids, episode_labels, seeds):
        self.ids = np.asarray(episode_ids); self.labels = np.asarray(episode_labels)
        self.seeds = list(seeds); self.mapping = {str(x): i for i, x in enumerate(self.ids)}
        self.entries = {}

    def add(self, spec, si, mi, num, den, classes=(1, 2, 3, 4, 5)):
        key = obj_hash(spec); num = np.asarray(num, float); den = np.asarray(den, float)
        require(num.shape == den.shape == (len(self.ids),), 'Episode metric shape differs')
        if key not in self.entries:
            self.entries[key] = dict(spec=spec, classes=classes, den=den.copy(),
                                    num=np.zeros((len(self.seeds), len(self.ids), len(METHODS))),
                                    seen=np.zeros((len(self.seeds), len(METHODS)), bool))
        entry = self.entries[key]
        require(np.array_equal(entry['den'], den) and entry['classes'] == classes, 'Method-dependent denominator')
        require(not entry['seen'][si, mi], 'Duplicate seed/method metric')
        entry['num'][si, :, mi] = num; entry['seen'][si, mi] = True

    def vectors(self, frame, numerator, denominator):
        require(not frame.episode_id.duplicated().any(), 'Duplicate episode metric rows')
        rows = np.array([self.mapping[str(x)] for x in frame.episode_id], dtype=int)
        a = np.zeros(len(self.ids)); b = np.zeros(len(self.ids))
        a[rows] = np.asarray(numerator, float); b[rows] = np.asarray(denominator, float)
        return a, b


def classify_rows(meta, pred, seed, method):
    y = meta.label_id.to_numpy(dtype=int)
    cm = np.bincount(y * 6 + pred, minlength=36).reshape(6, 6)
    denom = cm.sum(0) + cm.sum(1)
    f = np.divide(2. * np.diag(cm), denom, out=np.zeros(6), where=denom > 0)
    return dict(seed=seed, method=method, endpoints=len(y), six_class_macro_f1=float(f.mean()),
                accuracy=float((pred == y).mean()), normal_fpr=float(1. - safe_div(cm[0, 0], cm[0].sum())),
                **{f'recall_{label}': float(safe_div(cm[c, c], cm[c].sum())) for c, label in enumerate(LABELS)})


def analyze_predictions(ctx, reader, event_module, score_paths, thresholds, out, draws, bootstrap_seed):
    meta = ctx['endpoints']['test']; windows = ctx['windows']['test']
    evidence = reader.evidence_endpoints(windows, meta)
    write_csv(out / 'endpoint_evidence_test.csv', evidence)
    catalog = event_module.build_event_catalog(windows, meta)
    write_csv(out / 'event_catalog_test.csv', catalog)
    all_episodes = windows[['episode_id', 'label']].drop_duplicates().sort_values('episode_id')
    bank = MetricBank(all_episodes.episode_id.astype(str).to_numpy(),
                      all_episodes.label.map(dict(zip(LABELS, range(6)))).to_numpy(), ctx['seeds'])
    ei = np.array([bank.mapping[str(x)] for x in meta.episode_id], dtype=int)
    y = meta.label_id.to_numpy(dtype=int); n_ep = len(bank.ids)
    masks = {name: evidence.evidence_stratum.eq(name).to_numpy()
             for name in ('current_changed', 'history_only_changed', 'no_change_in_L10')}
    endpoint_den = {name: np.bincount(ei, weights=mask, minlength=n_ep) for name, mask in masks.items()}
    threshold_map = {(r['seed'], r['method'], r['target_normal_endpoint_fpr']): r['threshold'] for r in thresholds}
    class_rows, event_summaries, normal_rows = [], [], []
    run_rows = []; event_files = []
    times = event_module.time_seconds(meta.timestamp)
    groups = meta.groupby('episode_id', sort=False).indices
    for si, seed in enumerate(ctx['seeds']):
        print(f'  Metrics: head {si+1}/{len(ctx["seeds"])} ({seed})', flush=True)
        with np.load(score_paths[si], allow_pickle=False) as z:
            scores, predicted, types = z['attack_scores'], z['predicted_classes'], z['attack_types']
        require(scores.shape == predicted.shape == types.shape == (len(METHODS), len(meta)), 'Saved score array shape differs')
        seed_events = []
        for mi, method in enumerate(METHODS):
            class_rows.append(classify_rows(meta, predicted[mi], seed, method))
            correct = predicted[mi] == y
            for name, mask in masks.items():
                num = np.bincount(ei, weights=mask & correct, minlength=n_ep)
                bank.add(dict(metric='ungated_class_recall', evidence_group=name, target_fpr=None,
                              anchor=None, horizon_seconds=None), si, mi, num, endpoint_den[name])
            for target in TARGETS:
                alarm = scores[mi] > threshold_map[(seed, method, target)]
                for name, mask in masks.items():
                    num = np.bincount(ei, weights=mask & alarm & (types[mi] == y), minlength=n_ep)
                    bank.add(dict(metric='alarm_gated_correct_type_recall', evidence_group=name,
                                  target_fpr=target, anchor=None, horizon_seconds=None), si, mi, num, endpoint_den[name])
                ev = event_module.evaluate_alarm_events(catalog, meta, alarm, types[mi])
                ev['seed'] = seed; ev['condition'] = method; ev['target_normal_endpoint_fpr'] = target
                # Distinguish any post-onset alarm from one continuing across the onset.
                carried = []
                for row in ev.itertuples(index=False):
                    ids = np.asarray(groups.get(str(row.episode_id), []), dtype=int)
                    if row.anchor_missing or not len(ids): carried.append(np.nan); continue
                    before = ids[times[ids] < row.onset_epoch_seconds]
                    after = ids[times[ids] >= row.onset_epoch_seconds]
                    known = len(before) and len(after) and times[after[0]] - times[before[-1]] == 1
                    carried.append(bool(alarm[before[-1]] and alarm[after[0]]) if known else np.nan)
                ev['alarm_carried_across_onset'] = carried  # descriptive, no denominator filtering
                seed_events.append(ev)
                for anchor in ('active', 'observable'):
                    part = ev.loc[ev.anchor.eq(anchor)]
                    full = part.fully_followed_10s.to_numpy(dtype=bool)
                    for typ in ('any_attack', 'correct_type'):
                        for metric, column in (
                            ('event_detection_probability', f'{typ}_detected_10s'),
                            ('event_capped_delay_seconds', f'{typ}_capped_delay_10s')):
                            values = np.where(full, part[column].to_numpy(float), 0.)
                            a, b = bank.vectors(part, values, full.astype(float))
                            bank.add(dict(metric=metric, evidence_group=None, target_fpr=target,
                                          anchor=anchor, alarm_type=typ, horizon_seconds=10), si, mi, a, b)
                normal = event_module.normal_episode_counts(meta, alarm)
                lengths_all = []; longest = []; nruns = []
                for row in normal.itertuples(index=False):
                    ids = np.asarray(groups[str(row.episode_id)], int)
                    lengths = alarm_run_lengths(alarm[ids], times[ids]); lengths_all.extend(lengths.tolist())
                    longest.append(int(lengths.max()) if len(lengths) else 0); nruns.append(len(lengths))
                require(np.array_equal(nruns, normal.alarm_run_starts), 'Normal run definition differs from Step64')
                normal['longest_alarm_run_seconds'] = longest
                normal['seed'] = seed; normal['method'] = method; normal['target_fpr'] = target
                run_rows.append(normal)
                n = int(normal.eligible_normal_endpoints.sum()); alarms = int(normal.alarm_count.sum())
                normal_rows.append(dict(seed=seed, method=method, target_fpr=target,
                    normal_endpoint_fpr=float(safe_div(alarms, n)), normal_episodes=len(normal),
                    total_eligible_normal_seconds=n, alarm_occupied_seconds=alarms,
                    episode_any_alarm_probability=float(normal.episode_any_alarm.mean()),
                    run_starts_per_1000_seconds=float(safe_div(1000 * normal.alarm_run_starts.sum(), n)),
                    mean_alarm_run_seconds=float(np.mean(lengths_all)) if lengths_all else np.nan,
                    p95_alarm_run_seconds=float(np.quantile(lengths_all, .95)) if lengths_all else np.nan,
                    longest_alarm_run_seconds=max(lengths_all, default=0),
                    duration_note='Observed sample counts at 1 Hz; runs at episode edges can be censored; not recovery clearance'))
                for metric, column, denom, factor in (
                    ('normal_endpoint_fpr', 'alarm_count', normal.eligible_normal_endpoints.to_numpy(float), 1),
                    ('normal_episode_any_alarm_probability', 'episode_any_alarm', np.ones(len(normal)), 1),
                    ('normal_run_starts_per_1000_seconds', 'alarm_run_starts', normal.eligible_normal_endpoints.to_numpy(float), 1000),
                    ('normal_mean_episode_longest_alarm_seconds', 'longest_alarm_run_seconds', np.ones(len(normal)), 1)):
                    a, b = bank.vectors(normal, factor * normal[column].to_numpy(float), denom)
                    bank.add(dict(metric=metric, evidence_group='NORMAL', target_fpr=target,
                                  anchor=None, horizon_seconds=None), si, mi, a, b, (0,))
        evframe = pd.concat(seed_events, ignore_index=True)
        path = out / 'events_by_seed' / f'seed_{seed}.csv'
        write_csv(path, evframe); event_files.append(path)
        event_summaries.append(event_module.summarize_events_by_head(evframe))
    write_csv(out / 'classification_by_seed.csv', pd.DataFrame(class_rows))
    per_head = pd.concat(event_summaries, ignore_index=True)
    write_csv(out / 'event_metrics_by_seed.csv', per_head)
    write_csv(out / 'event_metrics_mean_heads.csv', event_module.mean_head_summary(per_head))
    write_csv(out / 'normal_metrics_by_seed.csv', pd.DataFrame(normal_rows))
    write_csv(out / 'normal_episode_results_by_seed.csv', pd.concat(run_rows, ignore_index=True))
    # Descriptive mean across head-specific durations, NOT one giant pooled run list.
    nf = pd.DataFrame(normal_rows)
    numeric = nf.select_dtypes(include='number').columns.difference(['seed', 'target_fpr'])
    mean_normal = nf.groupby(['method', 'target_fpr'], sort=False)[numeric].agg(lambda x: np.mean(x.to_numpy(float))).reset_index()
    write_csv(out / 'normal_metrics_mean_heads.csv', mean_normal)
    entries = list(bank.entries.values())
    require(all(x['seen'].all() for x in entries), 'Missing paired seed/method statistic')
    specs = [x['spec'] for x in entries]
    nums = np.stack([x['num'].mean(0) for x in entries]); dens = np.stack([x['den'] for x in entries])
    sets = [x['classes'] for x in entries]
    write_json(out / 'bootstrap_specifications.json', dict(specifications=specs, classes=sets, methods=METHODS,
                                                        seeds=ctx['seeds'], draws=draws, seed=bootstrap_seed))
    write_npz(out / 'bootstrap_episode_components.npz', episode_ids=bank.ids.astype(str), episode_labels=bank.labels,
              mean_head_numerators=nums, denominators=dens)
    print(f'  Paired episode bootstrap: {draws:,} draws; {len(specs)} metric definitions ...', flush=True)
    points, samples, status = bootstrap_bank(nums, dens, bank.labels, sets, draws, bootstrap_seed)
    intervals, comparisons, by_seed, primary_samples = [], [], [], {}
    for j, entry in enumerate(entries):
        spec = specs[j]; seed_points = np.zeros((len(ctx['seeds']), len(METHODS)))
        for c in sets[j]:
            ids = np.flatnonzero(bank.labels == c)
            seed_points += safe_div(entry['num'][:, ids].sum(1), entry['den'][ids].sum()) / len(sets[j])
        require(np.allclose(seed_points.mean(0), points[j], equal_nan=True), 'Mean-head estimand identity failed')
        for mi, method in enumerate(METHODS):
            q = np.quantile(samples[:, j, mi], [.025, .975]) if status[j] == 'ok' else [np.nan, np.nan]
            intervals.append(dict(spec, method=method, estimate=points[j, mi], ci95_low=q[0], ci95_high=q[1],
                                  ci_status=status[j], head_seed_sd=float(seed_points[:, mi].std(ddof=1))))
            for si, seed in enumerate(ctx['seeds']):
                by_seed.append(dict(spec, seed=seed, method=method, estimate=seed_points[si, mi]))
        primary_spec = ((spec['metric'] == 'alarm_gated_correct_type_recall' and spec['evidence_group'] == 'history_only_changed') or
                        (spec['metric'] == 'event_detection_probability' and spec['anchor'] == 'active' and spec.get('alarm_type') == 'any_attack'))
        for mi in range(1, len(METHODS)):
            difference = samples[:, j, 0] - samples[:, j, mi]
            q = np.quantile(difference, [.025, .975]) if status[j] == 'ok' else [np.nan, np.nan]
            primary = primary_spec and mi >= 2
            adjusted = np.quantile(difference, [.05 / 24, 1 - .05 / 24]) if primary and status[j] == 'ok' else [np.nan, np.nan]
            row = dict(spec, reference=METHODS[0], comparator=METHODS[mi], primary=primary,
                       actual_estimate=points[j, 0], comparator_estimate=points[j, mi],
                       difference=points[j, 0]-points[j, mi], ci95_low=q[0], ci95_high=q[1],
                       family_adjusted_low=adjusted[0], family_adjusted_high=adjusted[1],
                       primary_family_size=12 if primary else None, ci_status=status[j],
                       paired_head_seed_sd=float((seed_points[:, 0] - seed_points[:, mi]).std(ddof=1)))
            comparisons.append(row)
            if primary: primary_samples[f'spec_{j}_{METHODS[mi]}'] = difference
    cf = pd.DataFrame(comparisons); primary = cf.loc[cf.primary]
    require(len(primary) == 12, 'Expected exactly 12 prespecified primary contrasts')
    write_csv(out / 'metric_intervals.csv', pd.DataFrame(intervals))
    write_csv(out / 'metrics_by_seed.csv', pd.DataFrame(by_seed))
    write_csv(out / 'paired_comparisons.csv', cf)
    write_csv(out / 'primary_comparisons.csv', primary)
    write_npz(out / 'primary_bootstrap_samples.npz', **primary_samples)
    summary = dict(status='STEP65_ALARM_MEMORY_COMPLETE', methods=METHODS, head_seeds=ctx['seeds'],
        test_endpoints=len(meta), test_episodes=meta.episode_id.nunique(), primary_comparisons=primary.to_dict('records'),
        source_step62_identity=ctx['protocol']['identity'], training=False, encoder_inference=False,
        prefix_head_inference=True, analysis='post-hoc fixed-model within-generator follow-up',
        limitations=[
            'Only matched Step62 TST heads and output postprocessors are compared; no new pooling/rule/encoder architecture comparison.',
            'FiniteEWMA10 uses fixed alpha=.25 with normalized weights on exactly ten windows; not recursive EWMA and no test-based alpha search.',
            'Step63 ungated recall and threshold-gated correct-type alarm recall are distinct estimands.',
            'Normal alarm duration is a truncated within-episode description, not post-recovery clearance.',
            'Onset is maker-recorded active / observable change, not independently measured physical onset.',
            'Upstream monitors (e.g. freshness) may already summarize history; this tests additional downstream memory.',
            'Same validation split previously selected checkpoints; bootstrap holds thresholds and ten trained heads fixed.',
            'Twelve primary contrasts share Bonferroni-adjusted percentile intervals. All other intervals are exploratory.',
            'No equality, equivalence, physical deployment, causal mediation, external generalization or TST necessity is established automatically.'])
    write_json(out / 'summary.json', summary)
    return summary


def load_locked_thresholds(out, identity):
    lock = read_json(out / 'threshold_lock.json')
    require(lock['protocol_identity'] == identity, 'Threshold lock protocol differs')
    check_file(out / 'thresholds.json', lock['thresholds_sha256'])
    document = read_json(out / 'thresholds.json')
    require(document['protocol_identity'] == identity, 'Threshold payload identity differs')
    rows = document['rows']
    keys = [(r['seed'], r['method'], r['target_normal_endpoint_fpr']) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate thresholds')
    return rows


def calibrate_validation(ctx, reader, modules, args, out, identity):
    if reuse_stage(out, 'calibration', identity):
        print('Reusing verified validation threshold lock.', flush=True)
        return load_locked_thresholds(out, identity)
    original = reader.load_predictions(ctx, 'val')
    embeddings = load_embeddings(ctx, 'val', reader)
    normal = ctx['endpoints']['val'].label_id.to_numpy(dtype=int) == 0
    records = []
    for si, seed in enumerate(ctx['seeds']):
        print(f'Validation: head {si+1}/{len(ctx["seeds"])} ({seed}); prefix/canary only.', flush=True)
        p, _ = method_probabilities(ctx, 'val', si, original, embeddings, modules, out, identity, args.batch_size)
        for mi, method in enumerate(METHODS):
            score = 1. - p[mi, normal, 0]
            for target in TARGETS:
                record = select_threshold(score, target)
                reference = modules[64].select_threshold(score, target)
                require(all(record[k] == reference[k] for k in record), 'Calibration differs from pinned Step64')
                records.append(dict(seed=seed, method=method, target_normal_endpoint_fpr=target, **record))
    require(len(records) == len(ctx['seeds']) * 5 * 2, 'Incomplete calibration')
    write_json(out / 'thresholds.json', dict(protocol_identity=identity, rows=records,
                selection='NORMAL validation scores after finite postprocessing; no test selection',
                validation_reuse='Same validation as Step62 checkpoint selection; not independent calibration'))
    write_csv(out / 'thresholds.csv', pd.DataFrame(records))
    write_json(out / 'threshold_lock.json', dict(protocol_identity=identity,
                thresholds_sha256=digest(out / 'thresholds.json'), created_at_utc=now(),
                all_validation_jobs_complete=True, test_scoring_started=False))
    stage_record(out, 'calibration', identity, [out / 'thresholds.json', out / 'thresholds.csv', out / 'threshold_lock.json'])
    return load_locked_thresholds(out, identity)


def make_test_scores(ctx, reader, modules, args, out, identity):
    # This gate is evaluated before ANY test probability load or prefix inference.
    thresholds = load_locked_thresholds(out, identity)
    expected = {(s, m, t) for s in ctx['seeds'] for m in METHODS for t in TARGETS}
    require({(r['seed'], r['method'], r['target_normal_endpoint_fpr']) for r in thresholds} == expected,
            'Incomplete test threshold gate')
    original = reader.load_predictions(ctx, 'test')
    paths = []; embeddings = None
    for si, seed in enumerate(ctx['seeds']):
        name = f'test_scores_{seed}'; path = out / 'endpoint_predictions' / (name + '.npz')
        cached = reuse_stage(out, name, identity)
        if cached is None:
            if embeddings is None: embeddings = load_embeddings(ctx, 'test', reader)
            print(f'Test: head {si+1}/{len(ctx["seeds"])} ({seed}); same locked protocol.', flush=True)
            p, lag = method_probabilities(ctx, 'test', si, original, embeddings, modules, out, identity, args.batch_size)
            write_npz(path, attack_scores=1.-p[:, :, 0], predicted_classes=p.argmax(-1).astype(np.int8),
                      attack_types=(p[:, :, 1:].argmax(-1)+1).astype(np.int8), max_hold_origin_lag=lag,
                      methods=np.array(METHODS), sequence_ids=ctx['endpoints']['test'].sequence_id.to_numpy(dtype=np.int64))
            stage_record(out, name, identity, [path])
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z['sequence_ids'], ctx['endpoints']['test'].sequence_id) and
                    np.array_equal(z['methods'], METHODS), 'Cached score endpoint/method order differs')
            require(np.array_equal(z['attack_scores'][:2], 1.-original[si, :, :, 0]) and
                    np.array_equal(z['predicted_classes'][:2], original[si].argmax(-1)) and
                    np.array_equal(z['attack_types'][:2], original[si, :, :, 1:].argmax(-1)+1),
                    'Cached Step62 reference scores were changed')
        paths.append(path)
    return paths


def initialize_protocol(out, protocol):
    identity = obj_hash(protocol); document = dict(protocol, identity=identity)
    path = out / 'protocol.json'
    if path.exists():
        old = read_json(path)
        require(old == safe_json(document), 'Step65 output belongs to a different source/configuration. '
                'Use the same command, or an explicit NEW --output-dir; never overwrite the old protocol.')
    else:
        allowed = {'.step65.lock'}
        require({p.name for p in out.iterdir()} <= allowed, 'Output contains unrelated files')
        write_json(path, document)
    return identity


def output_files(out):
    return [p for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_manifest.json'
            and not p.name.endswith('.tmp') and not p.name.startswith('.') and p.suffix != '.zip']


def package_output(out, status, identity):
    manifest = dict(status=status, protocol_identity=identity, completed_at_utc=now(),
                    training=False, encoder_inference=False,
                    prefix_head_inference=any((out / 'prefix_predictions').glob('*.npz')),
                    prefix_head_inference_complete=status == 'STEP65_ALARM_MEMORY_COMPLETE',
                    output_sha256={p.relative_to(out).as_posix(): digest(p) for p in output_files(out)})
    write_json(out / 'run_manifest.json', manifest)
    suffix = '_review.zip' if status == 'STEP65_ALARM_MEMORY_COMPLETE' else '_diagnostic_review.zip'
    archive = out.parent / (out.name + suffix); tmp = archive.with_name(archive.name + '.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in output_files(out) + [out / 'run_manifest.json']:
            z.write(p, p.relative_to(out).as_posix())
    os.replace(tmp, archive)
    return archive


def verify_complete(out, identity):
    path = out / 'run_manifest.json'
    if not path.exists(): return False
    m = read_json(path)
    if m['status'] != 'STEP65_ALARM_MEMORY_COMPLETE': return False
    require(m['protocol_identity'] == identity, 'Completed result identity differs')
    for rel, sha in m['output_sha256'].items():
        p = (out / rel).resolve(); require(out.resolve() in p.parents, 'Unsafe output manifest path')
        check_file(p, sha)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--step62-run', type=Path)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--embeddings-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--bootstrap-draws', type=int, default=20000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026100865)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test: return self_test()
    require(args.threads > 0 and args.batch_size > 0 and args.bootstrap_draws >= 2000, 'Invalid threads/batch/draw count')
    require(0 <= args.bootstrap_seed < 2**63, 'Invalid bootstrap seed')
    root = args.project_root.resolve(); args.project_root = root
    run62 = resolve(root, args.step62_run or DEFAULT62)
    out = resolve(root, args.output_dir or DEFAULT_OUTPUT)
    source_dirs = [run62, resolve(root, args.dataset_dir or 'dataset/shared_nominal_v2'),
                   resolve(root, args.embeddings_dir or DEFAULT9)]
    require(out != root, 'Output cannot be project root')
    for p in source_dirs:
        require(out != p and out not in p.parents and p not in out.parents, 'Output overlaps an upstream input directory')
    code_paths = {n: locate_code(root, run62, n) for n in (63, 64, 62, 11)}
    reader = import_verified(code_paths[63], PINS[63], '_step65_reader63')
    require(reader.READER_API == 'step62_followup_reader_v1', 'Reader API differs')
    print('Step65: reviewed Step62 -> finite score memory; no retraining.', flush=True)
    ctx = reader.load_context(args)
    prepare_sources(args, reader, ctx, code_paths)
    try:
        import torch
    except ImportError as exc:
        raise ValueError('Use the existing project .venv with PyTorch. Encoder packages/models are not loaded.') from exc
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    modules = {63: reader}
    for n in (64, 62, 11): modules[n] = import_verified(code_paths[n], PINS[n], f'_step65_source{n}')
    runtime = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   torch=str(torch.__version__), device='cpu', precision='FP32', threads=args.threads, batch_size=args.batch_size)
    protocol = dict(version=VERSION, script_sha256=digest(__file__), source_pins=PINS,
        source_step62_identity=ctx['protocol']['identity'], dataset_id=reader.DATASET_ID,
        input_sha256=ctx['source_audit']['verified_files'], methods=METHODS, head_seeds=ctx['seeds'],
        memory_length=10, finite_ewma_alpha=.25, runtime=runtime, validation_fpr_targets=TARGETS,
        primary_family_size=12, bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed,
        primary_metrics=['history_only_changed alarm-gated correct-type five-class macro recall',
                         'maker-active any-attack 10-second five-class macro event probability'],
        primary_comparators=METHODS[2:], thresholds='strict >, each method/seed after postprocessing; validation only',
        max_hold='Choose entire probability vector with maximal 1-pNORMAL; latest tie wins; exactly ten windows',
        prefix='Infer only episode steps0..8 using saved selected repeat_current head and frozen Step9 embeddings',
        canary=dict(rows='first and last original L10 endpoint in every episode', atol=CANARY_ATOL, rtol=CANARY_RTOL),
        original_endpoint_scores='unchanged; all original Step62 common causal L10 endpoints retained',
        statistical_unit='class-stratified paired whole episode; fixed heads and thresholds; no row/seed resampling',
        correction='12-way Bonferroni percentile intervals, individual level 1-.05/12; other CIs exploratory',
        training=False, encoder_inference=False, new_prefix_head_inference=True, post_hoc=True,
        reset='only episode boundaries in original corpus; no onset-based state reset; not a continuous deployment evaluation',
        scope='Additional output memory versus additional embedding history within matched Step62 heads; '
              'does not implement experiment2, other encoder/pooling/rule comparisons or a recovery detector')
    managed = False; identity = None
    with exclusive_lock(out):
        identity = initialize_protocol(out, protocol); managed = True
        if verify_complete(out, identity):
            archive = out.parent / (out.name + '_review.zip')
            if not archive.exists(): archive = package_output(out, 'STEP65_ALARM_MEMORY_COMPLETE', identity)
            print('Verified completed Step65; no recomputation.\nReview ZIP: ' + str(archive), flush=True)
            return 0
        try:
            (out / 'source').mkdir(exist_ok=True)
            for n, p in code_paths.items(): shutil.copyfile(p, out / 'source' / p.name)
            shutil.copyfile(__file__, out / 'source' / Path(__file__).name)
            write_json(out / 'input_audit.json', ctx['source_audit'])
            population = [dict(split=s, windows=len(ctx['windows'][s]), endpoints=len(ctx['endpoints'][s]),
                               episodes=ctx['windows'][s].episode_id.nunique(),
                               required_prefix_rows=int(ctx['windows'][s].episode_step.lt(9).sum())) for s in ('val', 'test')]
            write_csv(out / 'population_audit.csv', pd.DataFrame(population))
            if args.preflight_only:
                write_json(out / 'preflight.json', dict(status='STEP65_PREFLIGHT_PASS', protocol_identity=identity,
                    training=False, prediction_loading=False, prefix_inference=False))
                print('PREFLIGHT PASS: sources, datasets, selected heads and Step9 files authenticated. '
                      'Canary/prefix inference occurs in the full run.\nOutput: ' + str(out), flush=True)
                return 0
            try:
                from threadpoolctl import threadpool_limits
                limits = threadpool_limits(limits=args.threads)
            except ImportError: limits = nullcontext()
            with limits:
                thresholds = calibrate_validation(ctx, reader, modules, args, out, identity)
                print('Validation thresholds locked. Test scoring may now begin.', flush=True)
                score_paths = make_test_scores(ctx, reader, modules, args, out, identity)
                analyze_predictions(ctx, reader, modules[64], score_paths, thresholds, out,
                                    args.bootstrap_draws, args.bootstrap_seed)
            # Detect an accidental simultaneous edit to any source actually relied on.
            for path, sha in ctx['source_audit']['verified_files'].items(): check_file(path, sha)
            require(digest(__file__) == protocol['script_sha256'], 'Step65 source changed during execution')
            if (out / 'failure.json').exists(): (out / 'failure.json').unlink()
            archive = package_output(out, 'STEP65_ALARM_MEMORY_COMPLETE', identity)
            print('STEP65_ALARM_MEMORY_COMPLETE\nResults: ' + str(out) + '\nReview ZIP: ' + str(archive), flush=True)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            if managed:
                write_json(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                    next_action='Rerun the identical command; verified prefix/test-score stages are reused.'))
                archive = package_output(out, 'STEP65_INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'STEP65_FAILED', identity)
                print('STOPPED: ' + repr(exc) + '\nDiagnostic ZIP: ' + str(archive), file=sys.stderr)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1


def self_test():
    import unittest
    from types import SimpleNamespace
    m = SimpleNamespace(**globals())

    class ControlsTests(unittest.TestCase):
     def setUp(self):
      self.assertTrue(callable(getattr(m,'finite_controls',None)), 'finite_controls is not implemented')
      self.p=np.tile([.98,.004,.004,.004,.004,.004],(22,1))
      self.p[0]=[.1,.85,.02,.01,.01,.01]
      self.p[9]=[.1,.01,.85,.02,.01,.01]
     def test_shapes(self):
      q,lag=m.finite_controls(self.p,np.array([9,10,19,20]),np.array(['a']*22))
      self.assertEqual(q.shape,(3,4,6));np.testing.assert_allclose(q.sum(-1),1)
     def test_latest_tie_and_expiration(self):
      q,lag=m.finite_controls(self.p,np.array([9,10,19]),np.array(['a']*22))
      np.testing.assert_array_equal(lag,[0,1,0]);np.testing.assert_array_equal(q[0,0],self.p[9])
      np.testing.assert_array_equal(q[0,2],self.p[19])
     def test_finite_not_recursive_ewma(self):
      q,_=m.finite_controls(self.p,np.array([19]),np.array(['a']*22))
      np.testing.assert_allclose(q[2,0],self.p[19])
     def test_ewma_recent_weight(self):
      q,_=m.finite_controls(self.p,np.array([9]),np.array(['a']*22))
      w=.75**np.arange(9,-1,-1);w/=w.sum()
      np.testing.assert_allclose(q[2,0],w@self.p[:10])
     def test_causality(self):
      q,_=m.finite_controls(self.p,np.array([9]),np.array(['a']*22))
      p=self.p.copy();p[10:]=[1,0,0,0,0,0]
      r,_=m.finite_controls(p,np.array([9]),np.array(['a']*22));np.testing.assert_array_equal(q,r)
     def test_boundary_rejected(self):
      with self.assertRaises(ValueError):m.finite_controls(self.p,np.array([10]),np.array(['a']*10+['b']*12))
     def test_prefix_not_imputed(self):
      with self.assertRaises(ValueError):m.finite_controls(self.p,np.array([5]),np.array(['a']*22))
     def test_bad_probs_rejected(self):
      p=self.p.copy();p[2,0]=np.nan
      with self.assertRaises(ValueError):m.finite_controls(p,np.array([9]),np.array(['a']*22))
     def test_input_read_only(self):
      p=self.p.copy();m.finite_controls(self.p,np.array([9]),np.array(['a']*22));np.testing.assert_array_equal(p,self.p)
    

    class MetricTests(unittest.TestCase):
     def test_strict_tied_threshold(self):
      d=m.select_threshold(np.array([.1,.2,.2,.2,.9]),.4)
      self.assertEqual(d['threshold'],.2);self.assertEqual(d['validation_alarm_count'],1)
     def test_zero_budget(self):
      d=m.select_threshold(np.array([.1,.3]),.01);self.assertEqual(d['threshold'],.3)
     def test_gap_runs(self):
      np.testing.assert_array_equal(m.alarm_run_lengths([1,1,1,0,1],[0,1,5,6,7]),[2,1,1])
     def test_empty_runs(self):
      self.assertEqual(len(m.alarm_run_lengths([0,0],[0,1])),0)
     def test_bank_pair_identity(self):
      self.assertTrue(callable(getattr(m,'bootstrap_bank',None)),'bootstrap_bank is not implemented')
      num=np.array([[[1,1],[0,0],[1,1],[0,0]]],float)
      den=np.ones((1,4));labels=np.array([1,1,2,2])
      point,boot,status=m.bootstrap_bank(num,den,labels,[(1,2)],200,71)
      np.testing.assert_array_equal(boot[:,:,0],boot[:,:,1]);np.testing.assert_allclose(point,[[.5,.5]])
     def test_missing_subgroup_not_dropped(self):
      self.assertTrue(callable(getattr(m,'bootstrap_bank',None)),'bootstrap_bank is not implemented')
      num=np.zeros((1,4,2));den=np.array([[0,0,1,1]])
      p,b,status=m.bootstrap_bank(num,den,np.array([1,1,2,2]),[(1,2)],100,71)
      self.assertTrue(np.isnan(p).all());self.assertNotEqual(status[0],'ok')

    suite = unittest.TestSuite()
    for case in (ControlsTests, MetricTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print("Step65 stopped safely: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
