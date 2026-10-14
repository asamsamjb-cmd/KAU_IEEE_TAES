#!/usr/bin/env python3
"""Step70: RF jamming raw-measurement statistical baselines on shared_nominal_v2.

Run beside the numbered scripts in the existing environment:
    python "70.rf_statistical_baselines.py"
    python "70.rf_statistical_baselines.py" --preflight-only
    python "70.rf_statistical_baselines.py" --self-test

One narrow RF_LINK_JAMMING-vs-rest comparison, NOT new generation/external testing.
New methods: RF_Threshold_L1 and RF_FiniteCUSUM_L10. Only the observed link's
SNR/SINR and frame-loss rate are consumed. Train NORMAL estimates mean/std per
link. k=.5 is fixed. The L10 statistic restarts at t-9, accumulates separately by
link/measurement, holds (no update) on unobserved links and expires at t-10.
This is a finite-window adaptation of one-sided tabular CUSUM, not an unbounded
CUSUM with claimed average-run-length guarantees. No delay/onset/recovery study.

Existing references: Step16 pre-fusion RF evidence, Step62 actual_history RF
probabilities (10 heads), Step59 v2 RawScoped L10 HGB (5 fits). Step59 saved hard
labels are insufficient for a common operating budget, so ONLY the original HGB
models receive CPU predict_proba with strict replay of their saved hard labels.
No fit(), Torch, neural inference, encoder/head training or Internet access.

All methods calibrated separately to 1%/5% NORMAL validation endpoint RF alarm
budgets with strict score > threshold. Test scores are read/computed only after
calibration lock. Shared raw companion CSVs contain all splits and are hash/read
scanned; train/val membership filters precede fitting. Metadata may be audited
before lock; do not describe this as never accessing test-containing bytes.

All original common L10 endpoints and quiet scenario labels are retained.
RF-positive F1 (not six-class macro-F1), precision, recall, NORMAL false RF alarm
rate and other-attack-to-RF misidentification are reported. Primary family: four
TST-minus-new-baseline RF-F1 contrasts (2 methods x 2 budgets), Bonferroni CIs.
20,000 class-stratified whole-episode paired bootstrap draws; each method's fixed
fit-set is averaged as metrics, NOT an ensemble, and seeds are not resampled.
Conditional retrospective internal comparison; unequal observation information
and training objectives mean this does not isolate semantic/TST superiority.

Requires original local Step59 v2 directory INCLUDING model.joblib, original
raw companion CSVs, Step62 and Step16 completed outputs. Review ZIPs without
HGB weights cannot perform the required probability replay. Old outputs are
read-only. Rerun identical command to reuse completed stages after SHA checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import time
import traceback
import warnings
import zipfile

import numpy as np
import pandas as pd

VERSION = 'shared_nominal_rf_statistics_v1'
DATASET_ID = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
PROJECTION_SHA = 'adfe6e2ea4268287efa593fb63e16ca84bc2dddc1bf2d22cd95ad360ca17d71d'
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
META = ('scenario_id', 'episode_id', 'episode_type', 'episode_step', 'window_id', 'timestamp', 'label')
FIELDS = tuple('observation_stage path_type direction source_role destination_role protocol_context operational_phase message_type service_context command_intent payload_or_parameter_context sequence_freshness_status control_security_state signal_quality_state tracking_stability navigation_consistency timing_consistency decoder_status command_execution_result retry_timeout_state session_state state_transition_consistency'.split())
LINKS = ('ttc', 'isl', 'iot', 'uav')
LINK_OF_PATH = dict(zip(('GATEWAY_LEO', 'LEO_LEO', 'IOT_LEO', 'UAV_LEO'), LINKS))
NO_LINK_PATHS = ('GNSS_LEO', 'INTERNAL_ONBOARD')
RF_COLS = {l: (l + ('_sinr_db' if l == 'uav' else '_snr_db'), l + '_frame_loss_rate') for l in LINKS}
LINK_COLS = {l: RF_COLS[l] + tuple(l + s for s in ('_crc_error_count', '_decode_fail_count', '_retry_count', '_timeout_count')) for l in LINKS}
ORIGIN_FLAGS = ('source_unverified_benign_flag', 'valid_looking_benign_flag')
METHODS = ('RF_Threshold_L1', 'RF_FiniteCUSUM_L10', 'Rule16_RF_L10', 'RawScoped_HGB_L10', 'TST62_ActualHistory_L10')
FIT_SEEDS = ((-1,), (-1,), (-1,), tuple(range(20260610, 20260615)), tuple(range(20260610, 20260620)))
TARGETS = (.01, .05)
LENGTH, REFERENCE_K = 10, .5
DEFAULT59 = Path('outputs/shared_nominal_v2/59_raw_semantic/run_raw_semantic_v2')
DEFAULT62 = Path('outputs/shared_nominal_v2/62_temporal_information_control/run_information_control_v1')
DEFAULT_OUT = Path('outputs/shared_nominal_v2/70_rf_statistical_baselines/run_rf_statistics_v1')
SOURCE_REFERENCE = 'https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm'
SLOTS = [(m, s) for m, seeds in zip(METHODS, FIT_SEEDS) for s in seeds]
METRICS = ('rf_f1', 'rf_precision', 'rf_recall', 'normal_rf_fpr', 'rest_rf_fpr', 'other_attack_rf_mislabel_rate')


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def safe_value(x):
    if isinstance(x, dict): return {str(k): safe_value(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [safe_value(v) for v in x]
    if isinstance(x, np.ndarray): return safe_value(x.tolist())
    if isinstance(x, np.generic): return safe_value(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def obj_hash(x):
    return hashlib.sha256(json.dumps(safe_value(x), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write_json(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(safe_value(x), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(t, p)


def write_csv(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_name(p.name + '.tmp')
    pd.DataFrame(x).to_csv(t, index=False, encoding='utf-8-sig', float_format='%.17g'); os.replace(t, p)


def write_npz(p, **x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_name(p.name + '.tmp')
    with t.open('wb') as f: np.savez_compressed(f, **x)
    os.replace(t, p)


def read_csv(p, **kw):
    return pd.read_csv(p, keep_default_na=False, encoding='utf-8-sig', **kw)


def resolve(root, p):
    p = Path(p).expanduser(); return (p if p.is_absolute() else root / p).resolve()


def safe_child(root, relative):
    s = str(relative); p = PurePosixPath(s)
    require(s and '\\' not in s and ':' not in s and not p.is_absolute() and '..' not in p.parts,
            'Unsafe relative path: ' + s)
    out = (Path(root) / s).resolve(); base = Path(root).resolve()
    require(out != base and base in out.parents, 'Path escapes source: ' + s)
    return out


def ints(a, name):
    a = np.asarray(pd.to_numeric(a, errors='raise'), dtype=float)
    require(np.isfinite(a).all() and np.equal(a, np.floor(a)).all(), name + ' must be finite integers')
    return a.astype(np.int64)


def labels(frame):
    require('label' in frame and set(frame.label.astype(str)) <= set(LABELS), 'Unknown scenario label')
    return frame.label.map(dict(zip(LABELS, range(6)))).to_numpy(dtype=np.int64)


def endpoint_indices(frame):
    require(len(frame) > 0 and {'episode_id', 'episode_step', 'window_id', 'label'} <= set(frame), 'Missing timeline keys')
    e = frame.episode_id.astype(str).to_numpy(); steps = ints(frame.episode_step, 'episode_step'); wins = ints(frame.window_id, 'window_id')
    if 'timestamp' in frame:
        numeric = pd.to_numeric(frame.timestamp, errors='coerce')
        if numeric.notna().all():
            delta = np.r_[np.nan, np.diff(numeric.to_numpy(dtype=float))]
        else:
            parsed = pd.to_datetime(frame.timestamp, errors='raise', utc=True)
            delta = parsed.diff().dt.total_seconds().to_numpy()
        within = np.r_[False, e[1:] == e[:-1]]
        require(np.all(np.isfinite(delta[within])) and np.allclose(delta[within], 1., atol=1e-6, rtol=0),
                'Within-episode timestamps are not consecutive one-second windows; no bridging or time-axis repair')
    y = labels(frame); start = np.r_[0, np.flatnonzero(e[1:] != e[:-1]) + 1]; end = np.r_[start[1:], len(e)]
    require(len(set(e[start])) == len(start), 'Interleaved/repeated episode block')
    pieces = []
    for a, b in zip(start, end):
        require(np.array_equal(steps[a:b], np.arange(b-a)) and np.all(np.diff(wins[a:b]) == 1), 'Episode steps/windows have gaps or do not start at zero')
        require(np.all(y[a:b] == y[a]), 'Mixed labels within episode')
        if b-a >= LENGTH: pieces.append(np.arange(a+LENGTH-1, b, dtype=np.int64))
    require(pieces, 'No full L10 windows')
    return np.concatenate(pieces)


def observed_rf(frame):
    require('raw_path_type' in frame, 'raw_path_type missing')
    paths = frame.raw_path_type.astype(str).to_numpy()
    require(set(paths) <= set(LINK_OF_PATH) | set(NO_LINK_PATHS), 'Unknown raw_path_type; no guessed link mapping')
    idx = np.array([LINKS.index(LINK_OF_PATH[p]) if p in LINK_OF_PATH else -1 for p in paths], dtype=np.int8)
    x = np.full((len(frame), 2), np.nan, dtype=np.float64)
    for k, link in enumerate(LINKS):
        require(set(RF_COLS[link]) <= set(frame), 'Missing observed RF columns for ' + link)
        m = idx == k
        if not m.any(): continue
        v = frame.loc[m, list(RF_COLS[link])].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        require(np.isfinite(v).all(), 'Nonfinite observed RF measurement; no normal-value imputation')
        require(((v[:, 1] >= 0) & (v[:, 1] <= 1)).all(), 'Frame loss rate outside [0,1]')
        x[m] = v
    return idx, x


def fit_reference(train):
    y = labels(train); link, x = observed_rf(train); result = {}
    for k, name in enumerate(LINKS):
        v = x[(y == 0) & (link == k)]
        require(len(v) >= 20, f'Insufficient train NORMAL observed-link support: {name}, n={len(v)}')
        mu, sd = v.mean(0), v.std(0, ddof=1)
        require(np.isfinite(sd).all() and np.all(sd > 1e-12), 'Degenerate train NORMAL scale: ' + name)
        result[name] = dict(n=int(len(v)), columns=list(RF_COLS[name]), mean=mu.tolist(), sample_std=sd.tolist())
    return dict(fit_population='train NORMAL only; observed link only', links=result, ddof=1,
                directions=['mean_minus_snr', 'loss_minus_mean'], k=REFERENCE_K,
                note='Includes benign variability; no removal based on attack/effect/nominal flags')


def standardized_rf(frame, reference):
    link, x = observed_rf(frame); z = np.full_like(x, np.nan)
    for k, name in enumerate(LINKS):
        m = link == k; r = reference['links'][name]
        z[m] = (x[m] - np.asarray(r['mean'])) / np.asarray(r['sample_std'])
        z[m, 0] *= -1
    return link, z


def finite_rf_scores(z, links, episodes, ends):
    z = np.asarray(z, float); links = np.asarray(links); episodes = np.asarray(episodes); ends = np.asarray(ends)
    require(z.ndim == 2 and z.shape[1] == 2 and links.shape == (len(z),) and episodes.shape == links.shape, 'Invalid RF score dimensions')
    require(ends.dtype.kind in 'iu' and len(ends) > 0 and ends.min() >= 9 and ends.max() < len(z), 'Invalid/missing L10 endpoints')
    require(np.isin(links, [-1, 0, 1, 2, 3]).all() and np.isfinite(z[links >= 0]).all(), 'Invalid observed link/standardized value')
    ix = ends[:, None] - np.arange(9, -1, -1)
    require(np.all(episodes[ix] == episodes[ends, None]), 'RF history crosses episode boundary')
    current = np.zeros(len(ends)); m = links[ends] >= 0
    current[m] = np.maximum(0, z[ends[m]].max(1))
    state = np.zeros((len(ends), 4, 2))
    # Recompute bounded ten-window recursion; missing links HOLD, never normal-impute.
    for lag in range(10):
        rows = ix[:, lag]
        for k in range(4):
            selected = np.flatnonzero(links[rows] == k)
            state[selected, k] = np.maximum(0, state[selected, k] + z[rows[selected]] - REFERENCE_K)
    return current, state.max(axis=(1, 2))


def fpr_threshold(normal_scores, target):
    x = np.asarray(normal_scores, float)
    require(x.ndim == 1 and len(x) and np.isfinite(x).all() and np.all(x >= 0) and 0 < target < 1, 'Invalid NORMAL calibration scores/budget')
    a = np.sort(x); budget = int(np.floor(len(x) * target)); t = float(a[len(a)-budget-1])
    require(int(np.sum(x > t)) <= budget, 'Validation budget exceeded')
    return t


def binary_metrics(y, alarm):
    y = np.asarray(y); a = np.asarray(alarm, bool)
    require(y.shape == a.shape and y.ndim == 1 and np.isin(y, range(6)).all(), 'Invalid binary population')
    tp = int(np.sum(a & (y == 1))); fp = int(np.sum(a & (y != 1))); fn = int(np.sum(~a & (y == 1)))
    def rate(mask): return float(a[mask].mean()) if mask.any() else None
    return dict(rf_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                rf_precision=tp/(tp+fp) if tp+fp else 0., rf_recall=rate(y == 1),
                normal_rf_fpr=rate(y == 0), rest_rf_fpr=rate(y != 1), other_attack_rf_mislabel_rate=rate(y >= 2),
                tp=tp, fp=fp, fn=fn, tn=int(np.sum(~a & (y != 1))))


class BoundRun:
    """Only consume explicitly inventoried artifacts; omitted model caches are not globally checked."""
    def __init__(self, folder, status, audit):
        self.root = Path(folder).resolve(); self.audit = audit
        p = self.root/'run_manifest.json'; require(p.is_file(), 'Missing completed run manifest: '+str(p))
        self.manifest = read_json(p); audit[str(p)] = digest(p)
        require(self.manifest.get('status') == status, 'Wrong/incomplete run status: '+str(p))
        self.hashes = dict(self.manifest.get('output_sha256', {}))
        if not self.hashes:
            self.hashes = {r['file']: r['sha256'] for r in self.manifest.get('outputs', [])}
        require(self.hashes, 'No artifact inventory in '+str(p))
    def file(self, rel):
        p = safe_child(self.root, rel); h = self.hashes.get(rel)
        require(isinstance(h, str) and len(h) == 64 and p.is_file(), 'Required inventoried input missing: '+str(p))
        require(digest(p) == h, 'Input SHA256 mismatch: '+str(p))
        self.audit[str(p)] = h; return p
    def json(self, rel): return read_json(self.file(rel))


def checked_source(root, relative, pins, audit):
    relative = str(relative).replace('\\', '/'); p = safe_child(root, relative)
    matches = {v for k, v in pins.items() if str(k).replace('\\', '/').endswith('/'+relative) or str(k).replace('\\','/') == relative}
    require(len(matches) == 1, 'Missing/ambiguous recorded Step59 source hash: '+relative)
    h = next(iter(matches)); require(p.is_file(), 'Original input missing: '+str(p))
    require(digest(p) == h, 'Original source changed: '+str(p)); audit[str(p)] = h; return p


def canonical_endpoint(frame):
    f = frame.copy(); step = 'episode_step' if 'episode_step' in f else 'sequence_end_step'
    require({'episode_id', step} <= set(f), 'Endpoint identity columns missing')
    f['_step'] = ints(f[step], step); f['_episode'] = f.episode_id.astype(str)
    return pd.MultiIndex.from_frame(f[['_episode', '_step']])


def align_endpoint(other, endpoint):
    source, target = canonical_endpoint(other), canonical_endpoint(endpoint)
    require(source.is_unique and target.is_unique and len(source) == len(target), 'Endpoint population missing/duplicated')
    order = source.get_indexer(target); require(np.all(order >= 0), 'Endpoint keys differ')
    f = other.iloc[order].reset_index(drop=True); y = labels(endpoint)
    if 'label' in f: require(np.array_equal(f.label.astype(str).to_numpy(), endpoint.label.astype(str).to_numpy()), 'Endpoint label strings differ')
    for k in ('y_true', 'label_id', 'true_label_id'):
        if k in f: require(np.array_equal(ints(f[k], k), y), 'Endpoint true label mapping differs: '+k)
    for k in ('window_id','sequence_end_window_id'):
        if k in f: require(np.array_equal(ints(f[k], k), ints(endpoint.window_id, 'window_id')), 'Endpoint window identity differs')
    if 'sequence_length' in f: require(np.all(ints(f.sequence_length, 'sequence_length') == 10), 'Not L10 context')
    if 'sequence_start_step' in f: require(np.array_equal(ints(f.sequence_start_step, 'sequence_start_step'), ints(endpoint.episode_step,'episode_step')-9), 'Context start differs')
    return f


def locate_rule(root, explicit, dataset, references=None):
    if explicit: return resolve(root, explicit)
    matches = []
    for p in (root/'outputs/shared_nominal_v2/16_observable_rule_baseline').glob('run_*/run_manifest.json'):
        try:
            m = read_json(p); c = m.get('source_contract', m.get('dataset_contract', {}))
            if m.get('status') == 'COMPLETE' and c.get('dataset_id') == dataset and not m.get('calibrate_only'): matches.append(p.parent)
        except (OSError, ValueError): continue
    if len(matches) > 1 and references:
        # Reuse a Step59-recorded Rule reference by artifact identity, never by score/mtime.
        jobs=[j for j in references.get('jobs',[]) if j.get('group')=='Reference16_Rule_L10']
        if len(jobs)==1:
            selected=[]
            for folder in matches:
                p=folder/'test_predictions_Rule_L10.csv'
                if p.is_file() and digest(p)==jobs[0].get('sha256'):selected.append(folder)
            if len(selected)==1:return selected[0]
    require(len(matches) == 1, f'Found {len(matches)} completed current Step16 runs. Use --rule-run to name the intended run; no newest/best selection.')
    return matches[0]


def prepare(root, args):
    audit = {}; r59 = BoundRun(resolve(root, args.step59_run), 'RAW_SEMANTIC_BASELINE_COMPLETE', audit)
    p59 = r59.json('protocol.json')
    require(p59.get('version') == 'shared_nominal_raw_semantic_v2' and p59.get('dataset_id') == DATASET_ID, 'Requires current Step59 v2, not v1/legacy data')
    require(p59.get('identity') == obj_hash({k:v for k,v in p59.items() if k != 'identity'}), 'Step59 protocol identity differs')
    require(p59.get('identity') == r59.manifest.get('protocol_identity'), 'Step59 manifest protocol binding differs')
    require(tuple(p59['labels']) == LABELS and tuple(p59['seeds']) == FIT_SEEDS[3] and 'RawScoped' in p59['representations'], 'Step59 class/seed/representation contract differs')
    require(tuple(p59['semantic_fields']) == FIELDS and p59['link_of_path'] == LINK_OF_PATH, 'Step59 semantic/path contract differs')
    base = checked_source(root, 'dataset/shared_nominal_v2/build_manifest.json', p59['inputs'], audit)
    build = read_json(base); require(build.get('dataset_id') == DATASET_ID and len(build['sources']) == 6, 'Wrong Step7 dataset/source list')
    mapper = checked_source(root, 'shared_nominal.py', p59['inputs'], audit)
    require(digest(mapper) == PROJECTION_SHA, 'Unknown mapper version; no automatic adaptation')
    ia = r59.json('input_audit.json'); require(ia.get('dataset_id') == DATASET_ID, 'Step59 input audit dataset differs')
    rewrite = ia['split_integrity']['source_id_rewrite_applied']; require(isinstance(rewrite, bool), 'Unknown source ID rewrite contract')
    frames = {}; ends = {}; endpoints = {}
    for split in ('train','val','test'):
        path = checked_source(root, 'dataset/shared_nominal_v2/splits/'+split+'_observable.csv', p59['inputs'], audit)
        f = read_csv(path, dtype=str, usecols=list(META)+['path_type']); labels(f)
        require(f[['episode_id','episode_step','window_id']].duplicated().sum() == 0, 'Duplicate split row keys')
        ix = endpoint_indices(f); ep = f.iloc[ix].reset_index(drop=True)
        expected = read_csv(r59.file(f'endpoints_{split}_L10.csv'), dtype=str)
        aligned = align_endpoint(expected, ep)
        require(np.array_equal(ints(aligned.source_row_index,'source_row_index'), ix), 'Step59 endpoint-to-row linkage differs')
        require(len(f) == ia['alignment'][split]['rows'] and len(ix) == ia['alignment'][split]['l10_endpoints'], 'Step59 population audit differs')
        frames[split], ends[split], endpoints[split] = f, ix, ep
    for a,b in (('train','val'),('train','test'),('val','test')):
        require(not set(frames[a].episode_id) & set(frames[b].episode_id), 'Episode split overlap')
    rawpaths=[]
    for entry in build['sources']:
        require(entry['label'] in LABELS, 'Unknown source label')
        path=checked_source(root, 'dataset/'+entry['raw_audit']['file'], p59['inputs'], audit)
        rawpaths.append((path, entry['label']))
    require(len({v for _,v in rawpaths}) == 6, 'Repeated source class')
    encoder = r59.json('encoder_RawScoped.json')
    cat = ['raw_'+x for x in FIELDS[:11]]; rawfields = p59['raw_fields']
    require(encoder['categorical'] == cat and len(set(rawfields)) == len(rawfields), 'Unexpected RawScoped encoder categories/raw fields')
    require(encoder['numeric'] == p59['scoped_fields'] and set(encoder['numeric']) == (set(rawfields)-set(cat)) | {'link_observed_'+x for x in LINKS}, 'Unexpected RawScoped numeric order')
    expected_names=list(encoder['numeric'])
    for c in cat: expected_names += [c+'='+v for v in encoder['categories'][c]]+[c+'=<UNSEEN>']
    require(encoder['window_feature_names'] == expected_names, 'Encoder feature order differs')
    hgb=[]
    for seed in FIT_SEEDS[3]:
        rel=f'jobs/RawScoped_L10_HGB_seed{seed}'; complete=r59.json(rel+'/complete.json')
        expected_job=obj_hash(dict(protocol=p59['identity'],key=f'RawScoped_L10_HGB_seed{seed}',encoder=encoder))
        require(complete['identity'] == expected_job, 'HGB job identity differs')
        model=r59.file(rel+'/model.joblib')
        require(digest(model) == complete['files']['model.joblib'], 'HGB checkpoint differs from completed training job')
        val=r59.file(rel+'/validation_predictions.npz')
        require(digest(val) == complete['files']['validation_predictions.npz'], 'HGB validation replay source differs')
        # Verify existence/inventory now; test prediction contents are opened only after lock.
        require(rel+'/test_predictions.npz' in r59.hashes and safe_child(r59.root,rel+'/test_predictions.npz').is_file(), 'Missing HGB test replay artifact')
        hgb.append(dict(seed=seed,rel=rel,model=model,val=val))
    r62=BoundRun(resolve(root,args.step62_run),'TEMPORAL_INFORMATION_CONTROL_COMPLETE',audit);p62=r62.json('protocol.json')
    require(p62.get('dataset_id') == DATASET_ID and p62.get('identity') == obj_hash({k:v for k,v in p62.items() if k!='identity'}) == r62.manifest.get('protocol_identity'), 'Step62 dataset/protocol mismatch')
    require(tuple(p62['seeds']) == FIT_SEEDS[4] and 'actual_history' in p62['conditions'], 'Need all ten Step62 actual_history fits')
    sc=p62['source_contract']
    require(sc['dataset_id']==DATASET_ID,'Step62 source dataset differs')
    for split in ('train','val','test'):
        name=f'splits/{split}_observable.csv';expected_hash=sc['input_sha256'].get(name)
        if expected_hash is not None:
            require(expected_hash==audit[str((root/'dataset/shared_nominal_v2'/name).resolve())], 'Step62 split source differs: '+split)
        ep62=read_csv(r62.file(f'endpoint_audit_{split}.csv'),dtype=str);aligned=align_endpoint(ep62,endpoints[split])
        require(np.array_equal(ints(aligned.sequence_end_row,'sequence_end_row'),ends[split]),'Step62 endpoint row differs')
    env_references=p59.get('references',{})
    r16=BoundRun(locate_rule(root,args.rule_run,DATASET_ID,env_references),'COMPLETE',audit)
    c16=r16.manifest.get('source_contract',r16.manifest.get('dataset_contract',{}))
    require(c16.get('dataset_id')==DATASET_ID,'Step16 dataset differs')
    for key in ('step8_run_id','step9_run_id','step10_run_id'):
        if key in c16:require(c16[key]==sc[key],'Step16 encoder/sequence lineage mismatch: '+key)
    for split in ('val','test'):
        require(f'{split}_predictions_Rule_L10.csv' in r16.hashes,'Step16 L10 predictions missing')
        for seed in FIT_SEEDS[4]:require(f'seed_{seed}/actual_history/{split}_predictions.csv' in r62.hashes,'Step62 predictions missing')
    return dict(root=root,audit=audit,r59=r59,p59=p59,r62=r62,p62=p62,r16=r16,frames=frames,ends=ends,
                endpoints=endpoints,rawpaths=rawpaths,rewrite=rewrite,encoder=encoder,hgb=hgb)


def load_raw(env, split, rf_only=False):
    wanted=env['frames'][split]; root=env['root']
    rfcols=['raw_path_type']+[x for l in LINKS for x in RF_COLS[l]]
    fields=rfcols if rf_only else list(env['p59']['raw_fields'])+list(ORIGIN_FLAGS)
    cols=list(dict.fromkeys(list(META)+fields)); keys=['label','episode_id','episode_step','window_id']
    target=pd.MultiIndex.from_frame(wanted[keys]); require(target.is_unique,'Nonunique raw alignment target')
    # Store only the requested split. Raw input files themselves contain all splits.
    pieces=[];positions=[];seen=np.zeros(len(wanted),dtype=bool)
    for path,cls in env['rawpaths']:
        print(f'  raw {split}: {cls}',flush=True)
        for chunk in read_csv(path,dtype=str,usecols=cols,chunksize=5000):
            require(chunk.label.eq(cls).all(),'Raw source label differs from build manifest')
            if env['rewrite']:chunk['episode_id']=chunk.label+'::'+chunk.episode_id
            pos=target.get_indexer(pd.MultiIndex.from_frame(chunk[keys]));keep=pos>=0
            if not keep.any():continue
            idx=pos[keep];part=chunk.loc[keep].reset_index(drop=True)
            require(len(np.unique(idx))==len(idx) and not seen[idx].any(),'Duplicate raw-source row for requested split')
            expected=wanted.iloc[idx].reset_index(drop=True)
            for c in META:require(np.array_equal(part[c].astype(str),expected[c].astype(str)),'Raw metadata mismatch: '+c)
            require(np.array_equal(part.raw_path_type.astype(str),expected.path_type.astype(str)), 'Raw/semantic path mismatch')
            seen[idx]=True;pieces.append(part);positions.append(idx)
    require(seen.all(),'Missing raw companion rows in '+split)
    raw=pd.concat(pieces,ignore_index=True).iloc[np.argsort(np.concatenate(positions))].reset_index(drop=True)
    return raw


def scoped_hgb_matrix(raw, encoder):
    numeric=encoder['numeric'];categorical=encoder['categorical'];f=raw.copy()
    for c in numeric:
        if c.startswith('link_observed_'):continue
        require(c in f,'HGB raw input missing: '+c)
        f[c]=pd.to_numeric(f[c],errors='raise').astype(float)
        require(np.isfinite(f[c]).all(),'Nonfinite original HGB input: '+c)
    for dest,origin in (('source_unverified_flag','source_unverified_benign_flag'),('policy_inconsistent_flag','valid_looking_benign_flag')):
        a=pd.to_numeric(f[origin],errors='raise').to_numpy(dtype=float)
        require(np.isin(a,[0.,1.]).all(),'Invalid cause-neutral coalescing flag')
        f[dest]=np.maximum(f[dest].to_numpy(),a)
    paths=f.raw_path_type.astype(str).to_numpy();require(set(paths)<=set(LINK_OF_PATH)|set(NO_LINK_PATHS),'Unknown HGB raw path')
    for link,columns in LINK_COLS.items():
        observed=np.array([LINK_OF_PATH.get(p)==link for p in paths]);f['link_observed_'+link]=observed.astype(float)
        f.loc[~observed,list(columns)]=0.
    out=np.zeros((len(f),len(encoder['window_feature_names'])),dtype=np.float64)
    out[:,:len(numeric)]=f[numeric].to_numpy(dtype=float);j=len(numeric);rows=np.arange(len(f))
    for c in categorical:
        vocab=encoder['categories'][c];index={v:i for i,v in enumerate(vocab)}
        values=f[c].astype(str);require(values.str.len().gt(0).all(),'Blank categorical value')
        ix=values.map(index).fillna(len(vocab)).to_numpy(dtype=int)
        out[rows,j+ix]=1.;j+=len(vocab)+1
    require(np.isfinite(out).all(),'Nonfinite HGB encoded matrix')
    return out


def aggregate_batch(matrix, ends):
    ix=ends[:,None]-np.arange(9,-1,-1);x=matrix[ix]
    return np.concatenate([x.mean(1),x.std(1,ddof=0),x.min(1),x.max(1),x[:,-1]],axis=1)


def hgb_probabilities(env, raw, split, out, batch=512):
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    matrix=scoped_hgb_matrix(raw,env['encoder']);ends=env['ends'][split];y=labels(env['endpoints'][split]);all_prob=[];all_native=[];reports=[]
    for j in env['hgb']:
        if split=='test':assert_test_gate(out)
        path=env['r59'].file(j['rel']+('/test_predictions.npz' if split=='test' else '/validation_predictions.npz'))
        with np.load(path,allow_pickle=False) as z:
            yt=z['y_true'].copy();yp=z['y_pred'].copy()
        require(np.array_equal(yt,y) and yp.shape==y.shape and np.isin(yp,range(6)).all(),'HGB stored endpoint labels differ')
        print(f'  HGB probability replay {split}, seed={j["seed"]}',flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error',InconsistentVersionWarning)
            model=joblib.load(j['model'])
        require(type(model).__name__=='HistGradientBoostingClassifier' and np.array_equal(model.classes_,np.arange(6)), 'Unexpected HGB checkpoint/class order')
        require(model.n_features_in_==matrix.shape[1]*5,'HGB feature count mismatch')
        p=np.empty((len(y),6),dtype=float)
        for a in range(0,len(y),batch):
            p[a:a+batch]=model.predict_proba(aggregate_batch(matrix,ends[a:a+batch]))
        require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.allclose(p.sum(1),1,atol=1e-8,rtol=0),'Invalid HGB probabilities')
        mismatch=int(np.sum(p.argmax(1)!=yp))
        require(mismatch==0,f'HGB {split} replay differs at {mismatch} endpoints; do not recalibrate/accept altered predictions')
        all_prob.append(p[:,1]);all_native.append(yp==1)
        reports.append(dict(seed=j['seed'],split=split,endpoints=len(y),hard_label_mismatches=mismatch,source_sha256=digest(j['model']),feature_count=int(model.n_features_in_)))
        del model
    write_json(out/f'hgb_replay_{split}.json',reports)
    return all_prob,all_native


def score_split(env, raw, split, reference, out):
    if split=='test':assert_test_gate(out)
    ep=env['endpoints'][split];ends=env['ends'][split];link,z=standardized_rf(raw,reference)
    current,cusum=finite_rf_scores(z,link,raw.episode_id.to_numpy(),ends)
    score=[current,cusum];native=[np.full(len(ep),-1,dtype=np.int8),np.full(len(ep),-1,dtype=np.int8)]
    rule=align_endpoint(read_csv(env['r16'].file(f'{split}_predictions_Rule_L10.csv'),dtype={'episode_id':str}),ep)
    column='evidence_RF_LINK_JAMMING';require(column in rule,'Step16 RF evidence is absent; hard labels cannot substitute for evidence scores')
    rule_pred=ints(rule.y_pred,'y_pred');require(np.isin(rule_pred,range(6)).all(),'Invalid Rule native predicted class')
    score.append(pd.to_numeric(rule[column],errors='raise').to_numpy(dtype=float));native.append(rule_pred==1)
    hprob,hnative=hgb_probabilities(env,raw,split,out);score.extend(hprob);native.extend(hnative)
    for seed in FIT_SEEDS[4]:
        p=align_endpoint(read_csv(env['r62'].file(f'seed_{seed}/actual_history/{split}_predictions.csv'),dtype={'episode_id':str}),ep)
        require('condition' in p and p.condition.eq('actual_history').all() and np.all(ints(p.seed,'seed')==seed),'Step62 fit/condition differs')
        probs=p[[f'prob_{i}' for i in range(6)]].to_numpy(dtype=float)
        require(np.isfinite(probs).all() and np.all((probs>=0)&(probs<=1)) and np.allclose(probs.sum(1),1,atol=5e-7,rtol=0),'Invalid Step62 probabilities')
        pred=ints(p.y_pred,'y_pred');require(np.isin(pred,range(6)).all(),'Invalid TST native predicted class')
        score.append(probs[:,1]);native.append(pred==1)
    scores=np.stack(score);native=np.stack(native).astype(np.int8)
    require(scores.shape==(len(SLOTS),len(ep)) and np.isfinite(scores).all() and np.all(scores>=0),'Invalid score matrix')
    ix=ends[:,None]-np.arange(9,-1,-1)
    cov=ep[list(META)].copy();cov['current_link']=link[ends];cov['current_rf_observed']=(link[ends]>=0).astype(int)
    cov['observed_rf_windows_L10']=(link[ix]>=0).sum(1)
    for k,l in enumerate(LINKS):cov['observed_'+l+'_windows_L10']=(link[ix]==k).sum(1)
    write_csv(out/f'coverage_{split}.csv',cov)
    write_npz(out/f'scores_{split}.npz',scores=scores,native_rf=native,methods=np.array([x[0] for x in SLOTS]),seeds=np.array([x[1] for x in SLOTS]),y_true=labels(ep),episode_id=ep.episode_id.astype(str).to_numpy(dtype=str),episode_step=ints(ep.episode_step,'step'))
    # Numeric components allow the two raw score arms to be independently reconstructed.
    write_npz(out/f'raw_rf_components_{split}.npz',z=z,observed_link=link,endpoint_rows=ends,episode_id=raw.episode_id.to_numpy(dtype=str))
    return scores,native


def seal_stage(out,name,files,identity):
    write_json(out/(name+'_complete.json'),dict(protocol_identity=identity,completed_at_utc=now(),artifacts={r:digest(safe_child(out,r)) for r in files}))


def verify_stage(out,name,identity=None):
    p=out/(name+'_complete.json')
    if not p.exists():return False
    d=read_json(p);require(d.get('artifacts'),'Empty stage inventory')
    if identity is not None:require(d['protocol_identity']==identity,'Stage protocol identity changed')
    for r,h in d['artifacts'].items():require(safe_child(out,r).is_file() and digest(safe_child(out,r))==h,'Stage artifact changed: '+r)
    return True


def assert_test_gate(out):
    out=Path(out);require(verify_stage(out,'calibration'),'Test scores forbidden before validation calibration lock')
    require(read_json(out/'thresholds.json')['test_scores_read_before_lock'] is False,'Invalid calibration lock')


def calibration(env, reference, scores, out, identity):
    y=labels(env['endpoints']['val']);rows=[];thresholds=np.empty((len(TARGETS),len(SLOTS)))
    for q,target in enumerate(TARGETS):
        for i,(method,seed) in enumerate(SLOTS):
            v=scores[i,y==0];t=fpr_threshold(v,target);thresholds[q,i]=t
            rows.append(dict(method=method,seed=seed,target_normal_rf_fpr=target,threshold=t,comparison='strict >',normal_validation_endpoints=len(v),budget=int(np.floor(len(v)*target)),achieved_normal_rf_alarms=int(np.sum(v>t)),achieved_normal_rf_fpr=float(np.mean(v>t))))
    write_json(out/'thresholds.json',dict(protocol_identity=identity,created_at_utc=now(),test_scores_read_before_lock=False,calibration='NORMAL val endpoints only; same val previously used for checkpoint selection',slots=SLOTS,targets=TARGETS,thresholds=thresholds))
    write_csv(out/'thresholds.csv',rows)
    write_json(out/'calibration_input_audit.json',env['audit'])
    files=['normal_reference.json','thresholds.json','thresholds.csv','scores_val.npz','raw_rf_components_val.npz','coverage_val.csv','hgb_replay_val.json','calibration_input_audit.json']
    seal_stage(out,'calibration',files,identity);return thresholds


def metrics_array(alarm_counts,class_sizes):
    # alarm_counts [..., 6], class_sizes broadcastable to [..., 6].
    a=alarm_counts;n=class_sizes;tp=a[...,1];fp=a.sum(-1)-tp;fn=n[...,1]-tp
    divide=lambda num,den,zero=0.:np.divide(num,den,out=np.full(np.broadcast_shapes(np.shape(num),np.shape(den)),zero,dtype=float),where=den>0)
    return np.stack((divide(2*tp,2*tp+fp+fn,np.nan),divide(tp,tp+fp),divide(tp,n[...,1],np.nan),
                     divide(a[...,0],n[...,0],np.nan),divide(fp,n.sum(-1)-n[...,1],np.nan),
                     divide(a[...,2:].sum(-1),n[...,2:].sum(-1),np.nan)),axis=-1)


def statistics(y,episodes,scores,thresholds,out,draws=20000,bootstrap_seed=2026100970):
    y=np.asarray(y);episodes=np.asarray(episodes);ids,ei=np.unique(episodes,return_inverse=True);ne=len(ids)
    alarm=scores[None,:,:]>thresholds[:,:,None];class_sizes=np.zeros((ne,6),dtype=np.int64)
    np.add.at(class_sizes,(ei,y),1)
    require(np.all((class_sizes>0).sum(1)==1) and np.all(class_sizes.sum(0)>0),'Bootstrap requires class-pure episodes and all six classes')
    strata=[np.flatnonzero(class_sizes[:,c]>0) for c in range(6)]
    counts=np.zeros((ne,len(TARGETS),len(SLOTS),6),dtype=np.int64)
    for q in range(len(TARGETS)):
        for i in range(len(SLOTS)):np.add.at(counts[:,q,i],(ei,y),alarm[q,i].astype(np.int64))
    point_by_fit=metrics_array(counts.sum(0),class_sizes.sum(0)[None,None,:])
    cuts=np.cumsum([0]+[len(s) for s in FIT_SEEDS])
    mean_methods=lambda x:np.stack([x[...,cuts[k]:cuts[k+1],:].mean(-2) for k in range(len(METHODS))],axis=-2)
    point=mean_methods(point_by_fit);flat=counts.reshape(ne,-1).astype(float);rng=np.random.default_rng(bootstrap_seed)
    dist=np.empty((draws,len(TARGETS),len(METHODS),len(METRICS)))
    for start in range(0,draws,250):
        n=min(250,draws-start);w=np.zeros((n,ne))
        for ix in strata:w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=n)
        a=(w@flat).reshape(n,len(TARGETS),len(SLOTS),6);den=(w@class_sizes)[:,None,None,:]
        dist[start:start+n]=mean_methods(metrics_array(a,den))
        if start%2500==0:print(f'  bootstrap {start+n:,}/{draws:,}',flush=True)
    long=[];table=[];by_fit=[];mis=[]
    all_counts=counts.sum(0);sizes=class_sizes.sum(0)
    for q,target in enumerate(TARGETS):
        for k,method in enumerate(METHODS):
            row=dict(method=method,target_normal_rf_fpr=target,fits=len(FIT_SEEDS[k]),endpoints=len(y),episodes=ne)
            for j,metric in enumerate(METRICS):
                samples=dist[:,q,k,j];ok=np.isfinite(samples).all();lo,hi=np.quantile(samples,[.025,.975]) if ok else (np.nan,np.nan)
                values=point_by_fit[q,cuts[k]:cuts[k+1],j]
                long.append(dict(method=method,target_normal_rf_fpr=target,metric=metric,estimate=point[q,k,j],ci95_low=lo,ci95_high=hi,ci_status='ok' if ok else 'undefined_denominator',fixed_fit_sd=float(values.std(ddof=1)) if len(values)>1 else None))
                row[metric]=point[q,k,j]
            table.append(row)
            for cls in range(6):
                vals=all_counts[q,cuts[k]:cuts[k+1],cls]/sizes[cls]
                mis.append(dict(method=method,target_normal_rf_fpr=target,true_class=LABELS[cls],endpoints=int(sizes[cls]),rf_alarm_rate=float(vals.mean()),interpretation='RF recall' if cls==1 else 'RF false alarm or other-attack misidentification'))
        for i,(method,seed) in enumerate(SLOTS):
            by_fit.append(dict(method=method,seed=seed,target_normal_rf_fpr=target,**dict(zip(METRICS,point_by_fit[q,i]))))
    contrasts=[];samples=[]
    # Planned four primary comparisons, not a test-selected winning model.
    for q,target in enumerate(TARGETS):
        for k in (0,1):
            d=dist[:,q,4,0]-dist[:,q,k,0];lo,hi=np.quantile(d,[.025,.975]);bl,bh=np.quantile(d,[.025/4,1-.025/4])
            contrasts.append(dict(reference=METHODS[4],comparator=METHODS[k],metric='rf_f1',target_normal_rf_fpr=target,difference=point[q,4,0]-point[q,k,0],ci95_low=lo,ci95_high=hi,family_size=4,family_adjusted_low=bl,family_adjusted_high=bh,interpretation='Not an architecture-only/representation-only comparison; fixed fits and thresholds'))
            samples.append(d)
    write_csv(out/'model_summary.csv',table);write_csv(out/'metric_intervals.csv',long);write_csv(out/'metrics_by_fit.csv',by_fit)
    write_csv(out/'rf_alarm_by_true_class.csv',mis);write_csv(out/'primary_comparisons.csv',contrasts)
    write_npz(out/'bootstrap_components.npz',episodes=ids,class_sizes=class_sizes,alarm_counts=counts,methods=np.array([x[0] for x in SLOTS]),seeds=np.array([x[1] for x in SLOTS]),targets=np.array(TARGETS))
    write_npz(out/'bootstrap_distribution.npz',metrics=dist,primary_differences=np.stack(samples,axis=1),method_names=np.array(METHODS),metric_names=np.array(METRICS),targets=np.array(TARGETS))
    return table,contrasts


@contextmanager
def run_lock(folder):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.step70.lock').open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise ValueError('Another Step70 process owns this output') from e
        yield


def recheck(audit):
    for p,h in audit.items():require(Path(p).is_file() and digest(p)==h,'Input changed during analysis: '+p)


def package(out,status,extra=None):
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ('run_manifest.json','.step70.lock') and not p.name.endswith('.tmp'))
    m=dict(status=status,version=VERSION,updated_at_utc=now(),output_sha256={p.relative_to(out).as_posix():digest(p) for p in files},**(extra or {}))
    write_json(out/'run_manifest.json',m)
    dest=out.with_name(out.name+('_FAILED_review.zip' if status=='FAILED' else '_review.zip'));t=dest.with_name(dest.name+'.tmp')
    with zipfile.ZipFile(t,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'run_manifest.json']:z.write(p,p.relative_to(out).as_posix())
    os.replace(t,dest);return dest


def plan_protocol(env,args):
    import sklearn,joblib
    # All metadata and upstream identities fixed before calibration; no test scores.
    return dict(version=VERSION,dataset_id=DATASET_ID,script_sha256=digest(Path(__file__)),
                step59_identity=env['p59']['identity'],step62_identity=env['p62']['identity'],
                step16_manifest_sha256=digest(env['r16'].root/'run_manifest.json'),
                input_paths=dict(step59=str(env['r59'].root),step62=str(env['r62'].root),rule=str(env['r16'].root)),
                methods=METHODS,fit_seeds=FIT_SEEDS,rf_measurements=RF_COLS,link_of_path=LINK_OF_PATH,
                reference='All train NORMAL observed-link means/sample std; no effect conditioning',
                cusum=dict(length=10,k=REFERENCE_K,source=SOURCE_REFERENCE,
                           algorithm='Within [t-9,t] recompute max(0,C+z-k) independently for each link and each of two measurements; hold if link unobserved; max across link/components at t.',
                           signed_increments=True,no_pre_rectification=True,history_outside_L10=False,arl_guarantee=False),
                targets=TARGETS,score_definition='RF class probability (index1), or RF pre-fusion rule evidence, or raw statistic; never 1-pNORMAL',
                calibration='Each method/fit NORMAL-val endpoint quantile; strict >, ties conservative; val reused from checkpoint selection',
                primary='RF positive-class F1 TST62 minus each of two new raw baselines at two budgets, four-comparison family',
                metric_conventions='RF_LINK_JAMMING is positive; all other five scenario classes negative. Precision=0 when no RF alarms; a missing actual class denominator remains undefined. Normal RF FPR is not total any-attack FPR.',
                bootstrap=dict(draws=args.bootstrap,seed=2026100970,unit='class-stratified paired whole episode',fit_seed_resampling=False,conditioned_on='fixed model fits, normalization, validation thresholds and realization'),
                runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,threads=args.threads),
                population={s:dict(rows=len(env['frames'][s]),endpoints=len(env['ends'][s]),episodes=env['endpoints'][s].episode_id.nunique()) for s in env['frames']},
                scope=['Post-hoc internal RF-vs-rest comparison on previously inspected benchmark; not external confirmation.',
                       'Raw inputs are simulated monitor measurements, not RF waveform samples.',
                       'Raw detector sees only two measurements of selected link; Rule/TST and HGB use more information and different training objectives.',
                       'All episode scenario labels retained including unobserved RF windows; not instantaneous attack-active detection.',
                       'Raw measurement files contain all splits; only train NORMAL selected rows fit normalization, only val scores calibrate; test score computation begins after lock.',
                       'No new data, labels, neural training, neural inference, latency, onset or recovery metrics.',
                       'HGB-only CPU probability inference required because Step59 stored hard labels; exact hard-label replay required before use.',
                       'A threshold-calibrated Rule RF arm is distinct from native Rule16 multi-class decision. Original configuration is not overwritten.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--step59-run',type=Path,default=DEFAULT59)
    p.add_argument('--step62-run',type=Path,default=DEFAULT62)
    p.add_argument('--rule-run',type=Path)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--bootstrap',type=int,default=20000)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(argv)
    if args.self_test:return self_test()
    require(1<=args.threads<=8 and args.bootstrap>=1000,'threads must be 1..8; bootstrap at least1000')
    root=args.project_root.resolve();out=resolve(root,args.output_dir);started=time.perf_counter()
    # Refuse overlap with data and the three upstream experiment trees.
    for b in (root/'dataset',resolve(root,args.step59_run),resolve(root,args.step62_run),root/'outputs/shared_nominal_v2/16_observable_rule_baseline'):
        b=b.resolve();require(out!=b and out not in b.parents and b not in out.parents,'Output overlaps an upstream input directory')
    env=None;bound=False
    try:
        from threadpoolctl import threadpool_limits
        with run_lock(out),threadpool_limits(limits=args.threads):
            print('Step70: RF jamming only; raw current/CUSUM10 + existing fixed references.',flush=True)
            print('[1/5] Verify original datasets, split/endpoint identities and existing runs ...',flush=True)
            env=prepare(root,args);protocol=plan_protocol(env,args);protocol['identity']=obj_hash(protocol)
            if (out/'protocol.json').exists():require(read_json(out/'protocol.json')==safe_value(protocol),'Step70 protocol/runtime/input changed; preserve old run and specify a separate --output-dir')
            else:
                require(not [f for f in out.iterdir() if f.name!='.step70.lock'],'Output contains unrelated files')
                write_json(out/'protocol.json',protocol)
            bound=True;identity=protocol['identity']
            # Verify before touching any completed artifact; never rewrite input_audit on resume.
            if (out/'run_manifest.json').exists() and read_json(out/'run_manifest.json').get('status')=='STEP70_RF_STATISTICS_COMPLETE':
                old=BoundRun(out,'STEP70_RF_STATISTICS_COMPLETE',{})
                for rel in old.hashes:old.file(rel)
                recheck(read_json(out/'input_audit.json'))
                archive=out.with_name(out.name+'_review.zip')
                if not archive.is_file():
                    # Rebuild ZIP bytes only, preserving the completed manifest and artifacts.
                    temp=archive.with_name(archive.name+'.tmp')
                    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                        for rel in sorted(old.hashes)+['run_manifest.json']:z.write(out/rel,rel)
                    os.replace(temp,archive)
                print('Verified completed result; no recomputation.\nUPLOAD_THIS='+str(archive),flush=True);return 0
            (out/'source').mkdir(exist_ok=True);shutil.copyfile(Path(__file__),out/'source'/Path(__file__).name)
            for name,run in (('step59',env['r59']),('step62',env['r62']),('step16',env['r16'])):
                write_json(out/(name+'_manifest.json'),run.manifest)
            write_json(out/'step59_protocol.json',env['p59']);write_json(out/'step62_protocol.json',env['p62'])
            write_json(out/'hgb_encoder.json',env['encoder'])
            for s,ep in env['endpoints'].items():write_csv(out/f'endpoints_{s}.csv',ep)
            write_json(out/'input_audit.json',env['audit'])
            if args.preflight_only:
                dest=package(out,'STEP70_PREFLIGHT_COMPLETE',dict(protocol_identity=identity,training=False,test_scoring=False))
                print('STEP70_PREFLIGHT_COMPLETE\nUPLOAD_THIS='+str(dest),flush=True);return 0
            if verify_stage(out,'calibration',identity):
                print('[2/5] Reuse verified train reference, validation scores and thresholds.',flush=True)
                prior=read_json(out/'calibration_input_audit.json');recheck(prior);env['audit'].update(prior)
                reference=read_json(out/'normal_reference.json');thresholds=np.asarray(read_json(out/'thresholds.json')['thresholds'])
            else:
                print('[2/5] Fit train NORMAL link statistics; validate original HGB and calibrate val ...',flush=True)
                train=load_raw(env,'train',rf_only=True);reference=fit_reference(train);del train
                write_json(out/'normal_reference.json',reference)
                val=load_raw(env,'val');scores,native=score_split(env,val,'val',reference,out);del val
                recheck(env['audit']);thresholds=calibration(env,reference,scores,out,identity);del scores,native
            print('[3/5] Test score evaluation under the frozen validation lock ...',flush=True)
            if not verify_stage(out,'test',identity):
                assert_test_gate(out);test=load_raw(env,'test');score_split(env,test,'test',reference,out);del test
                write_json(out/'test_input_audit.json',env['audit'])
                seal_stage(out,'test',['scores_test.npz','raw_rf_components_test.npz','coverage_test.csv','hgb_replay_test.json','test_input_audit.json'],identity)
            else:
                prior=read_json(out/'test_input_audit.json');recheck(prior);env['audit'].update(prior)
                print('  Reuse verified test scores.',flush=True)
            with np.load(out/'scores_test.npz',allow_pickle=False) as z:
                scores=z['scores'].copy();native=z['native_rf'].copy();y=z['y_true'].copy();episodes=z['episode_id'].copy()
            if not verify_stage(out,'statistics',identity):
                print('[4/5] RF-vs-rest metrics and paired whole-episode bootstrap ...',flush=True)
                table,contrasts=statistics(y,episodes,scores,thresholds,out,args.bootstrap)
                native_rows=[]
                for i,(method,seed) in enumerate(SLOTS):
                    if i>=2:native_rows.append(dict(method=method,seed=seed,operating_rule='original multi-class prediction==RF; no Step70 budget calibration',**binary_metrics(y,native[i].astype(bool))))
                write_csv(out/'native_decision_reference.csv',native_rows)
                write_json(out/'summary.json',dict(status='STEP70_RF_STATISTICS_COMPLETE',dataset_id=DATASET_ID,
                    analysis='RF-positive F1, NOT six-class macro-F1; all common L10 endpoints',model_summary=table,primary_comparisons=contrasts,
                    model_training=False,neural_inference=False,hgb_probability_inference=True,external_validation=False,onset_or_recovery_evaluation=False,
                    cautions=protocol['scope']))
                seal_stage(out,'statistics',['model_summary.csv','metric_intervals.csv','metrics_by_fit.csv','rf_alarm_by_true_class.csv','primary_comparisons.csv','bootstrap_components.npz','bootstrap_distribution.npz','native_decision_reference.csv','summary.json'],identity)
            print('[5/5] Recheck source integrity and package review ...',flush=True)
            recheck(env['audit']);write_json(out/'input_audit.json',env['audit'])
            dest=package(out,'STEP70_RF_STATISTICS_COMPLETE',dict(protocol_identity=identity,elapsed_seconds=time.perf_counter()-started,model_training=False,neural_inference=False,hgb_probability_inference=True))
            print('STEP70_RF_STATISTICS_COMPLETE\nUPLOAD_THIS='+str(dest),flush=True);return 0
    except (Exception,KeyboardInterrupt) as exc:
        diagnostic=out.parent/('diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        diagnostic.mkdir(parents=True,exist_ok=False)
        write_json(diagnostic/'failure.json',dict(status='FAILED',type=type(exc).__name__,message=str(exc),bound=bound,project_root=str(root),output=str(out)))
        (diagnostic/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if env is not None:write_json(diagnostic/'consumed_inputs.json',env['audit'])
        dest=package(diagnostic,'FAILED')
        print('STOPPED: '+str(exc)+'\nCompleted prior results preserved.\nUPLOAD_THIS='+str(dest),file=sys.stderr,flush=True)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1


def self_test():
    import unittest
    class Checks(unittest.TestCase):
        def test_budget(self):self.assertEqual(np.sum(np.arange(100)>fpr_threshold(np.arange(100),.05)),5)
        def test_ties(self):self.assertEqual(fpr_threshold(np.r_[np.zeros(90),np.ones(10)],.05),1)
        def test_expiry(self):
            z=np.zeros((20,2));z[0]=100;c,s=finite_rf_scores(z,np.zeros(20,int),np.array(['a']*20),np.arange(9,20));self.assertGreater(s[0],0);self.assertEqual(s[1],0)
        def test_links(self):
            z=np.zeros((10,2));z[8]=3;z[9]=3;l=np.zeros(10,int);l[9]=1;c,s=finite_rf_scores(z,l,np.array(['a']*10),np.array([9]));self.assertEqual(s[0],2.5)
        def test_scope(self):
            f=pd.DataFrame({'raw_path_type':['GATEWAY_LEO','GNSS_LEO']})
            for l in LINKS:f[RF_COLS[l][0]]=[10,10];f[RF_COLS[l][1]]=[.01,.01]
            k,x=observed_rf(f);self.assertEqual(k.tolist(),[0,-1]);self.assertTrue(np.isnan(x[1]).all())
        def test_no_boundary(self):
            with self.assertRaises(ValueError):finite_rf_scores(np.zeros((11,2)),np.zeros(11,int),np.array(['a']*10+['b']),np.array([10]))
        def test_binary(self):self.assertEqual(binary_metrics(np.arange(6),np.zeros(6,bool))['rf_f1'],0)
        def test_paths(self):
            with self.assertRaises(ValueError):safe_child(Path('.'),'../x')
    return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks)).wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())
