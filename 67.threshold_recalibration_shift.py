#!/usr/bin/env python3
"""Step67: benign-quantile recalibration of the 22-field projection under shift.

Question. Step61 showed that, when benign variability grows (Step49 regimes
P_mild/P_moderate/P_severe), the fixed-threshold 22-field projection loses more
than a same-scope continuous-input learner, mostly through GNSS-directed false
alarms. Is that loss located in the fixed token thresholds (recoverable at
deployment by recalibrating to the platform's benign baseline, without
retraining) or in the frozen detectors themselves?

Design (prespecified).
* Data: the existing Step49 regime generations (P_reference, P_mild,
  P_moderate, P_severe). No regeneration. Each regime shares the same 150
  nominal episode blocks with six class replicas.
* Block split: a fixed seeded permutation of the shared blocks gives
  calibration blocks and evaluation blocks (default 50/50). Calibration blocks
  are further split into target-fit (A) and source-fit (B) halves. The target
  benign distribution is P_reference's NORMAL windows in A; the source benign
  distribution of regime R is R's NORMAL windows in B. P_reference therefore
  receives a map between two independent benign samples of the same
  distribution: a no-harm control for the procedure itself.
* Recalibration: for each thresholded measurement channel of the pinned mapper
  (16 continuous, 18 integer), a monotone mid-mass quantile map source->target
  fitted on benign windows only (integer channels: smallest target value reaching
  the mid-mass probability, zero stays zero; slope-1 extension outside the fitted
  range; physical bounds clipped). The map is
  applied to the actual AND the paired nominal raw record; the pinned
  normalizer (shared_nominal.py, unchanged) then produces the recalibrated
  tokens. Binary flags, counts of commands/sessions and context are untouched.
* Detectors: the frozen Step59 v2 Semantic22 HGB/MLP L10 models (5 seeds), with
  their train-fitted vocabularies. No fit on regime data.
* Evaluation on evaluation blocks only, identical endpoints for fixed and
  recalibrated tokens: Macro-F1, NORMAL false-alarm rate, per-class recall,
  the evidence-availability reference (endpoints without any token change in
  the ten windows assigned NORMAL) and per-token false-trigger rates on NORMAL
  windows. Paired whole-block bootstrap (20000 draws); contrasts recalibrated
  minus fixed, Bonferroni 16 over 4 regimes x 2 learners x 2 metrics.
* Expectation recorded before running: NORMAL false alarms move toward the
  reference level, Macro-F1 recovers partially, the residual loss tracks the
  drop of the evidence-availability reference; P_reference changes ~0.
  Report regardless of direction.

Usage (same .venv as Step59; CPU only):
    python "67.threshold_recalibration_shift.py" --preflight-only
    python "67.threshold_recalibration_shift.py"
Optional: --step49-run PATH --step59-run PATH --calibration-fraction 0.5
Outputs: outputs/shared_nominal_v2/67_threshold_recalibration/run_<utc>/ + review ZIP.
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, platform, re, shutil, sys, time, traceback, warnings, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'shared_nominal_threshold_recalibration_v2'
STEP59_VERSION = 'shared_nominal_raw_semantic_v2'
CONDITIONS = ('P_reference', 'P_mild', 'P_moderate', 'P_severe')
TARGET_CONDITION = 'P_reference'
TOKEN_CONDITIONS = ('fixed', 'recalibrated')
FAMILIES = ('HGB', 'MLP')
SEEDS = tuple(range(20260610, 20260615))
REPRESENTATION = 'Semantic22'
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING', 'UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY', 'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
LINKS = ('ttc', 'isl', 'iot', 'uav')
SNR = {'ttc': 'ttc_snr_db', 'isl': 'isl_snr_db', 'iot': 'iot_snr_db', 'uav': 'uav_sinr_db'}
CONTINUOUS = tuple([SNR[l] for l in LINKS] + ['gnss_cn0_mean_dbhz', 'gnss_tracking_jitter', 'gnss_carrier_lock_ratio',
              'position_jump_m', 'pseudorange_residual_m', 'velocity_jump_mps', 'time_residual_ns', 'clock_drift_ppm']
              + [l + '_frame_loss_rate' for l in LINKS])
INTEGER = tuple(['gnss_lock_loss_count', 'gnss_cycle_slip_count']
                + [l + '_' + c for l in LINKS for c in ('crc_error_count', 'decode_fail_count', 'retry_count', 'timeout_count')])
RECAL_CHANNELS = CONTINUOUS + INTEGER
BOUNDS = {**{l + '_frame_loss_rate': (0., 1.) for l in LINKS}, 'gnss_carrier_lock_ratio': (0., 1.),
          **{k: (0., None) for k in ('gnss_tracking_jitter', 'position_jump_m', 'pseudorange_residual_m', 'velocity_jump_mps',
                                     'time_residual_ns', 'clock_drift_ppm', 'gnss_cn0_mean_dbhz')},
          **{k: (0., None) for k in INTEGER}}
METRICS = ('macro_f1', 'normal_false_alarm_rate') + tuple('recall_' + x for x in LABELS)

# ----------------------------------------------------------------------------- utilities
def require(ok, message):
    if not bool(ok):
        raise ValueError(message)

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def json_safe(v):
    if isinstance(v, dict): return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray): return json_safe(v.tolist())
    if isinstance(v, np.generic): return json_safe(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, Path): return str(v)
    return v

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8'); tmp.replace(path)

def write_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + '.tmp')
    pd.DataFrame(frame).to_csv(tmp, index=False, encoding='utf-8-sig'); tmp.replace(path)

def write_npz(path, **arrays):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)

def read_csv(path, **kw):
    return pd.read_csv(path, keep_default_na=False, encoding='utf-8-sig', **kw)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

class BoundRun:
    """Read files of a completed run only after checking them against its manifest."""
    def __init__(self, root, manifest):
        self.root = Path(root).resolve(); self.manifest = manifest; self.hashes = dict(manifest['output_sha256']); self.observed = {}
    def path(self, rel):
        require(rel in self.hashes, 'File not bound in run manifest: ' + rel)
        p = (self.root / rel).resolve(); require(p.is_relative_to(self.root) and p.is_file(), 'Missing bound file: ' + str(p))
        h = digest(p); require(h == self.hashes[rel], 'Bound file hash differs: ' + rel); self.observed[rel] = h; return p
    def json(self, rel): return read_json(self.path(rel))
    def csv(self, rel, **kw): return read_csv(self.path(rel), **kw)
    def recheck(self):
        for rel, h in self.observed.items(): require(digest(self.root / rel) == h, 'Consumed file changed during run: ' + rel)

# ----------------------------------------------------------------------------- Step59 (frozen detectors)
def locate_run(base, explicit, want):
    if explicit:
        p = Path(explicit).expanduser().resolve(); require(p.is_dir(), 'Run directory missing: ' + str(p)); return p
    matches = [p.parent for p in sorted(Path(base).glob('run_*/run_manifest.json')) if want(p.parent)]
    require(len(matches) == 1, f'Expected exactly one matching completed run under {base}; found {len(matches)}. Pass the path explicitly.')
    return matches[0].resolve()

def open_step59(root, explicit):
    def ok(run):
        try:
            m = read_json(run / 'run_manifest.json'); p = read_json(run / 'protocol.json')
            return m.get('status') == 'RAW_SEMANTIC_BASELINE_COMPLETE' and p.get('version') == STEP59_VERSION
        except (OSError, ValueError, KeyError):
            return False
    run = locate_run(root / 'outputs/shared_nominal_v2/59_raw_semantic', explicit, ok)
    m = read_json(run / 'run_manifest.json'); reader = BoundRun(run, m)
    p = reader.json('protocol.json')
    require(m['status'] == 'RAW_SEMANTIC_BASELINE_COMPLETE' and m.get('test_used_for_selection') is False, 'Step59 run incomplete')
    require(p['version'] == STEP59_VERSION and p['identity'] == m['protocol_identity'], 'Step59 protocol/manifest mismatch (need the v2 run)')
    require(p['seeds'] == list(SEEDS) and REPRESENTATION in p['representations'], 'Step59 seed/representation inventory differs')
    src = reader.path('source/59.raw_semantic_representation_baseline.py'); require(digest(src) == p['script_sha256'], 'Step59 source differs from its protocol')
    base = load_module(src, 'step67_step59'); require(tuple(base.FIELDS) == tuple(p['semantic_fields']) and tuple(base.LABELS) == LABELS, 'Step59 field/label contract differs')
    marker = reader.json('all_training_complete.json'); require(marker['protocol_identity'] == p['identity'] and marker['test_scoring_started'] is False, 'Step59 training barrier missing')
    return reader, p, base

def restore_encoder(base, record):
    require(record['numeric'] == [] and record['categorical'] == list(base.FIELDS) and record['unknown_policy'] == 'one explicit unseen category per field', 'Unexpected Semantic22 encoder record')
    enc = base.WindowEncoder([], list(base.FIELDS)); enc.categories = record['categories']; enc.names = record['window_feature_names']
    require(enc.record() == record, 'Encoder restoration differs'); return enc

def collect_models(reader, protocol, encoder):
    cases = []
    for family in FAMILIES:
        for seed in SEEDS:
            key = f'{REPRESENTATION}_L10_{family}_seed{seed}'; folder = 'jobs/' + key
            done = reader.json(folder + '/complete.json'); report = reader.json(folder + '/training_report.json')
            require(done['identity'] == object_hash(dict(protocol=protocol['identity'], key=key, encoder=encoder.record())), 'Step59 job identity differs: ' + key)
            for name, h in done['files'].items():
                require(reader.hashes.get(folder + '/' + name) == h, 'Manifest/job hash mismatch: ' + key + '/' + name)
            model = reader.root / folder / 'model.joblib'
            require(model.is_file(), 'Missing model.joblib: ' + str(model) + ' (use the ORIGINAL Step59 run on the training PC; review ZIPs omit models)')
            require(report['features'] == len(encoder.names) * 5 and report['family'] == family and report['seed'] == seed, 'Saved model report differs: ' + key)
            cases.append(dict(key=key, family=family, seed=seed, path=str(model), features=report['features']))
    return cases

def load_model(case, protocol):
    import sklearn, joblib
    require(sklearn.__version__ == protocol['runtime']['sklearn'], f"Installed sklearn {sklearn.__version__} differs from Step59's {protocol['runtime']['sklearn']}; run in the Step59 .venv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); model = joblib.load(case['path'])
    require(not caught, 'Model deserialization warnings: ' + str([str(w.message) for w in caught]))
    require(np.array_equal(model.classes_, np.arange(6)) and model.n_features_in_ == case['features'], 'Restored model schema differs: ' + case['key'])
    return model

# ----------------------------------------------------------------------------- Step49 (regime generations)
def open_step49(root, explicit):
    def ok(run):
        try:
            m = read_json(run / 'run_manifest.json')
            return all((run / 'generation' / c / 'complete.json').is_file() for c in CONDITIONS) and 'output_sha256' in m
        except (OSError, ValueError):
            return False
    run = locate_run(root / 'outputs/shared_nominal_v2/49_parameter_stress', explicit, ok)
    m = read_json(run / 'run_manifest.json'); reader = BoundRun(run, m); p = reader.json('protocol.json')
    return reader, p

def rows_per_class(protocol, fallback):
    for a, b in (('generator', 'rows_per_class'), ('generation', 'rows_per_class')):
        try:
            return int(protocol[a][b])
        except (KeyError, TypeError, ValueError):
            pass
    return fallback

def block_ids_for(meta, reader, folder):
    """Shared nominal block of each endpoint episode. Prefer the run's own block file; else parse the episode number."""
    episodes = meta.episode_id.astype(str).to_numpy()
    rel = folder + '/endpoint_blocks.csv'
    if rel in reader.hashes:
        tab = reader.csv(rel)
        ecol = next((c for c in tab.columns if 'episode' in c.lower()), None)
        bcol = next((c for c in tab.columns if 'block' in c.lower() and c != ecol), None)
        if ecol is not None and bcol is not None:
            mapping = dict(zip(tab[ecol].astype(str), pd.to_numeric(tab[bcol], errors='raise').astype(int)))
            if all(e in mapping for e in set(episodes)):
                return np.array([mapping[e] for e in episodes], dtype=np.int64), 'endpoint_blocks.csv:' + bcol
    numbers = [re.search(r'(\d+)\s*$', e) for e in episodes]
    require(all(numbers), 'Cannot derive shared nominal block from episode ids; endpoint_blocks.csv lacks an episode/block mapping')
    return np.array([int(n.group(1)) for n in numbers], dtype=np.int64), 'trailing integer of episode_id'

def load_condition(reader, protocol, condition, base):
    folder = 'generation/' + condition
    frame = reader.csv(folder + '/evaluation_records.csv')
    done = reader.json(folder + '/complete.json')
    columns = list(dict.fromkeys(list(base.META) + list(base.FIELDS) + list(base.RAW_KEYS) + ['baseline_' + k for k in base.RAW_KEYS]
                                 + ['baseline_observable_' + f for f in base.FIELDS] + ['raw_observable_effect_present']))
    n = rows_per_class(protocol, len(frame) // 6); parts = []
    for i, label in enumerate(base.LABELS, 1):
        name = f'raw_class_{i}.csv.gz'; rel = folder + '/' + name
        if isinstance(done.get('artifact_sha256'), dict) and name in done['artifact_sha256']:
            require(done['artifact_sha256'][name] == reader.hashes.get(rel), 'Generation artifact not bound to run manifest: ' + rel)
        raw = read_csv(reader.path(rel), usecols=lambda c: c in columns)
        require(set(columns) <= set(raw.columns), 'Raw regime file lacks columns: ' + str(sorted(set(columns) - set(raw.columns))))
        wanted = frame.iloc[(i - 1) * n:i * n].reset_index(drop=True)
        require(len(raw) == n == len(wanted) and raw.label.eq(label).all() and wanted.label.eq(label).all(), 'Raw/record class inventory differs: ' + condition + ' ' + label)
        for k in base.META + base.FIELDS:
            a = raw[k].astype(str).to_numpy(); b = wanted[k].astype(str).to_numpy()
            if k in ('scenario_id', 'episode_id') and not np.array_equal(a, b):
                a = np.char.add('P_', a.astype(str))
            require(np.array_equal(a, b), 'Raw/record alignment differs: ' + condition + ' ' + k)
        parts.append(raw)
    raw = pd.concat(parts, ignore_index=True)
    seq = np.load(reader.path(folder + '/sequences_L10.npz'), allow_pickle=True)
    require('indices' in seq and 'y' in seq, 'sequences_L10.npz must contain indices and y')
    indices = np.asarray(seq['indices'], dtype=np.int64); y = np.asarray(seq['y'], dtype=np.int64)
    require(indices.ndim == 2 and indices.shape[1] == 10 and len(y) == len(indices) and indices.min() >= 0 and indices.max() < len(frame), 'Invalid L10 sequence index table')
    labels = frame.label.map(dict(zip(base.LABELS, range(6)))).to_numpy()
    require(np.array_equal(labels[indices[:, -1]], y), 'Sequence targets differ from record labels')
    require((frame.episode_id.to_numpy()[indices] == frame.episode_id.to_numpy()[indices[:, [0]]]).all(), 'A sequence crosses an episode boundary')
    meta = pd.DataFrame(dict(sequence_id=np.arange(len(y)), episode_id=frame.episode_id.to_numpy()[indices[:, -1]], label=frame.label.to_numpy()[indices[:, -1]], label_id=y,
                             sequence_end_row=indices[:, -1]))
    ep_rel = folder + '/evaluation_endpoints_L10.csv'; endpoint_table_consistent = None
    if ep_rel in reader.hashes:
        ep = reader.csv(ep_rel); endpoint_table_consistent = len(ep) == len(meta)
        if endpoint_table_consistent and 'episode_id' in ep: endpoint_table_consistent = np.array_equal(ep.episode_id.astype(str).to_numpy(), meta.episode_id.astype(str).to_numpy())
        if endpoint_table_consistent and 'sequence_end_row' in ep: endpoint_table_consistent = np.array_equal(pd.to_numeric(ep.sequence_end_row).to_numpy(), indices[:, -1])
    blocks, source = block_ids_for(meta, reader, folder)
    rows_block = np.full(len(frame), -1, dtype=np.int64); rows_block[indices[:, -1]] = blocks
    ep_block = dict(zip(meta.episode_id.astype(str), blocks))
    frame_block = frame.episode_id.astype(str).map(ep_block)
    orphan = frame_block.isna().to_numpy()            # records of episodes without any L10 endpoint (e.g. a truncated last episode)
    frame_block = frame_block.fillna(-1).to_numpy(dtype=np.int64)
    orphan_episodes = sorted(set(frame.episode_id.astype(str).to_numpy()[orphan]))
    require(orphan.mean() < 0.02, f'Too many records outside endpoint episodes ({orphan.sum()}): block mapping is probably wrong')
    per_block = pd.DataFrame(dict(block=frame_block[~orphan], label=frame.label.to_numpy()[~orphan])).drop_duplicates()
    counts = per_block.groupby('block').label.agg(['count', 'nunique'])
    bad = counts[(counts['count'] != 6) | (counts['nunique'] != 6)]
    require(len(bad) == 0, f'Each shared block must hold exactly one episode per class (block source: {source}); first offending blocks: {bad.head(5).to_dict("index")}')
    return dict(condition=condition, frame=frame, raw=raw, indices=indices, y=y, meta=meta, endpoint_block=blocks, record_block=frame_block,
                blocks=np.sort(counts.index.to_numpy()), block_source=source, records=len(frame), endpoints=len(y),
                orphan_records=int(orphan.sum()), orphan_episodes=orphan_episodes, endpoint_table_consistent=endpoint_table_consistent)

# ----------------------------------------------------------------------------- recalibration maps
class QuantileMap:
    """Monotone quantile map source->target fitted on benign samples.

    A source value v is placed at the mid-mass probability p(v) = (F_s(v-) + F_s(v))/2 of the
    source benign sample and mapped to the target benign quantile at p(v): linear-interpolated
    for continuous channels, the smallest target value whose CDF reaches p(v) for integer
    channels (so a count of zero stays zero and ties never jump to the top of their mass).
    Outside the fitted source range the map continues with slope 1 from the nearest end.
    """
    def __init__(self, source, target, integer, bounds):
        s = np.sort(np.asarray(source, float)); t = np.sort(np.asarray(target, float))
        require(len(s) >= 200 and len(t) >= 200 and np.isfinite(s).all() and np.isfinite(t).all(), 'Insufficient/invalid calibration samples')
        self.s = s; self.t = t; self.integer = integer; self.bounds = bounds
    def target_quantile(self, p):
        p = np.clip(np.asarray(p, float), 0., 1.)
        if self.integer:
            k = np.clip(np.ceil(p * len(self.t) - 1e-9).astype(int) - 1, 0, len(self.t) - 1)
            return self.t[k]
        return np.quantile(self.t, p, method='linear')
    def __call__(self, v):
        v = np.asarray(v, float); n = len(self.s)
        p = (np.searchsorted(self.s, v, 'left') + np.searchsorted(self.s, v, 'right')) / (2. * n)
        out = self.target_quantile(p)
        out = np.where(v < self.s[0], self.t[0] + (v - self.s[0]), out)
        out = np.where(v > self.s[-1], self.t[-1] + (v - self.s[-1]), out)
        lo, hi = self.bounds
        if lo is not None: out = np.maximum(out, lo)
        if hi is not None: out = np.minimum(out, hi)
        if self.integer:
            out = np.rint(out); out = np.where(v == 0, 0., out)
        return out
    def record(self, grid=201):
        p = np.linspace(0., 1., grid); xs = np.quantile(self.s, p, method='linear'); ys = self(xs)
        rec = dict(integer=self.integer, bounds=list(self.bounds), n_source=int(len(self.s)), n_target=int(len(self.t)),
                   source_quantiles=xs.round(10).tolist(), mapped_quantiles=np.asarray(ys).round(10).tolist())
        if self.integer:
            uniq = np.unique(self.s); rec['integer_map'] = {str(int(u)): int(self(np.array([u]))[0]) for u in uniq}
        return rec

def benign_windows(cond, blocks, base):
    mask = np.isin(cond['record_block'], blocks) & (cond['frame'].label.to_numpy() == 'NORMAL')
    return base.sanitize_raw(cond['raw'].loc[mask])

def fit_maps(target_benign, source_benign, grid):
    maps = {}
    for ch in RECAL_CHANNELS:
        maps[ch] = QuantileMap(source_benign[ch].to_numpy(), target_benign[ch].to_numpy(), ch in INTEGER, BOUNDS.get(ch, (None, None)))
    return maps

def apply_maps(sanitized, maps):
    out = sanitized.copy()
    for ch, mp in maps.items():
        out[ch] = mp(out[ch].to_numpy())
    return out

def project(sanitized, base, projection):
    rows = [projection.normalize_raw_to_observable(base.restore_projection_row(r)) for r in sanitized.to_dict('records')]
    return pd.DataFrame(rows)[list(base.FIELDS)].astype(str).reset_index(drop=True)

# ----------------------------------------------------------------------------- metrics
def confusion(y, p):
    return np.bincount(np.asarray(y, int) * 6 + np.asarray(p, int), minlength=36).reshape(6, 6)

def metrics_from_cm(cm):
    cm = np.asarray(cm, float); tp = np.diagonal(cm, axis1=-2, axis2=-1); support = cm.sum(-1); den = support + cm.sum(-2)
    f = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0).mean(-1)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return np.concatenate([f[..., None], (1 - recall[..., 0])[..., None], recall], axis=-1)

def block_confusions(y, preds, codes, nblocks):
    out = np.empty((len(preds), nblocks, 6, 6), dtype=np.int64)
    for i, p in enumerate(preds):
        out[i] = np.bincount(codes * 36 + y * 6 + p.astype(int), minlength=nblocks * 36).reshape(nblocks, 6, 6)
    return out

def bootstrap_weights(nblocks, draws, seed):
    rng = np.random.default_rng(seed); return rng.multinomial(nblocks, np.full(nblocks, 1 / nblocks), size=draws).astype(float)

def bootstrap(counts, weights):
    out = np.empty((len(weights), counts.shape[0], len(METRICS)))
    for s in range(0, len(weights), 256):
        cm = np.einsum('bk,skij->bsij', weights[s:s + 256], counts, optimize=True); out[s:s + 256] = metrics_from_cm(cm)
    return out

def evidence_reference(y, indices, changed):
    """Oracle: endpoints with any token change in the ten windows get their true label; others NORMAL."""
    present = changed[indices].any(axis=1); pred = np.where(present, y, 0)
    return pred, present

# ----------------------------------------------------------------------------- main
def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--step49-run', type=Path); p.add_argument('--step59-run', type=Path)
    p.add_argument('--calibration-fraction', type=float, default=0.5)
    p.add_argument('--split-seed', type=int, default=2026100767)
    p.add_argument('--quantile-grid', type=int, default=201)
    p.add_argument('--bootstrap-draws', type=int, default=20000); p.add_argument('--bootstrap-seed', type=int, default=2026100768)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--preflight-only', action='store_true', help='Verify sources, fit maps, check fixed-token roundtrip; no predictions')
    return p

def main(argv=None):
    args = parser().parse_args(argv); started = time.perf_counter(); out = None
    try:
        require(0.2 <= args.calibration_fraction <= 0.8 and args.quantile_grid >= 51 and args.bootstrap_draws >= 2000, 'Invalid calibration/grid/bootstrap setting')
        root = args.project_root.expanduser().resolve()
        reader59, p59, base = open_step59(root, args.step59_run)
        projection = base.load_projection(root)
        require(set(RECAL_CHANNELS) <= set(base.RAW_KEYS), 'Recalibration channel outside the pinned mapper allowlist')
        encoder = restore_encoder(base, reader59.json('encoder_' + REPRESENTATION + '.json'))
        cases = collect_models(reader59, p59, encoder)
        reader49, p49 = open_step49(root, args.step49_run)
        require(digest(reader49.path('source/shared_nominal.py')) == base.PROJECTION_SHA and digest(reader49.path('source/shared_context_v4.py')) == base.CONTEXT_SHA, 'Step49 mapper differs from Step59 mapper')
        if 'source_contract' in p49 and 'dataset_id' in p49['source_contract']:
            require(p49['source_contract']['dataset_id'] == p59['dataset_id'], 'Step49 and Step59 come from different datasets')
        out = root / 'outputs/shared_nominal_v2/67_threshold_recalibration' / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
        out.mkdir(parents=True, exist_ok=False); (out / 'source').mkdir()
        for p in [Path(__file__), Path(cases[0]['path']).parents[2] / 'source' / '59.raw_semantic_representation_baseline.py', root / 'shared_nominal.py', root / 'shared_context_v4.py']:
            shutil.copyfile(p, out / 'source' / p.name)
        from threadpoolctl import threadpool_limits
        import sklearn, joblib
        print(f'Step67 {VERSION}: frozen Step59 {REPRESENTATION} HGB/MLP x {len(SEEDS)} seeds; four regimes; CPU only.', flush=True)
        with threadpool_limits(limits=args.threads):
            conds = {}
            for c in CONDITIONS:
                print('  Loading and aligning ' + c + ' ...', flush=True); conds[c] = load_condition(reader49, p49, c, base)
            blocks = conds[TARGET_CONDITION]['blocks']
            for c in CONDITIONS:
                require(np.array_equal(conds[c]['blocks'], blocks) and conds[c]['endpoints'] == conds[TARGET_CONDITION]['endpoints'], 'Regimes do not share the block/endpoint population')
                require(np.array_equal(conds[c]['endpoint_block'], conds[TARGET_CONDITION]['endpoint_block']) and np.array_equal(conds[c]['y'], conds[TARGET_CONDITION]['y']), 'Endpoint block/label order differs across regimes')
            rng = np.random.default_rng(args.split_seed); perm = rng.permutation(blocks)
            ncal = int(round(len(blocks) * args.calibration_fraction)); cal = np.sort(perm[:ncal]); eva = np.sort(perm[ncal:])
            cal_target = np.sort(cal[:len(cal) // 2]); cal_source = np.sort(cal[len(cal) // 2:])
            require(len(cal_target) >= 5 and len(cal_source) >= 5 and len(eva) >= 10, 'Too few blocks for the split')
            split = dict(blocks=len(blocks), calibration_blocks=cal.tolist(), target_fit_blocks=cal_target.tolist(), source_fit_blocks=cal_source.tolist(), evaluation_blocks=eva.tolist(), split_seed=args.split_seed)
            write_json(out / 'block_split.json', split)
            write_csv(out / 'cohort_audit.csv', [dict(condition=c, records=conds[c]['records'], endpoints=conds[c]['endpoints'], shared_blocks=len(conds[c]['blocks']), block_source=conds[c]['block_source'],
                                                     records_outside_endpoint_episodes=conds[c]['orphan_records'], episodes_outside=';'.join(conds[c]['orphan_episodes']), endpoint_table_consistent=conds[c]['endpoint_table_consistent']) for c in CONDITIONS])
            # --- fit maps (benign windows only; target = P_reference blocks A, source = regime blocks B)
            target_benign = benign_windows(conds[TARGET_CONDITION], cal_target, base)
            maps = {}; map_records = {}; benign_summary = []
            for c in CONDITIONS:
                source_benign = benign_windows(conds[c], cal_source, base)
                maps[c] = fit_maps(target_benign, source_benign, args.quantile_grid); map_records[c] = {ch: m.record(args.quantile_grid) for ch, m in maps[c].items()}
                for ch in RECAL_CHANNELS:
                    s = source_benign[ch].to_numpy(); t = target_benign[ch].to_numpy(); m = maps[c][ch](s)
                    benign_summary.append(dict(condition=c, channel=ch, source_mean=s.mean(), source_sd=s.std(), source_q99=np.quantile(s, .99), target_mean=t.mean(), target_sd=t.std(), target_q99=np.quantile(t, .99),
                                               mapped_mean=m.mean(), mapped_sd=m.std(), mapped_q99=np.quantile(m, .99), n_source=len(s), n_target=len(t)))
            write_json(out / 'calibration_maps.json', map_records); write_csv(out / 'benign_channel_summary.csv', benign_summary)
            protocol = dict(version=VERSION, script_sha256=digest(__file__), dataset_id=p59['dataset_id'], step59_run=str(reader59.root), step59_protocol_identity=p59['identity'], step59_code_sha256=p59['script_sha256'],
                            step49_run=str(reader49.root), step49_protocol_identity=p49.get('identity'), mapper_sha256=base.PROJECTION_SHA, conditions=list(CONDITIONS), target_condition=TARGET_CONDITION,
                            detectors=[dict(representation=REPRESENTATION, family=f) for f in FAMILIES], seeds=list(SEEDS), training=False, new_fit_on_regime_data='quantile maps on benign windows of calibration blocks only',
                            recalibration=dict(channels=list(RECAL_CHANNELS), integer_channels=list(INTEGER), bounds={k: list(v) for k, v in BOUNDS.items()}, grid=args.quantile_grid,
                                               map='monotone mid-mass quantile map source->target fitted on benign windows (continuous: linear target quantile; integer: smallest target value reaching the mid-mass probability, zero stays zero); slope-1 extension outside the fitted range; physical bounds clipped',
                                               applied_to='actual and paired nominal raw records of every window; pinned normalizer unchanged', untouched='binary monitor flags, command/session counts, context categories'),
                            block_split=split, evaluation='evaluation blocks only; identical endpoints under fixed and recalibrated tokens',
                            bootstrap=dict(draws=args.bootstrap_draws, seed=args.bootstrap_seed, unit='whole shared nominal block (six class replicas)', estimator='mean of per-seed endpoint metrics',
                                           intervals='percentile 95; primary family Bonferroni 16 over 4 regimes x 2 learners x 2 metrics (recalibrated minus fixed)'),
                            expectation_before_running=['NORMAL false-alarm rate moves toward the P_reference fixed level', 'Macro-F1 recovers partially', 'residual loss tracks the drop of the evidence-availability reference',
                                                        'P_reference recalibrated minus fixed is approximately zero (no-harm control)', 'reported regardless of direction'],
                            scope='Within-generator, fixed-encoder/fixed-detector diagnosis of where the shift loss is located; recalibration assumes a benign operating period of the deployment environment; no external validity, physical calibration or retraining claim',
                            runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__, platform=platform.platform(), threads=args.threads))
            protocol['identity'] = object_hash(protocol); write_json(out / 'protocol.json', protocol)
            # --- tokens: fixed (stored; roundtrip-verified) and recalibrated (recomputed)
            token_rates = []; token_frames = {}; changed = {}
            for c in CONDITIONS:
                cond = conds[c]; print('  ' + c + ': fixed-token roundtrip and recalibrated projection ...', flush=True)
                actual = base.sanitize_raw(cond['raw']); nominal = base.sanitize_raw(cond['raw'], 'baseline_')
                fixed_actual = project(actual, base, projection); fixed_nominal = project(nominal, base, projection)
                stored_actual = cond['frame'][list(base.FIELDS)].astype(str).reset_index(drop=True); stored_nominal = cond['raw'][['baseline_observable_' + f for f in base.FIELDS]].astype(str).reset_index(drop=True)
                require((fixed_actual.to_numpy() == stored_actual.to_numpy()).all() and (fixed_nominal.to_numpy() == stored_nominal.to_numpy()[:, :]).all(), 'Pinned normalizer does not reproduce the stored fixed tokens: ' + c)
                recal_actual = project(apply_maps(actual, maps[c]), base, projection); recal_nominal = project(apply_maps(nominal, maps[c]), base, projection)
                token_frames[c] = dict(fixed=fixed_actual, recalibrated=recal_actual)
                changed[c] = dict(fixed=(fixed_actual.to_numpy() != fixed_nominal.to_numpy()).any(axis=1), recalibrated=(recal_actual.to_numpy() != recal_nominal.to_numpy()).any(axis=1))
                normal_eval = np.isin(cond['record_block'], eva) & (cond['frame'].label.to_numpy() == 'NORMAL')
                for tk, fr in token_frames[c].items():
                    for f in base.FIELDS:
                        vals = fr[f].to_numpy()[normal_eval]; top = pd.Series(vals).value_counts()
                        token_rates.append(dict(condition=c, tokens=tk, field=f, normal_eval_windows=int(normal_eval.sum()), modal_token=top.index[0], modal_share=top.iloc[0] / len(vals),
                                                non_modal_share=1 - top.iloc[0] / len(vals), distinct=len(top)))
                del actual, nominal, fixed_nominal, recal_nominal; gc.collect()
            write_csv(out / 'token_rates_normal_eval.csv', token_rates)
            if args.preflight_only:
                write_json(out / 'summary.json', dict(status='PREFLIGHT_COMPLETE', predictions=False))
                archive = package(out, 'PREFLIGHT_COMPLETE', dict(protocol_identity=protocol['identity'], elapsed_seconds=time.perf_counter() - started))
                print('Preflight complete (sources bound, maps fitted, fixed tokens reproduced). Review ZIP: ' + str(archive), flush=True); return 0
            # --- evaluation on evaluation blocks
            eva_mask = np.isin(conds[TARGET_CONDITION]['endpoint_block'], eva); y = conds[TARGET_CONDITION]['y'][eva_mask]
            codes_all = pd.Series(conds[TARGET_CONDITION]['endpoint_block'][eva_mask]).map({b: i for i, b in enumerate(eva)}).to_numpy(dtype=np.int64)
            weights = bootstrap_weights(len(eva), args.bootstrap_draws, args.bootstrap_seed); write_npz(out / 'bootstrap_design.npz', blocks=eva, weights=weights)
            summaries = []; seed_rows = []; reference_rows = []; points = {}; samples = {}; unseen_rows = []
            groups = [(tk, fam) for tk in TOKEN_CONDITIONS for fam in FAMILIES]
            for c in CONDITIONS:
                cond = conds[c]; idx = cond['indices'][eva_mask]; print('  ' + c + ': frozen predictions and paired block bootstrap ...', flush=True)
                preds = []; keys = []
                for tk in TOKEN_CONDITIONS:
                    window = encoder.transform(token_frames[c][tk]); features = base.aggregate_windows(window, idx, 10); require(np.isfinite(features).all(), 'Nonfinite features')
                    for f in base.FIELDS:
                        unseen_rows.append(dict(condition=c, tokens=tk, field=f, unseen_windows=int((~token_frames[c][tk][f].isin(encoder.categories[f])).sum()), windows=len(token_frames[c][tk])))
                    for case in cases:
                        model = load_model(case, p59); p = base.predict_batched(model, features); require(np.isin(p, np.arange(6)).all(), 'Invalid prediction'); preds.append(p); keys.append((tk, case['family'], case['seed'])); del model
                    del window, features; gc.collect()
                    pred_ref, present = evidence_reference(y, idx, changed[c][tk]); cm = confusion(y, pred_ref); mm = metrics_from_cm(cm)
                    reference_rows.append(dict(condition=c, tokens=tk, reference_macro_f1=float(mm[0]), attack_endpoints_without_evidence=float(1 - present[y > 0].mean()),
                                               **{'evidence_free_share_' + l: float(1 - present[y == i].mean()) if (y == i).any() else None for i, l in enumerate(LABELS) if i > 0}))
                preds = np.stack(preds).astype(np.int8); write_npz(out / f'predictions_{c}.npz', predictions=preds, y=y.astype(np.int8), sequence_id=cond['meta'].sequence_id.to_numpy()[eva_mask], block_id=conds[TARGET_CONDITION]['endpoint_block'][eva_mask],
                                                                     tokens=np.array([k[0] for k in keys]), families=np.array([k[1] for k in keys]), seeds=np.array([k[2] for k in keys]))
                counts = block_confusions(y, preds, codes_all, len(eva)); full = metrics_from_cm(counts.sum(axis=1)).reshape(len(groups), len(SEEDS), len(METRICS))
                boot = bootstrap(counts, weights).reshape(args.bootstrap_draws, len(groups), len(SEEDS), len(METRICS)).mean(axis=2)
                for gi, (tk, fam) in enumerate(groups):
                    for mi, metric in enumerate(METRICS):
                        lo, hi = np.quantile(boot[:, gi, mi], [.025, .975])
                        summaries.append(dict(condition=c, tokens=tk, family=fam, metric=metric, mean=float(full[gi, :, mi].mean()), seed_sd_ddof1=float(full[gi, :, mi].std(ddof=1)), ci95_low=float(lo), ci95_high=float(hi),
                                              seeds=len(SEEDS), endpoints=int(len(y)), evaluation_blocks=int(len(eva))))
                    for si, seed in enumerate(SEEDS):
                        seed_rows.append(dict(condition=c, tokens=tk, family=fam, seed=seed, **{m: float(full[gi, si, mi]) for mi, m in enumerate(METRICS)}))
                points[c] = full.mean(axis=1); samples[c] = boot; del preds, counts, boot; gc.collect()
            contrasts = []
            for c in CONDITIONS:
                for fam in FAMILIES:
                    gi_r = groups.index(('recalibrated', fam)); gi_f = groups.index(('fixed', fam))
                    for mi, metric in enumerate(METRICS[:2]):
                        d = samples[c][:, gi_r, mi] - samples[c][:, gi_f, mi]; lo, hi = np.quantile(d, [.025, .975]); blo, bhi = np.quantile(d, [.05 / 32, 1 - .05 / 32])
                        contrasts.append(dict(condition=c, family=fam, metric=metric, fixed=float(points[c][gi_f, mi]), recalibrated=float(points[c][gi_r, mi]), recalibrated_minus_fixed=float(points[c][gi_r, mi] - points[c][gi_f, mi]),
                                              ci95_low=float(lo), ci95_high=float(hi), bonferroni16_low=float(blo), bonferroni16_high=float(bhi), favorable_direction='positive' if mi == 0 else 'negative',
                                              role='no_harm_control' if c == TARGET_CONDITION else 'primary'))
            write_csv(out / 'model_summary.csv', summaries); write_csv(out / 'per_seed_metrics.csv', seed_rows); write_csv(out / 'recalibration_contrasts.csv', contrasts)
            write_csv(out / 'evidence_reference.csv', reference_rows); write_csv(out / 'unknown_category_audit.csv', unseen_rows)
            reader49.recheck(); reader59.recheck(); require(digest(__file__) == protocol['script_sha256'], 'Script changed during run')
            write_json(out / 'input_audit.json', dict(step49_consumed_sha256=reader49.observed, step59_consumed_sha256=reader59.observed, rechecked_at_end=True))
            write_json(out / 'summary.json', dict(status='COMPLETE', protocol_identity=protocol['identity'], contrasts=contrasts, evidence_reference=reference_rows,
                                                   interpretation='Recalibrated-minus-fixed contrasts are conditional on the frozen detectors, the benign calibration blocks and this generator; the P_reference row is the no-harm control. Smaller loss is not superiority; compare with the evidence-availability reference per regime.'))
            archive = package(out, 'COMPLETE', dict(protocol_identity=protocol['identity'], elapsed_seconds=time.perf_counter() - started, training=False, test_used_for_selection=False))
            print('Step67 complete. Review ZIP: ' + str(archive), flush=True); return 0
    except Exception as exc:
        print('STOPPED: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
        if out is not None and out.is_dir():
            write_json(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            try:
                print('Diagnostic ZIP: ' + str(package(out, 'FAILED', dict(elapsed_seconds=time.perf_counter() - started))), file=sys.stderr)
            except Exception as packaging:
                print('Diagnostic packaging failed: ' + str(packaging), file=sys.stderr)
        return 1

def package(out, status, extra):
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'run_manifest.json' and not p.name.endswith('.tmp'))
    m = dict(status=status, version=VERSION, completed_at_utc=utc(), **extra, output_sha256={p.relative_to(out).as_posix(): digest(p) for p in files}); m['identity'] = object_hash(m)
    write_json(out / 'run_manifest.json', m)
    dest = out.with_name(out.name + ('_FAILED_review.zip' if status == 'FAILED' else '_review.zip')); tmp = dest.with_name(dest.name + '.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [out / 'run_manifest.json']:
            z.write(p, p.relative_to(out).as_posix())
    tmp.replace(dest); return dest

if __name__ == '__main__':
    sys.exit(main())
