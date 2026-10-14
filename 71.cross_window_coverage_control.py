#!/usr/bin/env python3
"""Step71: paired cross-window field-coverage control on shared_nominal_v2.

Run in the original project folder / existing .venv:
    python "71.cross_window_coverage_control.py"
    python "71.cross_window_coverage_control.py" --preflight-only
    python "71.cross_window_coverage_control.py" --self-test

ONE experiment; no TST/BERT, new generator, raw audit input, external data, alarm
thresholds, onset/delay/recovery or architecture comparison. New small CPU MLPs
are trained. Existing learned checkpoints and outputs are never modified.

Three views of each original L10 endpoint, with one SHARED model per fit seed:
  CurrentOnly: same masked terminal record repeated (not 140 distinct observations).
  FixedHistory: actual ten records, the same eight fields hidden at every time.
  VariableHistory: actual ten records, independently sampled eight-field masks;
                   its terminal mask and terminal values equal FixedHistory's.
Both HISTORY views retain exactly 140 field slots, NOT equal information/bits.
Artificial masks have their own channel, separate from natural UNK and unseen
categories. Input = mean of field-specific one-hot vectors -> MLP(128,64,6).
No condition name, episode ID, time, labels or audit flags enters the network.

Five CPU FP32 fits, 30 fixed epochs. Every training endpoint contributes all three
views to the SAME optimizer update, with equal weight. Fresh train masks each
epoch, identical across fit seeds. Three fixed paired validation/test mask banks.
The one common checkpoint per seed maximizes mean validation macro-F1 over all
3 conditions x 3 banks; ties lower NLL, then earlier epoch. All five training jobs
must complete before ANY test feature construction, forward pass or score.
Hashes and test METADATA are audited before training; this is not a claim that
no bytes containing test data were accessed. Vocabulary is fit on train only.

Primary: VariableHistory - FixedHistory six-class macro-F1; one prespecified
contrast. Other contrasts, NORMAL error and per-class metrics are exploratory.
20,000 paired class-stratified whole-episode bootstrap draws; all views and all
fixed fit/mask realizations share draws. Mean of per-fit/per-mask metrics, NOT
ensemble predictions. Conditional on this dataset, masks and jointly trained
models. Not a physical sensor-outage simulation, field-specific causal attribution,
proof of information-theoretic necessity, or independent external validation.

Masks are sampled per endpoint, not a single globally consistent outage stream.
Changing masks trades repeat observations of one subset for broader field coverage;
it does not isolate a particular recovered field or keep per-field repetitions fixed.
The shared model may respond to visible missingness patterns, although no explicit
view ID is supplied. This is not each view's separately optimized achievable score.

Requires original split CSVs and completed Step62 protocol/endpoint CSVs only.
No raw companions, embeddings, Step59/70 models, or network access are needed.
Dependencies: numpy, pandas, torch, threadpoolctl in the existing environment.
Epoch-boundary checkpoints include optimizer and best model state. Interrupted
partial epochs replay from the last committed epoch with deterministic RNGs.
Do not change training options to seek better test results. Old runs are retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd

VERSION = 'shared_nominal_cross_window_coverage_v1'
DATASET_ID = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
FIELDS = tuple('observation_stage path_type direction source_role destination_role protocol_context operational_phase message_type service_context command_intent payload_or_parameter_context sequence_freshness_status control_security_state signal_quality_state tracking_stability navigation_consistency timing_consistency decoder_status command_execution_result retry_timeout_state session_state state_transition_consistency'.split())
META = ('scenario_id', 'episode_id', 'episode_type', 'episode_step', 'window_id', 'timestamp', 'label')
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING', 'UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY', 'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
CONDITIONS = ('CurrentOnly', 'FixedHistory', 'VariableHistory')
FIT_SEEDS = tuple(range(20260610, 20260615))
MASK_BANKS = (71101, 71102, 71103)
LENGTH, HIDDEN_FIELDS, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10, 8, 20000, 2026100971
METRICS = ('macro_f1', 'accuracy', 'normal_false_attack_rate') + tuple('recall_' + x for x in LABELS)
DEFAULT62 = Path('outputs/shared_nominal_v2/62_temporal_information_control/run_information_control_v1')
OUTPUT = Path('outputs/shared_nominal_v2/71_cross_window_coverage/run_coverage_v1')
TRAINING = dict(epochs=30, batch_endpoints=1024, hidden=[128, 64], learning_rate=0.001,
                weight_decay=0.0001, clip_norm=5.0, dropout=0.0, label_smoothing=0.0)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def obj_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write_json(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def write_csv(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    pd.DataFrame(value).to_csv(tmp, index=False, encoding='utf-8-sig', float_format='%.17g')
    os.replace(tmp, p)


def write_npz(p, **arrays):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def safe_child(root, relative):
    s = str(relative); q = PurePosixPath(s)
    require(s and ':' not in s and '\\' not in s and not q.is_absolute() and '..' not in q.parts,
            'Unsafe relative path: ' + s)
    root = Path(root).resolve(); p = (root / s).resolve()
    require(p != root and root in p.parents, 'Path escapes source/output: ' + s)
    return p


def resolve(root, p):
    p = Path(p).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def read_csv(p, **kw):
    return pd.read_csv(p, keep_default_na=False, encoding='utf-8-sig', **kw)


def ints(values, name):
    a = np.asarray(pd.to_numeric(values, errors='raise'), dtype=np.float64)
    require(np.isfinite(a).all() and np.equal(a, np.floor(a)).all(), name + ': finite integers required')
    require(np.abs(a).max(initial=0) < 2**53, name + ': integers exceed exact float range')
    return a.astype(np.int64)


def label_ids(frame):
    require('label' in frame and set(frame.label.astype(str)) <= set(LABELS), 'Unknown scenario label')
    return frame.label.map(dict(zip(LABELS, range(6)))).to_numpy(dtype=np.int64)


def endpoint_indices(frame):
    require(len(frame) and set(META) <= set(frame), 'Empty split or missing metadata')
    e = frame.episode_id.astype(str).to_numpy(); st = ints(frame.episode_step, 'episode_step')
    wid = ints(frame.window_id, 'window_id'); y = label_ids(frame)
    require(not frame[['episode_id', 'episode_step', 'window_id']].duplicated().any(), 'Duplicate row keys')
    within = np.r_[False, e[1:] == e[:-1]]
    num = pd.to_numeric(frame.timestamp, errors='coerce')
    if num.notna().all():
        dt = np.r_[np.nan, np.diff(num.to_numpy(dtype=np.float64))]
    else:
        dt = pd.to_datetime(frame.timestamp, errors='raise', utc=True).diff().dt.total_seconds().to_numpy()
    require(np.all(np.isfinite(dt[within])) and np.allclose(dt[within], 1, rtol=0, atol=1e-6),
            'Within-episode time gap/irregularity; no automatic time repair')
    starts = np.r_[0, np.flatnonzero(e[1:] != e[:-1]) + 1]; stops = np.r_[starts[1:], len(e)]
    require(len(set(e[starts])) == len(starts), 'Interleaved/repeated episode blocks')
    ends = []
    for a, b in zip(starts, stops):
        require(np.array_equal(st[a:b], np.arange(b-a)) and np.all(np.diff(wid[a:b]) == 1),
                'Episode steps/window IDs have gaps or do not start at step zero')
        require(np.all(y[a:b] == y[a]), 'Episode changes label')
        if b-a >= LENGTH:
            ends.append(np.arange(a+LENGTH-1, b, dtype=np.int64))
    require(ends, 'No complete L10 endpoints')
    return np.concatenate(ends)


def align_endpoints(reference, target):
    ref = reference.copy(); step = 'episode_step' if 'episode_step' in ref else 'sequence_end_step'
    require({'episode_id', step} <= set(ref), 'Reference endpoint keys missing')
    a = pd.MultiIndex.from_arrays([ref.episode_id.astype(str), ints(ref[step], step)])
    b = pd.MultiIndex.from_arrays([target.episode_id.astype(str), ints(target.episode_step, 'episode_step')])
    require(a.is_unique and b.is_unique and len(a) == len(b), 'Endpoint population differs/has duplicates')
    pos = a.get_indexer(b); require(np.all(pos >= 0), 'Missing reference endpoints')
    ref = ref.iloc[pos].reset_index(drop=True)
    for key in META:
        if key in ref:
            require(np.array_equal(ref[key].astype(str), target[key].astype(str)), 'Endpoint differs: ' + key)
    for key in ('label_id', 'y_true', 'true_label_id'):
        if key in ref:
            require(np.array_equal(ints(ref[key], key), label_ids(target)), 'Reference label order differs')
    if 'sequence_length' in ref:
        require(np.all(ints(ref.sequence_length, 'sequence_length') == LENGTH), 'Reference is not L10')
    if 'sequence_start_step' in ref:
        require(np.array_equal(ints(ref.sequence_start_step, 'sequence_start_step'),
                               ints(target.episode_step, 'episode_step')-9), 'L10 context start differs')
    return ref


def commit_stage(out, marker, identity, names, extra=None):
    record = dict(identity=identity, files={s: digest(safe_child(out, s)) for s in names},
                  committed_at_utc=now())
    if extra:
        record.update(extra)
    write_json(safe_child(out, marker), record)


def verify_stage(out, marker, identity):
    p = safe_child(out, marker)
    if not p.is_file():
        return False
    record = read_json(p)
    require(record.get('identity') == identity, 'Completed stage identity differs: ' + str(p))
    require(isinstance(record.get('files'), dict) and record['files'], 'Missing stage inventory: ' + str(p))
    for relative, h in record['files'].items():
        f = safe_child(out, relative)
        require(f.is_file() and digest(f) == h, 'Completed artifact missing/changed: ' + str(f))
    return True


@contextmanager
def run_lock(path):
    f = Path(path).open('a+b')
    try:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('This run is already active; do not start a second copy') from exc
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('This run is already active') from exc
        yield
    finally:
        f.close()


def bind_inputs(root, step62):
    """Read no model predictions/weights. Verify split hashes and METADATA only."""
    pins = {}; root = Path(root).resolve(); step62 = Path(step62).resolve()
    def pin(p, expected=None):
        p = Path(p).resolve(); require(p.is_file(), 'Required file missing: ' + str(p))
        h = digest(p)
        if expected is not None:
            require(h == expected, 'Input SHA256 mismatch: ' + str(p))
        pins[str(p)] = h
        return p
    manifest_path = pin(step62/'run_manifest.json'); m = read_json(manifest_path)
    require(m.get('status') == 'TEMPORAL_INFORMATION_CONTROL_COMPLETE', 'Step62 is not completed')
    def bound(name):
        require(name in m.get('output_sha256', {}), 'Step62 inventory lacks ' + name)
        return pin(safe_child(step62, name), m['output_sha256'][name])
    ppath = bound('protocol.json'); p = read_json(ppath)
    require(p.get('dataset_id') == DATASET_ID, 'Step62 belongs to another benchmark')
    require(p.get('identity') == obj_hash({k:v for k,v in p.items() if k != 'identity'}) == m.get('protocol_identity'),
            'Step62 protocol digest/manifest binding mismatch')
    contract = p['source_contract']; require(contract.get('dataset_id') == DATASET_ID, 'Source dataset differs')
    db = root/'dataset/shared_nominal_v2'
    expected_build = contract.get('observability_audit', {}).get('step7_build_manifest_sha256')
    require(expected_build, 'Missing Step7 manifest hash in Step62 provenance')
    build_path = pin(db/'build_manifest.json', expected_build); build = read_json(build_path)
    require(build.get('dataset_id') == DATASET_ID, 'Build identity differs')
    for rel in ('splits/label_mapping.json', 'splits/episode_split_manifest.csv'):
        require(rel in contract['input_sha256'], 'Missing split contract hash: ' + rel)
        pin(db/rel, contract['input_sha256'][rel])
    frames = {}; ends = {}; endpoint_frames = {}; paths = {}; references = {}; population = {}
    for split in ('train', 'val', 'test'):
        rel = f'splits/{split}_observable.csv'
        require(rel in contract['input_sha256'], 'Missing split hash: ' + split)
        path = pin(db/rel, contract['input_sha256'][rel]); header = read_csv(path, nrows=0).columns.tolist()
        require(set(META + FIELDS) <= set(header), 'Split lacks required 22 observables/metadata: ' + split)
        frame = read_csv(path, dtype=str, usecols=list(META))
        ix = endpoint_indices(frame); ep = frame.iloc[ix].reset_index(drop=True)
        refpath = bound(f'endpoint_audit_{split}.csv'); ref = align_endpoints(read_csv(refpath, dtype=str), ep)
        require('sequence_end_row' in ref and np.array_equal(ints(ref.sequence_end_row, 'sequence_end_row'), ix),
                'Step62 endpoint row correspondence differs: ' + split)
        expected_n = contract['common_endpoint_report']['splits'][split]['common_endpoint_count']
        require(len(ix) == expected_n, 'Common endpoint count differs: ' + split)
        require(set(label_ids(ep)) == set(range(6)), 'All six endpoint classes required')
        frames[split], ends[split], endpoint_frames[split], paths[split] = frame, ix, ep, path
        references[split] = refpath
        population[split] = dict(rows=len(frame), episodes=int(frame.episode_id.nunique()),
                                 endpoints=len(ix), endpoint_episodes=int(ep.episode_id.nunique()),
                                 class_endpoints=np.bincount(label_ids(ep), minlength=6).tolist())
    for a,b in (('train','val'), ('train','test'), ('val','test')):
        require(not set(frames[a].episode_id) & set(frames[b].episode_id), 'Split episode leakage: ' + a + '/' + b)
    return dict(root=root, step62=step62, source_protocol=p, source_manifest=m, pins=pins,
                paths=paths, frames=frames, ends=ends, endpoints=endpoint_frames, population=population,
                references=references, build_path=build_path)


def load_fields(env, split):
    p = env['paths'][split]
    require(digest(p) == env['pins'][str(p)], 'Split changed before field read: ' + split)
    f = read_csv(p, dtype=str, usecols=list(META+FIELDS))
    for key in META:
        require(np.array_equal(f[key], env['frames'][split][key]), 'Field read row order differs: ' + key)
    for key in FIELDS:
        require(f[key].str.len().gt(0).all() and f[key].eq(f[key].str.strip()).all(),
                'Blank/whitespace value; no imputation: ' + key)
    return f


def fit_vocabulary(train):
    """No numerical averaging of category IDs; IDs index field-specific one-hots."""
    require(set(FIELDS) <= set(train), 'Missing train fields')
    categories = {f: sorted(train[f].astype(str).unique().tolist()) for f in FIELDS}
    sizes = [len(categories[f])+2 for f in FIELDS]
    offsets = np.cumsum([0]+sizes[:-1]).tolist()
    return dict(fields=list(FIELDS), categories=categories, offsets=offsets, sizes=sizes,
                unseen_codes=[a+len(categories[f]) for f,a in zip(FIELDS,offsets)],
                mask_codes=[a+len(categories[f])+1 for f,a in zip(FIELDS,offsets)],
                dimension=sum(sizes), fitted_on='train windows only',
                mask_policy='dedicated artificial-mask channel; natural UNK remains a genuine source category')


def encode(frame, vocabulary):
    require(vocabulary['fields'] == list(FIELDS), 'Vocabulary field order differs')
    require(vocabulary['dimension'] < 65536, 'Unexpectedly large vocabulary')
    out = np.empty((len(frame), len(FIELDS)), dtype=np.uint16)
    for j, f in enumerate(FIELDS):
        lookup = {x:i+vocabulary['offsets'][j] for i,x in enumerate(vocabulary['categories'][f])}
        out[:,j] = frame[f].astype(str).map(lookup).fillna(vocabulary['unseen_codes'][j]).to_numpy(dtype=np.uint16)
    return out


def derive_seed(*parts):
    return int(np.random.SeedSequence([int(x) for x in parts]).generate_state(1, dtype=np.uint64)[0])


def make_masks(n, seed):
    """True means hidden. Uniform eight-of-22; independent of values/labels/IDs."""
    require(isinstance(n, (int, np.integer)) and n > 0, 'Positive mask sample count required')
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    ranks = np.argsort(rng.random((int(n), LENGTH, len(FIELDS))), axis=2, kind='stable')
    masks = np.zeros((int(n), LENGTH, len(FIELDS)), dtype=bool)
    np.put_along_axis(masks, ranks[:,:,:HIDDEN_FIELDS], True, axis=2)
    return masks


def pack_masks(masks):
    masks = np.asarray(masks)
    require(masks.dtype == bool and masks.ndim == 3 and masks.shape[1:] == (10,22), 'Invalid mask tensor')
    require(np.all(masks.sum(2) == 8), 'Masks must hide exactly eight fields at every time')
    return np.sum(masks.astype(np.uint32) * (np.uint32(1) << np.arange(22, dtype=np.uint32)),
                  axis=2, dtype=np.uint32)


def unpack_masks(bits):
    bits = np.asarray(bits)
    require(bits.dtype.kind == 'u' and bits.ndim == 2 and bits.shape[1] == 10 and np.all(bits < 2**22),
            'Invalid packed mask codes')
    masks = ((bits[:,:,None] >> np.arange(22, dtype=np.uint32)) & 1).astype(bool)
    require(np.all(masks.sum(2) == 8), 'Corrupt masks: not eight hidden fields')
    return masks


def feature_counts(codes, ends, masks, vocabulary):
    """Integer histograms summing to 10 per field. Divide by 10 before the MLP."""
    codes, ends, masks = np.asarray(codes), np.asarray(ends), np.asarray(masks)
    require(codes.ndim == 2 and codes.shape[1] == 22 and codes.dtype.kind in 'iu', 'Bad categorical codes')
    require(ends.ndim == 1 and ends.dtype.kind in 'iu' and len(ends) > 0 and ends.min() >= 9 and ends.max() < len(codes), 'Invalid L10 endpoints')
    require(masks.dtype == bool and masks.shape == (len(ends),10,22) and np.all(masks.sum(2)==8), 'Bad view masks/budget')
    n, dim = len(ends), vocabulary['dimension']
    mask_codes = np.asarray(vocabulary['mask_codes'], dtype=np.int64)
    ix = ends[:,None] - np.arange(9,-1,-1)
    hist = codes[ix].astype(np.int64, copy=False)
    low = np.asarray(vocabulary['offsets']); high = low + np.asarray(vocabulary['sizes']) - 1
    require(np.all(hist >= low) and np.all(hist < high), 'Source code outside its field or already artificially masked')
    current = np.where(masks[:,-1], mask_codes[None,:], hist[:,-1])
    out = np.zeros((3,n,dim), dtype=np.uint8)
    out[0, np.arange(n)[:,None], current] = 10
    rows = np.arange(n, dtype=np.int64)[:,None,None] * dim
    for c, mask in ((1,masks[:,-1:, :]), (2,masks)):
        values = np.where(mask, mask_codes[None,None,:], hist)
        counts = np.bincount((rows+values).ravel(), minlength=n*dim).reshape(n,dim)
        require(counts.max(initial=0) <= 10, 'Field/category collision')
        out[c] = counts.astype(np.uint8)
    return out


def feature_views(codes, ends, masks, vocabulary):
    return feature_counts(codes,ends,masks,vocabulary).astype(np.float32) / np.float32(10)


def mask_diagnostics(masks):
    require(masks.ndim == 3 and masks.shape[1:] == (10,22) and np.all(masks.sum(2)==8), 'Invalid diagnostics masks')
    current = masks[:,-1]; ever = (~masks).any(axis=1); recovered = current & (~masks[:,:9]).any(axis=1)
    return dict(endpoints=len(masks), hidden_each_window=8, retained_each_history=140,
                fixed_unique_fields=14, variable_unique_fields_mean=float(ever.sum(1).mean()),
                variable_unique_fields_min=int(ever.sum(1).min()), variable_unique_fields_max=int(ever.sum(1).max()),
                current_hidden_fields_seen_in_past_mean=float(recovered.sum(1).mean()),
                current_view_pair_mismatches=0,
                hidden_counts_by_lag_field=masks.sum(axis=0).tolist(),
                note='Mask opportunities only, NOT availability of attack evidence or valid physical sensor readings')


def cached_evaluation(codes, ends, vocabulary, split, out, identity, memory_gib):
    """Three fixed banks. Compact uint8 histograms stay in RAM; no enormous feature file."""
    n, dim = len(ends), vocabulary['dimension']
    estimate = 3*3*n*dim + 3*n*10*22 + 1024*3*dim*8
    require(estimate < memory_gib*2**30, f'{split} estimated feature cache {estimate/2**30:.2f} GiB exceeds budget')
    stage = f'{split}_views_complete.json'; artifact = f'masks_{split}.npz'
    seed_codes = [derive_seed(71, 101 if split=='val' else 102, x) for x in MASK_BANKS]
    expected = dict(seeds=seed_codes, n=n, codes_sha256=array_hash(codes), ends_sha256=array_hash(ends))
    if verify_stage(out, stage, identity):
        record = read_json(out/stage); require(record['contract'] == expected, 'Mask cache inputs differ')
        with np.load(out/artifact, allow_pickle=False) as z:
            bits = z['variable_masks'].copy()
        require(bits.shape == (3,n,10), 'Wrong mask cache shape')
    else:
        bits = np.stack([pack_masks(make_masks(n,s)) for s in seed_codes])
        write_npz(out/artifact, variable_masks=bits, mask_banks=np.asarray(MASK_BANKS,dtype=np.int64),
                  effective_seeds=np.asarray(seed_codes,dtype=np.uint64), ends=ends)
    cache = np.empty((3,3,n,dim),dtype=np.uint8); reports=[]
    for b, seed in enumerate(seed_codes):
        masks = unpack_masks(bits[b])
        require(np.array_equal(pack_masks(make_masks(n,seed)),bits[b]), 'Mask seed replay differs')
        for start in range(0,n,1024):
            stop=min(n,start+1024)
            cache[b,:,start:stop]=feature_counts(codes,ends[start:stop],masks[start:stop],vocabulary)
        reports.append(dict(bank=MASK_BANKS[b],effective_seed=seed,**mask_diagnostics(masks)))
        print(f'[Views] {split} bank {b+1}/3; current mask paired; exactly 14 fields/window',flush=True)
    write_json(out/f'mask_audit_{split}.json',reports)
    write_npz(out/f'encoded_{split}.npz',codes=codes,ends=ends)
    commit_stage(out,stage,identity,[artifact,f'mask_audit_{split}.json',f'encoded_{split}.npz'],{'contract':expected})
    return cache


def confusion(y, pred):
    y,pred=np.asarray(y),np.asarray(pred)
    require(y.shape==pred.shape and np.isin(y,np.arange(6)).all() and np.isin(pred,np.arange(6)).all(), 'Invalid predicted/target IDs')
    return np.bincount(y.astype(np.int64)*6+pred.astype(np.int64),minlength=36).reshape(6,6)


def metrics_from_cm(cm):
    c=np.asarray(cm,dtype=np.float64);tp=np.diagonal(c,axis1=-2,axis2=-1)
    support=c.sum(-1);pred_support=c.sum(-2);den=support+pred_support
    f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(-1)
    acc=np.divide(tp.sum(-1),c.sum((-2,-1)),out=np.full_like(f1,np.nan),where=c.sum((-2,-1))>0)
    recalls=np.divide(tp,support,out=np.full_like(tp,np.nan),where=support>0)
    return np.concatenate([f1[...,None],acc[...,None],(1-recalls[...,0])[...,None],recalls],axis=-1)


def make_model(dimension, seed):
    import torch
    torch.manual_seed(int(seed))
    return torch.nn.Sequential(torch.nn.Linear(dimension,128),torch.nn.ReLU(),
                               torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,6)).cpu()


def state_hash(state):
    h=hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode());h.update(array_hash(state[k].detach().cpu().numpy()).encode())
    return h.hexdigest()


def copy_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_torch(path, value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load_torch(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)


def evaluate_model(model, cache, y, batch_size=4096, keep_probabilities=False):
    import torch
    require(cache.shape[:2]==(3,3) and cache.shape[2]==len(y),'Evaluation cache dimensions differ')
    model.eval();cms=np.zeros((3,3,6,6),dtype=np.int64);losses=np.zeros((3,3),dtype=np.float64)
    pred=np.empty((3,3,len(y)),dtype=np.int8)
    probs=np.empty((3,3,len(y),6),dtype=np.float32) if keep_probabilities else None
    with torch.no_grad():
        for b in range(3):
            for c in range(3):
                for start in range(0,len(y),batch_size):
                    stop=min(start+batch_size,len(y));x=torch.from_numpy(cache[b,c,start:stop].astype(np.float32)/np.float32(10))
                    logits=model(x);require(bool(torch.isfinite(logits).all()),'Nonfinite model output')
                    p=logits.argmax(1).cpu().numpy();pred[b,c,start:stop]=p
                    cms[b,c]+=confusion(y[start:stop],p)
                    loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y[start:stop]),reduction='sum')
                    losses[b,c]+=float(loss)
                    if probs is not None:probs[b,c,start:stop]=torch.softmax(logits,dim=1).cpu().numpy()
    values=metrics_from_cm(cms);require(np.isfinite(values).all(),'Undefined validation/test class metrics')
    return dict(predictions=pred,probabilities=probs,cms=cms,metrics=values,
                mean_f1=float(values[:,:,0].mean()),mean_nll=float((losses/len(y)).mean()),
                condition_f1=values[:,:,0].mean(0).tolist())


def train_fit(out, identity, seed, train_codes, train_ends, y, vocab, val_cache, yval, settings):
    """Each epoch is an atomic commit. No test argument or test data is accessible here."""
    import torch
    job=out/f'seed_{seed}';job.mkdir(exist_ok=True);jobid=obj_hash(dict(protocol=identity,seed=seed))
    marker='training_complete.json'
    if verify_stage(job,marker,jobid):
        print(f'[Train] seed {seed}: reuse verified completed fit',flush=True)
        return read_json(job/'training_report.json')
    model=make_model(vocab['dimension'],seed)
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    initial_hash=state_hash(model.state_dict());epoch=0;history=[];best=None;best_score=-math.inf;best_nll=math.inf;best_epoch=0
    pointer=job/'checkpoint.json';previous_file=None
    if pointer.is_file():
        r=read_json(pointer);require(r['identity']==jobid,'Checkpoint identity changed; do not silently restart')
        file=safe_child(job,r['file']);require(file.is_file() and digest(file)==r['sha256'],'Checkpoint damaged')
        state=load_torch(file);require(state['identity']==jobid and state['seed']==seed,'Checkpoint payload differs')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
        epoch=state['epoch'];history=state['history'];best=state['best'];best_score=state['best_score']
        best_nll=state['best_nll'];best_epoch=state['best_epoch'];previous_file=file
        require(epoch==len(history) and 0<=epoch<=settings['epochs'],'Invalid committed epoch')
        require(state['initial_hash']==initial_hash,'Initial weight reproduction differs')
        print(f'[Resume] seed {seed}: last committed epoch {epoch}; next epoch {epoch+1}',flush=True)
    n=len(train_ends);batch=settings['batch_endpoints']
    for ep in range(epoch+1,settings['epochs']+1):
        started=time.perf_counter();model.train();mask_seed=derive_seed(71,201,ep);shuffle_seed=derive_seed(71,202,ep)
        masks=make_masks(n,mask_seed);order=np.random.Generator(np.random.PCG64(shuffle_seed)).permutation(n)
        train_loss=0.;seen=0;updates=0;last_heartbeat=started
        for start in range(0,n,batch):
            sel=order[start:start+batch]
            # Interleave three equally weighted conditions for every endpoint.
            x=feature_counts(train_codes,train_ends[sel],masks[sel],vocab).transpose(1,0,2).reshape(-1,vocab['dimension'])
            xt=torch.from_numpy(x.astype(np.float32)/np.float32(10));yt=torch.from_numpy(np.repeat(y[sel],3))
            opt.zero_grad(set_to_none=True);logits=model(xt)
            loss=torch.nn.functional.cross_entropy(logits,yt)
            require(bool(torch.isfinite(loss)),'Nonfinite train loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_norm'],error_if_nonfinite=True)
            opt.step();train_loss+=float(loss.detach())*len(yt);seen+=len(yt);updates+=1
            if time.perf_counter()-last_heartbeat>=30:
                print(f'  seed {seed} epoch {ep}: {min(start+batch,n):,}/{n:,} endpoints x 3 views',flush=True)
                last_heartbeat=time.perf_counter()
        require(seen==3*n and updates==math.ceil(n/batch),'Training condition/sample exposure mismatch')
        val=evaluate_model(model,val_cache,yval)
        score,nll=val['mean_f1'],val['mean_nll']
        if score>best_score or (score==best_score and nll<best_nll):
            best,best_score,best_nll,best_epoch=copy_state(model),score,nll,ep
        row=dict(epoch=ep,train_loss=train_loss/seen,val_macro_f1_mean=score,val_nll_mean=nll,
                 val_f1_current=val['condition_f1'][0],val_f1_fixed=val['condition_f1'][1],val_f1_variable=val['condition_f1'][2],
                 best_epoch=best_epoch,updates=updates,examples_per_condition=n,
                 train_mask_seed=mask_seed,train_mask_sha256=array_hash(pack_masks(masks)),
                 shuffle_seed=shuffle_seed,shuffle_sha256=array_hash(order),seconds=time.perf_counter()-started)
        history.append(row)
        state=dict(identity=jobid,seed=seed,epoch=ep,model=copy_state(model),optimizer=opt.state_dict(),
                   torch_rng=torch.get_rng_state(),best=best,best_score=best_score,best_nll=best_nll,
                   best_epoch=best_epoch,history=history,initial_hash=initial_hash)
        file=job/'checkpoints'/f'epoch_{ep:03d}.pt';save_torch(file,state)
        write_json(pointer,dict(identity=jobid,file=file.relative_to(job).as_posix(),sha256=digest(file),epoch=ep))
        if previous_file is not None and previous_file!=file:previous_file.unlink(missing_ok=True)
        previous_file=file;write_csv(job/'history.csv',history)
        print(f'[Train] seed {seed} epoch {ep}/{settings["epochs"]}: val mean F1={score:.6f}; best={best_epoch}; {row["seconds"]:.1f}s',flush=True)
    require(best is not None,'No checkpoint selected')
    model.load_state_dict(best);replay=evaluate_model(model,val_cache,yval,keep_probabilities=True)
    require(abs(replay['mean_f1']-best_score)<1e-12 and abs(replay['mean_nll']-best_nll)<1e-8,'Selected validation checkpoint replay differs')
    save_torch(job/'model.pt',dict(identity=jobid,seed=seed,dimension=vocab['dimension'],model=best,best_epoch=best_epoch))
    write_npz(job/'val_predictions.npz',y_true=yval.astype(np.int8),y_pred=replay['predictions'],
              probabilities=replay['probabilities'],conditions=np.asarray(CONDITIONS),mask_banks=np.asarray(MASK_BANKS))
    report=dict(identity=jobid,seed=seed,initial_state_sha256=initial_hash,selected_state_sha256=state_hash(best),
                parameter_count=sum(p.numel() for p in model.parameters()),epochs_completed=settings['epochs'],best_epoch=best_epoch,
                selected_val_macro_f1=best_score,selected_val_nll=best_nll,
                examples_per_condition_per_epoch=n,updates_per_epoch=math.ceil(n/batch),
                test_accessed=False,one_shared_model_for_all_views=True,validation_replay='PASS',
                fixed_epoch_budget=True,validation_score_still_improving_at_final_epoch=(best_epoch==settings['epochs']),
                interpretation='Jointly trained diagnostic model; not condition-wise optimal classifiers or convergence guarantee')
    write_json(job/'training_report.json',report);write_csv(job/'history.csv',history)
    commit_stage(job,marker,jobid,['model.pt','val_predictions.npz','training_report.json','history.csv'])
    return report


def test_fit(out, identity, seed, vocabulary, cache, y):
    job=out/f'seed_{seed}';jobid=obj_hash(dict(protocol=identity,seed=seed))
    require(verify_stage(out,'all_training_complete.json',identity),'All fits must be frozen before test scoring')
    require(verify_stage(job,'training_complete.json',jobid),'Training incomplete')
    if verify_stage(job,'test_complete.json',jobid):
        with np.load(job/'test_predictions.npz',allow_pickle=False) as z:
            require(np.array_equal(z['y_true'],y),'Test targets changed')
            return z['y_pred'].copy()
    state=load_torch(job/'model.pt');require(state['identity']==jobid and state['dimension']==vocabulary['dimension'],'Selected model differs')
    model=make_model(vocabulary['dimension'],seed);model.load_state_dict(state['model'])
    before=state_hash(model.state_dict());result=evaluate_model(model,cache,y,keep_probabilities=True)
    require(state_hash(model.state_dict())==before,'Model changed during evaluation')
    write_npz(job/'test_predictions.npz',y_true=y.astype(np.int8),y_pred=result['predictions'],probabilities=result['probabilities'],
              conditions=np.asarray(CONDITIONS),mask_banks=np.asarray(MASK_BANKS))
    commit_stage(job,'test_complete.json',jobid,['test_predictions.npz'],
                 dict(shared_state_sha256=before,test_used_for_selection=False))
    print(f'[Test] seed {seed}: current/fixed/variable F1 = '+ ' / '.join(f'{x:.6f}' for x in result['condition_f1']),flush=True)
    return result['predictions']


def episode_components(y, episodes, predictions):
    y,episodes,predictions=np.asarray(y),np.asarray(episodes).astype(str),np.asarray(predictions)
    require(predictions.ndim==4 and predictions.shape[2:]==(3,len(y)),'Expected [fit,bank,condition,endpoint]')
    require(len(episodes)==len(y) and np.isin(predictions,np.arange(6)).all(),'Invalid predictions')
    ids,ei=np.unique(episodes,return_inverse=True);e=len(ids);labs=np.empty(e,dtype=np.int8)
    for i in range(e):
        unique=np.unique(y[ei==i]);require(len(unique)==1,'Mixed bootstrap episode labels');labs[i]=unique[0]
    require(set(labs)==set(range(6)),'All six bootstrap strata required')
    counts=np.zeros((e,*predictions.shape[:3],6,6),dtype=np.uint32)
    for s in range(predictions.shape[0]):
        for b in range(predictions.shape[1]):
            for c in range(3):
                counts[:,s,b,c]=np.bincount(ei*36+y*6+predictions[s,b,c],minlength=e*36).reshape(e,6,6)
    return dict(counts=counts,episode_labels=labs,episode_ids=ids)


def bootstrap_metrics(comp,draws,seed,progress=False):
    counts=comp['counts'];labs=comp['episode_labels'];n=len(labs)
    require(draws>0 and set(labs)==set(range(6)),'Invalid bootstrap request')
    flat=counts.reshape(n,-1).astype(np.float64);rng=np.random.Generator(np.random.PCG64(seed))
    out=np.empty((draws,3,len(METRICS)),dtype=np.float64)
    for start in range(0,draws,100):
        size=min(100,draws-start);weights=np.zeros((size,n))
        for cls in range(6):
            ix=np.flatnonzero(labs==cls);weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=size)
        cms=(weights@flat).reshape(size,*counts.shape[1:])
        values=metrics_from_cm(cms).mean(axis=(1,2))
        require(np.isfinite(values).all(),'Undefined bootstrap metric; do not discard bad draws')
        out[start:start+size]=values
        if progress and (start==0 or (start+size)%2000==0):
            print(f'[Bootstrap] {start+size:,}/{draws:,} whole-episode draws',flush=True)
    return out


def summarize(out,identity,endpoint,y,predictions,seeds):
    if verify_stage(out,'statistics_complete.json',identity):
        return read_json(out/'summary.json')
    comp=episode_components(y,endpoint.episode_id.to_numpy(),predictions)
    write_npz(out/'episode_components.npz',**comp,conditions=np.asarray(CONDITIONS),fit_seeds=np.asarray(seeds),mask_banks=np.asarray(MASK_BANKS))
    total=comp['counts'].sum(0);values=metrics_from_cm(total);point=values.mean((0,1))
    rows=[];class_rows=[]
    for s,seed in enumerate(seeds):
        for b,bank in enumerate(MASK_BANKS):
            for c,cond in enumerate(CONDITIONS):
                rows.append(dict(seed=seed,mask_bank=bank,condition=cond,**dict(zip(METRICS,values[s,b,c].tolist()))))
                cm=total[s,b,c];tp=np.diag(cm);sup=cm.sum(1);den=sup+cm.sum(0)
                for cls in range(6):
                    class_rows.append(dict(seed=seed,mask_bank=bank,condition=cond,label=LABELS[cls],support=int(sup[cls]),
                                           recall=float(tp[cls]/sup[cls]),f1=float(2*tp[cls]/den[cls]) if den[cls] else 0.0))
    dist=bootstrap_metrics(comp,BOOTSTRAP_DRAWS,BOOTSTRAP_SEED,progress=True)
    write_npz(out/'bootstrap_distribution.npz',metrics=dist,metric_names=np.asarray(METRICS),conditions=np.asarray(CONDITIONS),
              primary_delta=dist[:,2,0]-dist[:,1,0])
    intervals=[];summary_rows=[]
    for c,cond in enumerate(CONDITIONS):
        fit_means=values[:,:,c].mean(1)
        for k,metric in enumerate(METRICS):
            lo,hi=np.quantile(dist[:,c,k],[.025,.975])
            intervals.append(dict(condition=cond,metric=metric,estimate=float(point[c,k]),ci95_low=float(lo),ci95_high=float(hi),
                                  fixed_fit_sd=float(fit_means[:,k].std(ddof=1)) if len(seeds)>1 else None,
                                  interval_scope='conditional fixed fits and mask banks, whole episodes'))
        summary_rows.append(dict(condition=cond,**dict(zip(METRICS,point[c].tolist())),
                                 macro_f1_fit_sd=float(fit_means[:,0].std(ddof=1)) if len(seeds)>1 else None,
                                 endpoint_count=len(y),episode_count=len(comp['episode_labels']),fit_count=len(seeds),mask_banks=3))
    contrasts=[]
    for a,b,name in ((2,1,'VariableHistory_minus_FixedHistory'),(1,0,'FixedHistory_minus_CurrentOnly'),(2,0,'VariableHistory_minus_CurrentOnly')):
        for k,metric in enumerate(METRICS):
            d=dist[:,a,k]-dist[:,b,k];lo,hi=np.quantile(d,[.025,.975])
            contrasts.append(dict(contrast=name,metric=metric,difference=float(point[a,k]-point[b,k]),
                                  ci95_low=float(lo),ci95_high=float(hi),primary=(a==2 and b==1 and k==0),
                                  family_size=1 if a==2 and b==1 and k==0 else None,
                                  role='single_primary' if a==2 and b==1 and k==0 else 'exploratory_pointwise',
                                  positive_fit_differences=int(np.sum((values[:,:,a,k]-values[:,:,b,k]).mean(1)>0)),
                                  fixed_fit_differences=(values[:,:,a,k]-values[:,:,b,k]).mean(1).tolist()))
    primary=[r for r in contrasts if r['primary']][0]
    warnings_list=[]
    for seed in seeds:
        report=read_json(out/f'seed_{seed}'/'training_report.json')
        if report['validation_score_still_improving_at_final_epoch']:
            warnings_list.append(f'Seed {seed}: best validation checkpoint at epoch budget; not a convergence proof.')
    result=dict(status='STEP71_COVERAGE_CONTROL_COMPLETE',dataset_id=DATASET_ID,conditions=list(CONDITIONS),
                seeds=list(seeds),mask_banks=list(MASK_BANKS),primary=primary,results=summary_rows,
                warnings=warnings_list,new_small_models_trained=True,existing_models_modified=False,
                inference_shared_weights=True,external_validation=False,
                interpretation='Coverage-pattern intervention at equal retained-slot budget; not equal information, individual-field attribution, sensor-outage realism or TST/BERT advantage.',
                selection='One common model per fit; checkpoint chosen using equal-weight 3-view/3-bank validation mean only.',
                statistics='Mean metrics across fixed fit x mask combinations, paired class-stratified episode bootstrap; no ensemble or seed/mask resampling.',
                scope='New diagnostic on already examined benchmark; not an independent confirmatory test set. No post-test retuning.')
    write_csv(out/'metrics_by_fit_mask.csv',rows);write_csv(out/'class_metrics.csv',class_rows)
    write_csv(out/'model_summary.csv',summary_rows);write_csv(out/'metric_intervals.csv',intervals)
    write_csv(out/'paired_comparisons.csv',contrasts);write_json(out/'summary.json',result)
    files=['episode_components.npz','bootstrap_distribution.npz','metrics_by_fit_mask.csv','class_metrics.csv',
           'model_summary.csv','metric_intervals.csv','paired_comparisons.csv','summary.json']
    commit_stage(out,'statistics_complete.json',identity,files)
    return result


def archive_run(out,manifest,status):
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in ('.run.lock','run_manifest.json','checkpoint.json')
           and 'checkpoints' not in p.relative_to(out).parts and not p.name.endswith('.tmp')]
    manifest=dict(manifest,status=status,updated_at_utc=now(),output_sha256={p.relative_to(out).as_posix():digest(p) for p in files},
                  review_excludes=['per-epoch optimizer checkpoints and pointers'],selected_small_model_weights_included=True)
    write_json(out/'run_manifest.json',manifest)
    suffix='_failed_review.zip' if status in ('FAILED','INTERRUPTED') else '_review.zip'
    path=out.with_name(out.name+suffix);tmp=path.with_name(path.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'run_manifest.json']:z.write(p,p.relative_to(out).as_posix())
    os.replace(tmp,path);return path


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--step62-run',type=Path,default=DEFAULT62)
    p.add_argument('--run-name',default=OUTPUT.name)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--memory-budget-gib',type=float,default=2.)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.self_test:return self_test()
    root=args.project_root.expanduser().resolve();step62=resolve(root,args.step62_run)
    require(Path(args.run_name).name==args.run_name and args.run_name not in ('','.', '..') and ':' not in args.run_name and '\\' not in args.run_name,
            '--run-name must be one safe directory name')
    require(args.threads>0 and math.isfinite(args.memory_budget_gib) and args.memory_budget_gib>0,'Invalid resource settings')
    out=(root/OUTPUT.parent/args.run_name).resolve()
    for source in (root/'dataset',step62):
        source=source.resolve();require(out!=source and source not in out.parents and out not in source.parents,'Output overlaps input directory')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out/'.run.lock'):
        bound=False;manifest=dict(version=VERSION,started_at_utc=now());stage='input_binding'
        try:
            print('Step71: same-current / equal-slot FixedHistory versus VariableHistory; small common MLP only.',flush=True)
            import torch
            from threadpoolctl import threadpool_limits
            torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            with threadpool_limits(limits=args.threads):
                env=bind_inputs(root,step62)
                train=load_fields(env,'train');val=load_fields(env,'val');vocab=fit_vocabulary(train)
                train_codes=encode(train,vocab);val_codes=encode(val,vocab);del train,val
                estimate=9*len(env['ends']['val'])*vocab['dimension']/2**30
                require(estimate<args.memory_budget_gib,'Validation cache exceeds memory budget before training')
                runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=str(torch.__version__),
                             platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),threads=args.threads,
                             torch_build=torch.__config__.show(),device='cpu',dtype='float32',deterministic_algorithms=True)
                payload=dict(version=VERSION,dataset_id=DATASET_ID,script_sha256=digest(Path(__file__)),source_step62_identity=env['source_protocol']['identity'],
                             input_sha256=env['pins'],fields=list(FIELDS),labels=list(LABELS),conditions=list(CONDITIONS),
                             length=10,masked_fields=8,fit_seeds=list(FIT_SEEDS),mask_banks=list(MASK_BANKS),training=TRAINING,
                             vocabulary_sha256=obj_hash(vocab),population=env['population'],runtime=runtime,
                             mask_protocol='PCG64 uniform eight-of-22 independently per endpoint/lag; Fixed reuses Variable terminal mask. No labels/values/IDs in mask generator.',
                             train_masks='Fresh each epoch; RNG seed derive_seed(71,201,epoch); identical across fit seeds. Equal three-view loss per endpoint.',
                             eval_masks='Three paired banks; derive_seed(71,101 for val or 102 for test,bank). Same across fit seeds.',
                             observation_budget='Both history views retain 14x10 slots; CurrentOnly is one masked record repeated, not independent 140 observations.',
                             representation='Train-only field one-hot plus unseen and artificial-mask channels; arithmetic mean over 10 -> shared MLP128/64/6. No normalization fit on val/test.',
                             checkpoint='One common checkpoint per fit: max equal-weight mean validation macro-F1 over 3 views x 3 banks; tie lower NLL then earliest epoch.',
                             primary='VariableHistory minus FixedHistory six-class macro-F1; one contrast, conditional paired episode CI95',
                             bootstrap=dict(draws=BOOTSTRAP_DRAWS,seed=BOOTSTRAP_SEED,unit='whole episode within each of six classes',fit_and_mask_resampling=False),
                             test_gate='No test fields encoded, masks/features built, or predictions generated before all five fits frozen; metadata and hashes inspected earlier.',
                             limitations=['Same synthetic benchmark already examined; not independent confirmation or external validity.',
                                          'Masks intervene on field slots, not a globally consistent sensor-outage trace.',
                                          'Equal slots are not equal information. Varying masks changes repetition/coverage and missingness pattern together.',
                                          'Shared diagnostic model is not an optimized separate classifier per condition; partial-observation ambiguity remains.',
                                          'Episode/scenario labels retained even when observable evidence is absent; no physical-active labels.',
                                          'No claim about TST/BERT, individual-field necessity, or semantic-vs-raw superiority.'])
                protocol=dict(payload,identity=obj_hash(payload));identity=protocol['identity']
                if (out/'protocol.json').exists():require(read_json(out/'protocol.json')==protocol,'Run protocol/input/runtime changed. Original results preserved; do not resume with altered settings.')
                else:write_json(out/'protocol.json',protocol)
                bound=True;manifest.update(protocol_identity=identity,dataset_id=DATASET_ID)
                write_json(out/'vocabulary.json',vocab)
                source=out/'source';source.mkdir(exist_ok=True);shutil.copy2(Path(__file__),source/Path(__file__).name)
                write_json(out/'step62_protocol.json',env['source_protocol']);write_json(out/'step62_manifest.json',env['source_manifest'])
                for split,ep in env['endpoints'].items():
                    result=ep[list(META)].copy();result.insert(0,'endpoint_index',np.arange(len(ep)));result['source_row_index']=env['ends'][split]
                    write_csv(out/f'endpoints_{split}.csv',result)
                write_json(out/'input_audit.json',dict(population=env['population'],hashes=env['pins'],
                           models_or_audit_flags_loaded=False,fit_fields=list(FIELDS),vocabulary_fit_split='train',
                           cache_val_gib=estimate,all_split_episodes_disjoint=True))
                if args.preflight_only:
                    print('[Preflight] Source hashes, endpoints and train-only vocabulary passed. No model training.',flush=True)
                    if (out/'run_manifest.json').exists() and read_json(out/'run_manifest.json').get('status')=='STEP71_COVERAGE_CONTROL_COMPLETE':return 0
                    dest=archive_run(out,manifest,'PREFLIGHT_COMPLETE');print('UPLOAD_THIS='+str(dest),flush=True);return 0
                if verify_stage(out,'statistics_complete.json',identity):
                    # Do not reseal tampered masks/encoded data as if a completed run were intact.
                    for split in ('val', 'test'):
                        require(verify_stage(out,f'{split}_views_complete.json',identity),
                                'Missing completed view cache: '+split)
                    # Also validate all saved selected models and test predictions before reuse.
                    require(verify_stage(out,'all_training_complete.json',identity),'Missing training gate')
                    for seed in FIT_SEEDS:
                        jid=obj_hash(dict(protocol=identity,seed=seed));job=out/f'seed_{seed}'
                        require(verify_stage(job,'training_complete.json',jid) and verify_stage(job,'test_complete.json',jid),'Incomplete reused fit')
                    result=read_json(out/'summary.json')
                else:
                    stage='validation_views';val_cache=cached_evaluation(val_codes,env['ends']['val'],vocab,'val',out,identity,args.memory_budget_gib)
                    ytrain=label_ids(env['endpoints']['train']);yval=label_ids(env['endpoints']['val'])
                    reports=[];stage='training'
                    for seed in FIT_SEEDS:
                        reports.append(train_fit(out,identity,seed,train_codes,env['ends']['train'],ytrain,vocab,val_cache,yval,TRAINING))
                    write_json(out/'training_reports.json',reports)
                    # Bind every final model/validation result and training commit, not just marker names.
                    names=['training_reports.json']
                    for seed in FIT_SEEDS:
                        rel=f'seed_{seed}/';names+=[rel+n for n in ('training_complete.json','model.pt','val_predictions.npz','training_report.json','history.csv')]
                    if not verify_stage(out,'all_training_complete.json',identity):
                        commit_stage(out,'all_training_complete.json',identity,names,dict(all_fit_seeds=list(FIT_SEEDS),test_scoring_started=False))
                    del val_cache,train_codes,val_codes
                    stage='test_views';require(verify_stage(out,'all_training_complete.json',identity),'Test gate not frozen')
                    test=load_fields(env,'test');test_codes=encode(test,vocab);del test
                    unknown={f:int(np.sum(test_codes[:,j]==vocab['unseen_codes'][j])) for j,f in enumerate(FIELDS)}
                    write_json(out/'test_encoding_audit.json',dict(unseen_counts=unknown,policy='fixed train vocabulary, no test refit'))
                    test_cache=cached_evaluation(test_codes,env['ends']['test'],vocab,'test',out,identity,args.memory_budget_gib)
                    ytest=label_ids(env['endpoints']['test']);stage='test_scoring'
                    predictions=np.stack([test_fit(out,identity,s,vocab,test_cache,ytest) for s in FIT_SEEDS])
                    del test_cache;stage='statistics'
                    result=summarize(out,identity,env['endpoints']['test'],ytest,predictions,FIT_SEEDS)
                stage='final_integrity'
                for path,h in env['pins'].items():require(digest(path)==h,'Input changed during run: '+path)
                require(digest(Path(__file__))==protocol['script_sha256'],'Script changed during run')
                (out/'failure.json').unlink(missing_ok=True)
                dest=archive_run(out,manifest,'STEP71_COVERAGE_CONTROL_COMPLETE')
                print('\nSTEP71_COVERAGE_CONTROL_COMPLETE\nPRIMARY_DELTA='+str(result['primary']['difference'])+'\nUPLOAD_THIS='+str(dest),flush=True)
                return 0
        except (Exception,KeyboardInterrupt) as exc:
            # A pre-binding mismatch must not overwrite a previously valid run manifest.
            diag=out if bound else out.parent/'diagnostics'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
            diag.mkdir(parents=True,exist_ok=True)
            write_json(diag/'failure.json',dict(stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                                               original_inputs_modified=False,last_committed_epoch_retained=True))
            dest=archive_run(diag,manifest,'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED')
            print('\nSTOPPED: '+str(exc)+'\nUPLOAD_THIS='+str(dest)+'\nRerun the SAME command after resolving the reported cause; committed epochs are retained.',file=sys.stderr,flush=True)
            return 1


def self_test():
    import unittest
    class CoreTests(unittest.TestCase):
        def test_masks(self):
            mask=make_masks(40,11);self.assertTrue((mask.sum(2)==8).all())
            np.testing.assert_array_equal(unpack_masks(pack_masks(mask)),mask)
        def test_vocabulary(self):
            f=pd.DataFrame({k:['A','UNK'] for k in FIELDS});v=fit_vocabulary(f);codes=encode(f,v)
            self.assertTrue(np.all(codes[1]!=np.array(v['mask_codes'])))
            f[FIELDS[0]]='NEW';self.assertTrue((encode(f,v)[:,0]==v['unseen_codes'][0]).all())
        def test_pairing(self):
            f=pd.DataFrame({k:['A']*10 for k in FIELDS});v=fit_vocabulary(f);codes=encode(f,v)
            masks=make_masks(1,1);fixed=np.repeat(masks[:,-1:],10,1)
            x=feature_views(codes,np.array([9]),fixed,v)
            np.testing.assert_array_equal(x[0],x[1]);np.testing.assert_array_equal(x[1],x[2])
        def test_perfect_metrics(self):
            x=metrics_from_cm(np.eye(6,dtype=np.int64));self.assertEqual(x[0],1.);self.assertEqual(x[2],0.)
        def test_path(self):
            with self.assertRaises(ValueError):safe_child(Path.cwd(),'../escape')
        def test_primary_not_ensemble(self):
            y=np.arange(6);p=np.broadcast_to(y,(2,3,3,6)).copy();p[1]=0
            comp=episode_components(y,np.array(list('abcdef')),p)
            d=bootstrap_metrics(comp,20,1)
            self.assertTrue(np.all(d[:,2,0]==d[:,1,0]))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    sys.exit(main())
