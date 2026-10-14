#!/usr/bin/env python3
"""Step 73 -- TAES novelty controls on the VERIFIED Step59-v2 benchmark.

Research questions (fixed before this run):
  Q1: For the SAME labeled training episodes, how does Semantic22 compare with
      RawScoped as the label budget is reduced? (HGB, L10, fractions 5/10/25/50/100%)
  Q2: Do joint Semantic22 features outperform three group-specific HGB models
      combined at decision level? (joint / uniform mean / validation stacker)

Protocol rules:
  * Reuse the exact Step59-v2 source audit, single-path raw scope, train-only
    categorical dictionaries, HGB hyperparameters, L10 feature statistics,
    partitions and common L10 test population. NEVER use metadata or the paired
    nominal as model input. The common dictionary is learned WITHOUT labels on
    the original training split; fractions limit LABELED EPISODES, not unlabeled
    dictionary construction. It must be disclosed in the manuscript.
  * Fix fractions, nested episode samples, seeds and comparisons before test.
  * Finish every fitted model and validation-only late-fusion fit BEFORE testing.
  * Paired whole-episode bootstrap, class-stratified, shared draws. CI measures
    test episode uncertainty conditional on trained fits and this generator.
  * NO claimed novelty or performance advantage is presumed; negative results
    are saved identically. This is a post-original-submission revision analysis.

Windows PowerShell (keep Step73 beside the current Step59 v2 and revised Step41):
  python ".\\73.novelty_evidence_validation.py" --mode preflight
  python ".\\73.novelty_evidence_validation.py" --mode run

Optional: --project-root, --step59, --run-name, --threads, --memory-budget-gib,
          --bootstrap. Re-run identical command to resume verified fitted jobs.
The script never modifies Step59 or existing model outputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'taes_novelty_controls_v1.0'
EXPECTED_DATASET_ID = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
EXPECTED_TEST_ENDPOINTS = 38078
EXPECTED_TEST_EPISODES = 752
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
SUBSET_SEEDS = (20260610, 20260611, 20260612)
BOOTSTRAP_SEED = 2026100973
SOURCE_GROUPS = {
    'message': tuple('message_type service_context command_intent payload_or_parameter_context sequence_freshness_status control_security_state'.split()),
    'receiver': tuple('signal_quality_state tracking_stability navigation_consistency timing_consistency'.split()),
    'trace': tuple('decoder_status command_execution_result retry_timeout_state session_state state_transition_consistency'.split()),
}


def check(ok, message):
    if not bool(ok):
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False, default=numpy_scalar) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def numpy_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    p = Path(path)
    tmp = p.with_name(p.name + '.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, p)


def frozen_protocol(out, payload):
    data = {'protocol_id': json_hash(payload), **payload}
    p = out / 'protocol.json'
    if p.exists():
        check(read_json(p) == data,
              'Frozen protocol changed in the same run folder. Use a NEW --run-name.')
    else:
        write_json(p, data)
    return data


@contextmanager
def lock_run(path):
    with Path(path).open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another Step73 process is already running here.') from exc
        else:
            import fcntl
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Another Step73 process is already running here.') from exc
        yield


def import_step59(path):
    path = Path(path).resolve()
    check(path.is_file(), 'Step59 source not found: ' + str(path))
    check(path.suffix == '.py', 'Step59 must be a Python source (.py).')
    spec = importlib.util.spec_from_file_location('taes_step59_v2_for_step73', path)
    check(spec is not None and spec.loader is not None, 'Cannot import Step59 source.')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    check(getattr(mod, 'VERSION', None) == 'shared_nominal_raw_semantic_v2',
          'Step73 requires Step59 v2, including RawScoped, not v1.')
    for attr in ('scope_raw', 'prepare_sources', 'feature_matrix', 'feature_encoder',
                 'fit_model', 'train_one', 'verify_job', 'episode_bootstrap'):
        check(hasattr(mod, attr), 'Step59 v2 missing API: ' + attr)
    check(tuple(mod.FIELDS[7:13]) == SOURCE_GROUPS['message'] and
          tuple(mod.FIELDS[13:17]) == SOURCE_GROUPS['receiver'] and
          tuple(mod.FIELDS[17:]) == SOURCE_GROUPS['trace'],
          'Step59 schema/group order changed. Stop instead of inventing groups.')
    return mod


def locate_step59(root, arg):
    if arg:
        return Path(arg).expanduser().resolve()
    paths = list(root.glob('59.raw_semantic_representation_baseline*.py'))
    valid = []
    for p in paths:
        source = p.read_text(encoding='utf-8-sig')
        if "VERSION='shared_nominal_raw_semantic_v2'" in source or \
           'VERSION = "shared_nominal_raw_semantic_v2"' in source:
            valid.append(p)
    check(len(valid) == 1, 'Expected exactly one Step59 v2 script. Found '
          f'{len(valid)}; use --step59 <exact-path> for the desired v2 source.')
    return valid[0]


def validate_population(env, mod, *, synthetic=False):
    dataset_id = env['build']['dataset_id']
    if not synthetic:
        check(dataset_id == EXPECTED_DATASET_ID,
              f'Different benchmark dataset_id: {dataset_id}. Stop; do not mix datasets.')
    split_episodes = {}
    for split in ('train', 'val', 'test'):
        frame = env['frames'][split]
        check(set(mod.FIELDS).issubset(frame.columns), 'Semantic22 field(s) missing')
        ix = env['indices'][split][10]
        check(ix.ndim == 2 and ix.shape[1] == 10 and len(ix) > 0,
              'Expected nonempty L10 indices: ' + split)
        check((ix.min() >= 0 and ix.max() < len(frame)), 'Out-of-range L10 indices')
        ep = frame['episode_id'].astype(str).to_numpy()
        labels = frame['label'].astype(str).to_numpy()
        check(np.all(ep[ix] == ep[ix[:, -1], None]),
              f'L10 crosses episode boundaries in {split}')
        check(set(labels).issubset(mod.LABELS), f'Unexpected label(s) in {split}')
        split_episodes[split] = set(ep.tolist())
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        check(not split_episodes[a].intersection(split_episodes[b]),
              f'Cross-split episode ID overlap {a}/{b}')
    if not synthetic:
        check(len(env['indices']['test'][10]) == EXPECTED_TEST_ENDPOINTS and
              len(split_episodes['test']) == EXPECTED_TEST_EPISODES,
              'Test population differs from 38,078 endpoints / 752 episodes')
    return {s: dict(windows=len(env['frames'][s]),
                    endpoints_L10=len(env['indices'][s][10]),
                    episodes=len(split_episodes[s])) for s in split_episodes}


def subset_design(env, mod):
    """Nested stratified EPISODE sampling. Index into L10 endpoints, not windows."""
    f = env['frames']['train'].iloc[env['indices']['train'][10][:, -1]].reset_index(drop=True)
    check(f.groupby('episode_id').label.nunique().eq(1).all(), 'Mixed-label train episodes')
    by_class = {c: np.asarray(sorted(f.loc[f.label == c, 'episode_id'].unique()), dtype=str)
                for c in mod.LABELS}
    check(all(len(v) >= 3 for v in by_class.values()), 'Not enough episodes/class')
    ep = f.episode_id.astype(str).to_numpy()
    selected = {}
    rows = []
    for seed in SUBSET_SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = {c: rng.permutation(ids) for c, ids in by_class.items()}
        earlier = set()
        for frac in FRACTIONS:
            picked = set()
            for c in mod.LABELS:
                n = max(2, math.ceil(frac * len(by_class[c])))
                picked.update(map(str, shuffled[c][:n]))
            check(earlier.issubset(picked), 'Subsets are not nested')
            earlier = picked
            indices = np.flatnonzero(np.isin(ep, sorted(picked)))
            check(indices.size > 0 and set(f.label.iloc[indices]) == set(mod.LABELS),
                  'Empty or incomplete six-class episode subset')
            selected[(frac, seed)] = indices
            counts = f.iloc[indices].groupby('label').episode_id.nunique().to_dict()
            rows.append(dict(fraction=frac, subset_seed=seed,
                             selected_episodes=len(picked), train_L10_endpoints=len(indices),
                             episode_counts_by_class=counts,
                             selected_episode_hash=json_hash(sorted(picked)),
                             endpoint_index_hash=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()))
    return selected, rows


def group_columns(encoder, semantic_fields):
    """Select each group's one-hot columns in ALL five temporal summary blocks."""
    names = list(encoder.names)
    fields = {name.split('=', 1)[0] for name in names}
    check(fields == set(semantic_fields), 'Unexpected Semantic22 base feature names')
    n = len(names)
    col = {}
    for group, group_fields in SOURCE_GROUPS.items():
        allowed = set(semantic_fields[:7]) | set(group_fields)
        base = np.asarray([i for i, name in enumerate(names)
                           if name.split('=', 1)[0] in allowed], dtype=np.int64)
        check(base.size > 0 and base.size < n, 'Invalid group field selection: ' + group)
        col[group] = np.concatenate([base + offset*n for offset in range(5)])
    return col


def evidence_overlap(env, mod, root, out, splits=('train', 'val')):
    """Paired-nominal GROUP audit. Test stratification is invoked ONLY AFTER all fits."""
    fields = set(mod.FIELDS)
    source = []
    build = env['build']
    rewrite = bool(env['report']['split_integrity']['source_id_rewrite_applied'])
    required = ['label', 'episode_id', 'episode_step', 'window_id',
                'raw_observable_difference_fields']
    for entry in build['sources']:
        path = root/'dataset'/entry['raw_audit']['file']
        check(path.is_file(), 'Paired nominal raw audit missing: ' + str(path))
        chunks = []
        for chunk in pd.read_csv(path, usecols=required, dtype=str,
                                 keep_default_na=False, encoding='utf-8-sig', chunksize=10000):
            check((chunk[required].apply(lambda s: s.str.len() > 0)).all().all(),
                  'Blank provenance in raw audit: ' + str(path))
            parts = chunk.raw_observable_difference_fields.str.split('|')
            # Explicitly check field names, even on audit-only paths.
            unknown = set().union(*(set(x) for x in parts)) - fields - {'NONE'}
            check(not unknown, 'Unknown changed field(s): ' + str(sorted(unknown)))
            row = chunk[['label', 'episode_id', 'episode_step', 'window_id']].copy()
            if rewrite:
                row['episode_id'] = row.label + '::' + row.episode_id
            for name, fs in SOURCE_GROUPS.items():
                row['ev_'+name] = parts.map(lambda x: bool(set(x) & set(fs))).to_numpy(dtype=bool)
            row['ev_context'] = parts.map(lambda x: bool(set(x) & set(mod.FIELDS[:7]))).to_numpy(dtype=bool)
            chunks.append(row)
        source.append(pd.concat(chunks, ignore_index=True))
    src = pd.concat(source, ignore_index=True)
    keys = ['label', 'episode_id', 'episode_step', 'window_id']
    ix = pd.MultiIndex.from_frame(src[keys])
    check(ix.is_unique, 'Raw audit source keys are not unique')
    rows = []
    endpoint_group_counts = {}
    for split in splits:
        frame = env['frames'][split]
        loc = ix.get_indexer(pd.MultiIndex.from_frame(frame[keys].astype(str)))
        check((loc >= 0).all(), f'Cannot align {split} source audit to split')
        ordered = src.iloc[loc].reset_index(drop=True)
        flags = ordered[['ev_message', 'ev_receiver', 'ev_trace', 'ev_context']].to_numpy(dtype=bool)
        any_difference = flags.any(axis=1)
        check(np.array_equal(any_difference, ~env['masks'][split]['semantic_no_change']),
              f'Semantic group evidence audit does not reconstruct {split} Step59 mask')
        l10 = env['indices'][split][10]
        present = np.column_stack([flags[l10, i].any(axis=1) for i in range(3)])
        endpoint_group_counts[split] = present.sum(axis=1).astype(np.int8)
        context_only = flags[l10, 3].any(axis=1) & ~present.any(axis=1)
        truly_no_group_or_context = ~flags[l10].any(axis=(1, 2))
        y = mod.targets(env, split, 10)
        for cls_idx, label in enumerate(mod.LABELS):
            mask = y == cls_idx
            for n_groups in range(4):
                count = int(np.count_nonzero(mask & (present.sum(axis=1) == n_groups)))
                rows.append(dict(split=split, label=label, present_source_groups=n_groups,
                                 endpoints=count, class_endpoints=int(mask.sum()),
                                 fraction=float(count/mask.sum()) if mask.sum() else None,
                                 context_only_in_zero_group=int(np.sum(mask & context_only)) if n_groups==0 else None,
                                 no_semantic_difference_in_zero_group=int(np.sum(mask & truly_no_group_or_context)) if n_groups==0 else None))
    file_name = 'source_overlap_train_val.csv' if set(splits)=={'train','val'} else 'source_overlap_test.csv'
    write_csv(out/file_name, rows)
    return rows, endpoint_group_counts


def ensure_matrix(env, mod, representation, split, encoder, out, memory_budget_gib):
    folder = out/'feature_cache'
    folder.mkdir(exist_ok=True)
    path = folder/f'{representation}_{split}_L10.npy'
    record = {'representation': representation, 'split': split,
              'encoder_hash': json_hash(encoder.record()),
              'dataset_id': env['build']['dataset_id'],
              'shape': [len(env['indices'][split][10]), len(encoder.names)*5],
              'dtype': 'float64', 'aggregation': 'mean_std0_min_max_last'}
    meta = path.with_suffix('.json')
    if path.exists() or meta.exists():
        check(path.exists() and meta.exists() and read_json(meta)['record'] == record,
              f'Incomplete/stale feature cache: {path}. Remove this cache only, NOT jobs.')
        check(digest(path) == read_json(meta)['file_sha256'],
              f'Feature cache changed or truncated: {path}')
    else:
        print(f'  [Feature cache] {representation} {split} ...', flush=True)
        x = mod.feature_matrix(env, representation, split, 10, encoder, memory_budget_gib)
        check(x.shape == tuple(record['shape']) and np.isfinite(x).all(),
              'Feature matrix size/finite check failed')
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('wb') as f:
            np.save(f, x, allow_pickle=False)
        os.replace(tmp, path)
        write_json(meta, {'record': record, 'file_sha256': digest(path)})
        del x
        gc.collect()
    return np.load(path, mmap_mode='r', allow_pickle=False)


def preflight_resources(env, mod, out, budget_gib):
    rows = []
    bytes_total = 0
    missing_cache_bytes = 0
    peak = 0
    for rep in ('RawScoped', 'Semantic22'):
        enc = mod.feature_encoder(env, rep)
        features = len(enc.names)*5
        for split in ('train', 'val', 'test'):
            n = len(env['indices'][split][10])
            b = n * features * 8
            bytes_total += b
            if not (out/'feature_cache'/f'{rep}_{split}_L10.npy').exists():
                missing_cache_bytes += b
            peak = max(peak, b*4)
            rows.append(dict(representation=rep, split=split, endpoints=n,
                             features=features, matrix_gib=round(b/2**30, 3)))
    require_gib = peak / 2**30
    check(require_gib <= budget_gib,
          f'Estimated four-matrix working set {require_gib:.2f} GiB > '
          f'--memory-budget-gib={budget_gib}. Stop before training.')
    free = shutil.disk_usage(out).free
    required_free = missing_cache_bytes * 1.25 + 2*2**30
    check(free > required_free,
          f'Insufficient free disk for remaining resumable feature matrices. '
          f'Need > {required_free/2**30:.1f} GiB free.')
    write_json(out/'resources.json', dict(feature_matrices=rows, estimated_all_feature_cache_gib=bytes_total/2**30,
                                          missing_cache_gib=missing_cache_bytes/2**30,
                                          largest_four_matrix_working_gib=require_gib,
                                          memory_budget_gib=budget_gib, free_disk_gib=free/2**30,
                                          note='HGB internals and pandas dataframes need further RAM; this is a lower bound, not a hard peak limit'))
    return rows


def job_key(rep, frac, seed):
    return f'label_{rep}_f{int(round(100*frac)):03d}_s{seed}'


def train_hgb_job(mod, folder, identity, x, y, xv, yv, seed, settings):
    # Reuse Step59's atomic job format and its validation replay check.
    check(set(np.unique(y)) == set(range(6)), 'Training lacks a class')
    return mod.train_one(folder, identity, 'HGB', x, y, xv, yv, seed, settings)


def read_hgb_job(mod, folder, identity):
    import joblib
    check(mod.verify_job(folder, identity), 'Missing or altered fitted job: ' + str(folder))
    return joblib.load(folder/'model.joblib')


def save_stacker(folder, identity, model, val_y, val_preds, mod):
    import joblib
    folder.mkdir(parents=True, exist_ok=True)
    done = folder/'complete.json'
    if done.exists():
        d = read_json(done)
        check(d['identity'] == identity and digest(folder/'model.joblib') == d['sha256'],
              'Stacker is inconsistent with frozen protocol')
        return
    tmp = folder/'model.joblib.tmp'
    joblib.dump(model, tmp)
    os.replace(tmp, folder/'model.joblib')
    write_json(folder/'validation_report.json',
               dict(validation_f1=float(mod.macro_f1(mod.confusion(val_y, val_preds))),
                    fit_data='Validation predictions ONLY; no test scores or test labels'))
    write_json(done, {'identity': identity, 'sha256': digest(folder/'model.joblib')})


def stack_features(probabilities):
    """Conservative validation-only meta model on log probabilities, fixed transform."""
    return np.concatenate([np.log(np.clip(p, 1e-6, 1.0)) for p in probabilities], axis=1)


def train_everything(env, mod, out, protocol, subsets, budget_gib):
    import joblib
    ytrain = mod.targets(env, 'train', 10)
    yval = mod.targets(env, 'val', 10)
    settings = mod.DEFAULT_TRAINING
    report = []
    for rep in ('RawScoped', 'Semantic22'):
        enc = mod.feature_encoder(env, rep)
        write_json(out/f'encoder_{rep}.json', enc.record())
        x = ensure_matrix(env, mod, rep, 'train', enc, out, budget_gib)
        xv = ensure_matrix(env, mod, rep, 'val', enc, out, budget_gib)
        for frac in FRACTIONS:
            for seed in SUBSET_SEEDS:
                key = job_key(rep, frac, seed)
                identity = json_hash(dict(protocol=protocol['protocol_id'], key=key,
                                          encoder=enc.record()))
                folder = out/'jobs'/key
                print(f'[Label budget] {key}', flush=True)
                tr_idx = subsets[(frac, seed)]
                # Note: unrestricted UNLABELED train vocabulary, subset-LABEL training only.
                if mod.verify_job(folder, identity):
                    print('  verified fit reused', flush=True)
                    fitted_report = read_json(folder/'training_report.json')
                else:
                    xx = x[tr_idx]  # Memory copy only for selected L10 endpoints.
                    fitted_report = train_hgb_job(mod, folder, identity, xx, ytrain[tr_idx],
                                                  xv, yval, seed, settings)
                    del xx
                    gc.collect()
                report.append(dict(job=key, representation=rep, fraction=frac,
                                   seed=seed, train_endpoints=len(tr_idx),
                                   validation_macro_f1=fitted_report['validation']['macro_f1']))
        del x, xv
        gc.collect()
    # The semantic joint model is a full-budget HGB, each component same budget.
    rep = 'Semantic22'
    enc = mod.feature_encoder(env, rep)
    columns = group_columns(enc, mod.FIELDS)
    x = ensure_matrix(env, mod, rep, 'train', enc, out, budget_gib)
    xv = ensure_matrix(env, mod, rep, 'val', enc, out, budget_gib)
    from sklearn.linear_model import LogisticRegression
    for seed in mod.SEEDS:
        val_prob = []
        for group in ('message', 'receiver', 'trace'):
            key = f'fusion_{group}_s{seed}'
            identity = json_hash(dict(protocol=protocol['protocol_id'], key=key,
                                      encoder=enc.record(), columns=columns[group].tolist()))
            folder = out/'jobs'/key
            print(f'[Group fusion] {key}', flush=True)
            if not mod.verify_job(folder, identity):
                xc = x[:, columns[group]]
                xvc = xv[:, columns[group]]
                train_hgb_job(mod, folder, identity, xc, ytrain, xvc, yval, seed, settings)
                del xc, xvc
                gc.collect()
            model = read_hgb_job(mod, folder, identity)
            val_prob.append(model.predict_proba(xv[:, columns[group]]))
            del model
            gc.collect()
        stack_key = f'fusion_stacker_s{seed}'
        identity = json_hash(dict(protocol=protocol['protocol_id'], key=stack_key,
                                  meta='validation_only_logprobs_logistic_C1_lbfgs',
                                  group_order=['message', 'receiver', 'trace']))
        stack_folder = out/'jobs'/stack_key
        if (stack_folder/'complete.json').exists():
            d = read_json(stack_folder/'complete.json')
            check(d['identity'] == identity and digest(stack_folder/'model.joblib') == d['sha256'],
                  'Validation stacker differs from frozen protocol')
        else:
            meta_x = stack_features(val_prob)
            stacker = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                         random_state=seed)
            stacker.fit(meta_x, yval)
            check(np.array_equal(stacker.classes_, np.arange(6)), 'Stacker class mismatch')
            save_stacker(stack_folder, identity, stacker, yval,
                         stacker.predict(meta_x), mod)
            del stacker, meta_x
        report.append(dict(job=stack_key, seed=seed, role='validation_stacker',
                           source='three group-specific val predictions'))
        del val_prob
        gc.collect()
    del x, xv
    gc.collect()
    write_json(out/'training_complete.json', dict(protocol_id=protocol['protocol_id'],
                                                    fitted_jobs=len(report), completed_at=stamp(),
                                                    report=report, test_scoring_started=False))
    return report


def predict_all(env, mod, out, protocol, budget_gib):
    """Test labels are used ONLY here; all jobs MUST be complete and verified first."""
    import joblib
    trained = read_json(out/'training_complete.json')
    check(trained['protocol_id'] == protocol['protocol_id'], 'No matching training complete marker')
    y = mod.targets(env, 'test', 10)
    predictions = {}
    for rep in ('RawScoped', 'Semantic22'):
        enc = mod.feature_encoder(env, rep)
        check(read_json(out/f'encoder_{rep}.json') == enc.record(), 'Vocabulary drift')
        x = ensure_matrix(env, mod, rep, 'test', enc, out, budget_gib)
        for frac in FRACTIONS:
            per_seed = []
            for seed in SUBSET_SEEDS:
                key = job_key(rep, frac, seed)
                identity = json_hash(dict(protocol=protocol['protocol_id'], key=key,
                                          encoder=enc.record()))
                model = read_hgb_job(mod, out/'jobs'/key, identity)
                per_seed.append(mod.predict_batched(model, x))
                del model
            predictions[f'{rep}_f{int(round(frac*100)):03d}'] = np.stack(per_seed)
        if rep == 'Semantic22':
            columns = group_columns(enc, mod.FIELDS)
            joint = []
            late_mean = []
            late_stack = []
            for seed in mod.SEEDS:
                key = job_key('Semantic22', 1.0, seed)
                if seed in SUBSET_SEEDS:
                    identity = json_hash(dict(protocol=protocol['protocol_id'], key=key,
                                              encoder=enc.record()))
                    model = read_hgb_job(mod, out/'jobs'/key, identity)
                else:
                    # Additional joint full-data seeds to match the fusion's five seeds.
                    key = f'fusion_joint_s{seed}'
                    identity = json_hash(dict(protocol=protocol['protocol_id'], key=key,
                                              encoder=enc.record()))
                    model = read_hgb_job(mod, out/'jobs'/key, identity)
                joint.append(mod.predict_batched(model, x))
                del model
                prob = []
                for group in ('message', 'receiver', 'trace'):
                    subkey = f'fusion_{group}_s{seed}'
                    subidentity = json_hash(dict(protocol=protocol['protocol_id'],
                                                key=subkey, encoder=enc.record(),
                                                columns=columns[group].tolist()))
                    model = read_hgb_job(mod, out/'jobs'/subkey, subidentity)
                    prob.append(model.predict_proba(x[:, columns[group]]))
                    del model
                late_mean.append(np.mean(prob, axis=0).argmax(axis=1).astype(np.int8))
                stackkey = f'fusion_stacker_s{seed}'
                stackidentity = json_hash(dict(protocol=protocol['protocol_id'],
                                               key=stackkey, meta='validation_only_logprobs_logistic_C1_lbfgs',
                                               group_order=['message', 'receiver', 'trace']))
                d = read_json(out/'jobs'/stackkey/'complete.json')
                check(d['identity'] == stackidentity and digest(out/'jobs'/stackkey/'model.joblib') == d['sha256'],
                      'Validation stacker failed verification')
                stacker = joblib.load(out/'jobs'/stackkey/'model.joblib')
                late_stack.append(stacker.predict(stack_features(prob)).astype(np.int8))
                del stacker, prob
                gc.collect()
            predictions.update(joint=np.stack(joint), late_mean=np.stack(late_mean),
                               late_stack=np.stack(late_stack))
        del x
        gc.collect()
    check(set(predictions) == {f'{rep}_f{int(frac*100):03d}' for rep in
                                     ('Semantic22','RawScoped') for frac in FRACTIONS} |
          {'joint','late_mean','late_stack'}, 'Incomplete output prediction families')
    for name, arr in predictions.items():
        check(arr.shape[1] == len(y) and np.isin(arr, range(6)).all(),
              f'Invalid test predictions: {name}')
    tmp = out/'predictions_L10.npz.tmp'
    with tmp.open('wb') as f:
        np.savez_compressed(f, y_true=y.astype(np.int8),
                            **{k:v.astype(np.int8) for k,v in predictions.items()})
    os.replace(tmp, out/'predictions_L10.npz')
    return predictions


def train_missing_joint_seeds(env, mod, out, protocol, budget_gib):
    """Two extra full-data models to equalize five joint/fusion seeds BEFORE test."""
    enc = mod.feature_encoder(env, 'Semantic22')
    x = ensure_matrix(env, mod, 'Semantic22', 'train', enc, out, budget_gib)
    xv = ensure_matrix(env, mod, 'Semantic22', 'val', enc, out, budget_gib)
    yt = mod.targets(env, 'train', 10)
    yv = mod.targets(env, 'val', 10)
    for seed in mod.SEEDS:
        if seed in SUBSET_SEEDS:
            continue
        key = f'fusion_joint_s{seed}'
        identity = json_hash(dict(protocol=protocol['protocol_id'], key=key,
                                  encoder=enc.record()))
        folder = out/'jobs'/key
        if not mod.verify_job(folder, identity):
            print('[Joint control] ' + key, flush=True)
            train_hgb_job(mod, folder, identity, x, yt, xv, yv, seed, mod.DEFAULT_TRAINING)
    del x, xv
    gc.collect()


def analyze(env, mod, out, protocol, predictions, draws, test_evidence_groups=None):
    y = mod.targets(env, 'test', 10)
    ix = env['indices']['test'][10][:, -1]
    ep = env['frames']['test'].iloc[ix].episode_id.to_numpy()
    check(len(set(ep)) == (EXPECTED_TEST_EPISODES if env['build']['dataset_id'] == EXPECTED_DATASET_ID else len(set(ep))),
          'Bootstrap episode population mismatch')
    # One shared test-episode bootstrap across ALL fractions and fusion arms.
    stats = mod.episode_bootstrap(y, ep, predictions, [], draws, BOOTSTRAP_SEED)
    names = stats['names']
    b = stats['metrics']
    lookup = {name: b[:, i] for i, name in enumerate(names)}
    detail = []
    for name, arr in predictions.items():
        by_fit = [float(mod.macro_f1(mod.confusion(y, p))) for p in arr]
        detail.append(dict(method=name, fits=len(arr), f1_mean=float(np.mean(by_fit)),
                           fit_sd=float(np.std(by_fit, ddof=1)) if len(by_fit)>1 else None,
                           episode_ci95_low=float(np.quantile(lookup[name], .025)),
                           episode_ci95_high=float(np.quantile(lookup[name], .975)),
                           normal_false_alarm_mean=float(np.mean([
                               mod.scores(y,p)['normal_false_alarm_rate'] for p in arr]))))
    write_csv(out/'model_results.csv', detail)
    contrasts = []
    for frac in FRACTIONS:
        n = int(round(100*frac))
        a = f'Semantic22_f{n:03d}'
        c = f'RawScoped_f{n:03d}'
        dist = lookup[a] - lookup[c]
        contrasts.append(dict(family='label_budget_five_fractions', fraction=frac,
                              contrast=f'{a}_minus_{c}',
                              delta=float(np.mean([mod.macro_f1(mod.confusion(y,p)) for p in predictions[a]]) -
                                                  np.mean([mod.macro_f1(mod.confusion(y,p)) for p in predictions[c]])),
                              ci95_low=float(np.quantile(dist, .025)),
                              ci95_high=float(np.quantile(dist, .975)),
                              bonferroni_ci99_low=float(np.quantile(dist, .005)),
                              bonferroni_ci99_high=float(np.quantile(dist, .995))))
    xx = np.log(np.asarray(FRACTIONS))
    low_selector = np.arange(3)
    diffs = np.column_stack([lookup[f'Semantic22_f{int(round(100*f)):03d}'] -
                             lookup[f'RawScoped_f{int(round(100*f)):03d}'] for f in FRACTIONS])
    low_aulc_dist = np.trapezoid(diffs[:, low_selector], xx[low_selector], axis=1) / (xx[2]-xx[0])
    all_aulc_dist = np.trapezoid(diffs, xx, axis=1) / (xx[-1]-xx[0])
    point_diffs = np.asarray([r['delta'] for r in contrasts])
    def aulc_point(values, positions):
        return float(np.trapezoid(values[positions], xx[positions]) /
                     (xx[positions][-1]-xx[positions][0]))
    summary = {
        'primary_prespecified_low_label_fraction_range': [0.05,0.10,0.25],
        'primary_contrast': 'Semantic22 minus RawScoped HGB L10; normalized AULC against log(labeled episode fraction)',
        'aulc_low_5_to_25': dict(delta=aulc_point(point_diffs, low_selector),
                                ci95_low=float(np.quantile(low_aulc_dist,.025)),
                                ci95_high=float(np.quantile(low_aulc_dist,.975))),
        'aulc_all_5_to_100_secondary': dict(delta=aulc_point(point_diffs, np.arange(5)),
                                           ci95_low=float(np.quantile(all_aulc_dist,.025)),
                                           ci95_high=float(np.quantile(all_aulc_dist,.975))),
    }
    # Fusion: a conservative strongest validation-trained late-fusion comparator.
    for comparator in ('late_mean', 'late_stack'):
        delta_dist = lookup['joint'] - lookup[comparator]
        point = next(z['f1_mean'] for z in detail if z['method']=='joint') - next(z['f1_mean'] for z in detail if z['method']==comparator)
        contrasts.append(dict(family='fusion_two_comparators', fraction=1.0,
                              contrast='joint_minus_'+comparator, delta=float(point),
                              ci95_low=float(np.quantile(delta_dist,.025)),
                              ci95_high=float(np.quantile(delta_dist,.975)),
                              bonferroni_ci975_low=float(np.quantile(delta_dist,.0125)),
                              bonferroni_ci975_high=float(np.quantile(delta_dist,.9875))))
    write_csv(out/'paired_contrasts.csv', contrasts)
    if test_evidence_groups is not None:
        group_count = np.asarray(test_evidence_groups, dtype=np.int64)
        check(group_count.shape == (len(y),) and np.isin(group_count, [0,1,2,3]).all(),
              'Post-test evidence-group count does not align to test endpoints')
        by_group = []
        for group in range(4):
            # The pre-specified mechanism diagnostic is ATTACK-ONLY conditional recall.
            # Normal class has zero attack evidence by construction; multiclass
            # macro-F1 on these conditional strata would be misleading.
            mask = (y != 0) & (group_count == group)
            for name in ('joint','late_mean','late_stack'):
                arr = predictions[name]
                values = [(p[mask] == y[mask]).mean() for p in arr] if mask.sum() else []
                by_group.append(dict(group_count=group, method=name,
                                     attack_endpoints=int(mask.sum()),
                                     correct_scenario_recall_mean=float(np.mean(values)) if values else None,
                                     fit_sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
        write_csv(out/'fusion_by_evidence_groups.csv', by_group)
    write_json(out/'analysis.json', dict(protocol_id=protocol['protocol_id'],
                                        draws=draws, dataset_id=env['build']['dataset_id'],
                                        common_endpoints=len(y), test_episodes=len(set(ep)),
                                        label_efficiency=summary,
                                        comparison_families={'label_budget':5,'fusion':2},
                                        uncertainty='Class-stratified paired whole-test-episode bootstrap. Training seeds and episode sub-samples FIXED; confidence intervals do not cover new generator roots or deployment.',
                                        preregistration='This is a revision analysis designed after seeing earlier main benchmark results; contrasts fixed before this Step73 test scoring.',
                                        interpretation='Differences are comparative model results under this finite protocol, not proof of Semantic22 information gain or deployment generalization.',
                                        predictions_sha256=digest(out/'predictions_L10.npz')))
    return detail, contrasts


def pack_review(out, status):
    """Only portable summaries/provenance/predictions; no model, giant features, raw logs."""
    allowed = ['protocol.json','preflight.json','population.json','resources.json','source_overlap_train_val.csv',
               'subset_design.csv','source_overlap_test.csv','fusion_by_evidence_groups.csv','encoder_RawScoped.json','encoder_Semantic22.json',
               'training_complete.json','predictions_L10.npz','model_results.csv',
               'paired_contrasts.csv','analysis.json','run_manifest.json','failure.json','error.txt']
    destination = out.parent/(out.name+'_review.zip')
    tmp = destination.with_name(destination.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in allowed:
            path = out/name
            if path.exists():
                z.write(path, arcname=name)
        for pattern in ('*/training_report.json', '*/validation_report.json', '*/complete.json'):
            for p in sorted((out/'jobs').glob(pattern)):
                z.write(p, arcname=p.relative_to(out).as_posix())
        z.writestr('review_status.json', json.dumps(dict(status=status, created_at_utc=stamp()), indent=2))
    os.replace(tmp, destination)
    return destination


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--step59', type=Path, default=None, help='Explicit current Step59 v2 Python source')
    p.add_argument('--mode', choices=['preflight','run'], default='preflight')
    p.add_argument('--run-name', default='run_novelty_v1')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--memory-budget-gib', type=float, default=8.)
    p.add_argument('--bootstrap', type=int, default=20000)
    return p


def run(args):
    root = args.project_root.expanduser().resolve()
    check(root.is_dir(), 'Project root not found: '+str(root))
    check(Path(args.run_name).name == args.run_name and args.run_name not in ('.','..',''),
          'Invalid --run-name')
    check(args.threads > 0 and args.memory_budget_gib > 0 and args.bootstrap >= 1000,
          'Invalid runtime settings')
    step59_path = locate_step59(root, args.step59)
    step59 = import_step59(step59_path)
    out = root/'outputs'/'shared_nominal_v2'/'73_novelty_evidence'/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    with lock_run(out/'.run.lock'):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            import sklearn
            print('[Step73] Audited Step59 v2 benchmark; no previous results modified.', flush=True)
            # Step59 audits all raw/paired projection rows and split hashes.
            temp = Path(tempfile.mkdtemp(prefix='.step73_source_audit_',dir=out))
            try:
                env = step59.prepare_sources(root, temp)
                populations = validate_population(env,step59)
                write_json(out/'population.json', dict(dataset_id=env['build']['dataset_id'], **populations))
                resources = preflight_resources(env,step59,out,args.memory_budget_gib)
                samples, design = subset_design(env,step59)
                write_csv(out/'subset_design.csv', design)
                # Source audit exposes NO test-group-based selection decisions.
                overlap, _ = evidence_overlap(env,step59,root,out)
                payload = dict(version=VERSION, code_sha256=digest(Path(__file__)),
                               step59_source=str(step59_path),step59_sha256=digest(step59_path),
                               dataset_id=env['build']['dataset_id'],
                               project_input_hashes=env['inputs'],
                               fractions=FRACTIONS,subset_seeds=SUBSET_SEEDS,
                               fusion_seeds=list(step59.SEEDS),classes=list(step59.LABELS),
                               features='Step59 v2; category dictionaries fit to FULL unlabeled original train split, never validation or test',
                               labeled_episode_subsets='Three nested class-stratified episode samples; all eligible L10 endpoints of each chosen episode',
                               scope_policy=step59.SCOPE_POLICY,
                               temporal_statistics='mean,std(ddof=0),min,max,last over 10 one-hot/numeric windows',
                               training=step59.DEFAULT_TRAINING,
                               group_schema={k:list(v) for k,v in SOURCE_GROUPS.items()},
                               posttest_mechanism='Attack-only correct-scenario recall conditioned on 0/1/2/3 evidence groups in the last 10 windows; paired-nominal, descriptive only',
                               fusion_methods=['joint HGB','uniform probability averaging of three group HGB','validation-only logistic stacking of log probabilities'],
                               primary='Low-label normalized log-fraction AULC Semantic22-RawScoped over 5,10,25% (3 subset seeds)',
                               secondary='Five fraction-wise contrasts, Bonferroni within five; joint-minus-late_mean and joint-minus-late_stack, Bonferroni within two',
                               bootstrap_draws=args.bootstrap,bootstrap_seed=BOOTSTRAP_SEED,
                               threads=args.threads,memory_budget_gib=args.memory_budget_gib,
                               software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                                             sklearn=sklearn.__version__,platform=platform.platform()),
                               claim_scope='Post-hoc revision experiment, fixed synthetic benchmark, not independent operational evaluation')
                protocol = frozen_protocol(out,payload)
                write_json(out/'preflight.json', dict(status='PREFLIGHT_PASS',updated_at=stamp(),
                                                      population=populations,resource_matrix_count=len(resources),
                                                      train_val_overlap_cells=len(overlap),
                                                      protocol_id=protocol['protocol_id']))
                if args.mode == 'preflight':
                    print('[Step73] PREFLIGHT PASS. No classifier was trained, no test predictions scored.', flush=True)
                    print('  '+str(out/'preflight.json'), flush=True)
                    print('  '+str(out/'source_overlap_train_val.csv'), flush=True)
                    return 0
                (out/'error.txt').unlink(missing_ok=True)
                train_everything(env,step59,out,protocol,samples,args.memory_budget_gib)
                train_missing_joint_seeds(env,step59,out,protocol,args.memory_budget_gib)
                # Only now may the code score the untouched common test set.
                predictions = predict_all(env,step59,out,protocol,args.memory_budget_gib)
                # Group-specific TEST evidence strata are reported only AFTER
                # all fitting and scoring protocols were frozen/completed.
                _, test_counts = evidence_overlap(env,step59,root,out,splits=('test',))
                detail, contrasts = analyze(env,step59,out,protocol,predictions,args.bootstrap,
                                            test_evidence_groups=test_counts['test'])
                for path, sha in env['inputs'].items():
                    check(digest(path)==sha, 'Upstream data changed during Step73: '+path)
                check(digest(step59_path)==payload['step59_sha256'], 'Step59 changed during run')
                check(digest(Path(__file__))==payload['code_sha256'], 'Step73 changed during run')
                write_json(out/'run_manifest.json', dict(status='NOVELTY_CONTROLS_COMPLETE',
                                                         protocol_id=protocol['protocol_id'],
                                                         dataset_id=payload['dataset_id'],
                                                         records=len(detail),contrasts=len(contrasts),
                                                         completed_at=stamp(),test_used_for_selection=False))
                arc=pack_review(out,'NOVELTY_CONTROLS_COMPLETE')
                print('[Step73] DONE: '+str(out/'paired_contrasts.csv'), flush=True)
                print('[Step73] Review bundle: '+str(arc), flush=True)
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                write_json(out/'failure.json',dict(status='FAILED',timestamp=stamp(),type=type(exc).__name__,message=str(exc)))
                arc=pack_review(out,'FAILED')
                print('\n[Step73] STOPPED: '+str(exc)+'\nReview bundle: '+str(arc),file=sys.stderr,flush=True)
                return 1
            finally:
                shutil.rmtree(temp,ignore_errors=True)


if __name__ == '__main__':
    sys.exit(run(argument_parser().parse_args()))
