#!/usr/bin/env python3
"""Step66: evidence transitions and residual alarms in the existing episode streams.

Run beside the original numbered scripts, after a COMPLETE Step65:
    python "66.evidence_transition_alarm_evaluation.py"
    python "66.evidence_transition_alarm_evaluation.py" --preflight-only
    python "66.evidence_transition_alarm_evaluation.py" --self-test

This is the supportable, retrospective part of the proposed continuity experiment.
It does NOT synthesize/concatenate episodes, train models, infer new probabilities,
change Step65 thresholds, or invent operational recovery/benign-lookalike labels.
It uses every existing within-episode audit transition and all five Step65 methods.

An audit intervention burst or observable-change run is NOT necessarily a separate
security incident. A score above threshold at a later run is not automatically a
new alert: a new alert needs an observed consecutive OFF->ON transition. Unknown
initial/warm-up alarm starts are recorded separately. Responses end before the next
same-kind onset; an alarm start cannot be credited to two same-kind events.

A terminal suffix with all three audit flags zero is called audit-quiet, NOT
verified recovery. Five consecutive nonalarm decisions confirm output silence,
not system recovery. There are no recall/false-alarm claims for quiet attack rows.
False positives are evaluated ONLY on NORMAL-labelled original episodes.

Sources: original Step65 result directory plus dataset/shared_nominal_v2 audit and
split files. No checkpoint, embedding, GPU, PyTorch or Transformers needed. NumPy,
pandas and optional threadpoolctl are sufficient. Each seed's analysis is saved
atomically and reused after checksum validation. Sources are read-only.

Fixed focal family: ActualHistory-MaxHold10, 2 validation budgets x 2 outcomes:
  interior-gap correct-type alarm fraction (higher) and first-10-window audit-quiet
  suffix alarm fraction (lower). Four-way Bonferroni percentile intervals are
  descriptive/post-hoc, conditional on fixed heads, encoder and calibration.
All other intervals are pointwise exploratory. Sparse/undefined macros are NA,
never silently computed over fewer attack classes. No winner is selected by test.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

VERSION = 'step66_within_episode_evidence_transitions_v1'
STEP65_SHA = '2d55e4da2748fb26959fdbbc9d7c86240feabc5011bfd6caf61533e3cc7895c5'
DATASET_ID = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
STEP62_ID = '2586c25f0b1a16d8fcd50fe025d46396d6a7da322ecba7b119bc401f9c2b3fe8'
METHODS = ('ActualHistory', 'RepeatCurrent', 'MaxHold10', 'ProbabilityMean10', 'FiniteEWMA10')
SEEDS = tuple(range(20260610, 20260620))
TARGETS = (.01, .05)
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
META = ('scenario_id', 'episode_id', 'episode_type', 'episode_step', 'window_id', 'timestamp', 'label')
AUDIT = ('audit_observable_effect_present', 'audit_observable_difference_count',
         'audit_persistent_effect_present', 'audit_attack_active')
DEFAULT65 = Path('outputs/shared_nominal_v2/65_alarm_memory_controls/run_memory_controls_v1')
DEFAULT_OUT = Path('outputs/shared_nominal_v2/66_evidence_transitions/run_evidence_transitions_v1')
DEFAULT_DATA = Path('dataset/shared_nominal_v2')
HORIZON, TAIL_WIDTH, QUIET_RUN = 10, 10, 5
FOCAL_SIZE = 4


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


def clean_json(x):
    if isinstance(x, dict): return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def obj_hash(x):
    return hashlib.sha256(json.dumps(clean_json(x), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_csv(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip', **kwargs)


def write_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    compression = {'method': 'gzip', 'mtime': 0} if path.name.endswith('.gz') else None
    frame.to_csv(tmp, index=False, encoding='utf-8-sig', float_format='%.17g', compression=compression)
    os.replace(tmp, path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def safe_child(root, relative):
    relative = str(relative)
    require('\\' not in relative and not PureWindowsPath(relative).drive,
            'Unsafe relative manifest path: ' + relative)
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and '..' not in rel.parts and rel.parts,
            'Unsafe relative manifest path: ' + relative)
    root = Path(root).resolve(); child = (root / relative).resolve()
    require(root in child.parents, 'Path escapes the source/output directory')
    return child


def check_file(path, sha):
    require(Path(path).is_file(), 'Required input missing: ' + str(path))
    require(digest(path) == sha, 'SHA256 mismatch; restore the original input, not the hash: ' + str(path))


def resolve(root, path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@contextmanager
def exclusive_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    f = (out / '.step66.lock').open('a+b')
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
            raise ValueError('Another Step66 process owns this output directory.') from exc
        yield
    finally:
        f.close()


def positive_runs(mask):
    """Inclusive local-index bounds; no dilation or gap merging."""
    a = np.asarray(mask)
    require(a.ndim == 1 and np.isin(a, (0, 1)).all(), 'Run mask must be a binary vector')
    a = a.astype(bool)
    edges = np.diff(np.r_[False, a, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist()))


def alarm_edges(alarms, scored):
    a, q = np.asarray(alarms, bool), np.asarray(scored, bool)
    require(a.ndim == 1 and a.shape == q.shape, 'Alarm/scored shape mismatch')
    prev_known = np.r_[False, q[:-1]]
    prev_alarm = np.r_[False, a[:-1]]
    return a & q & prev_known & ~prev_alarm, a & q & ~prev_known


def event_response(alarms, scored, start, next_start, horizon=HORIZON):
    """Inclusive [start,start+H], stopped strictly before the next same-kind onset.

    Full follow-up is based ONLY on record/next-onset coverage. Early detections do
    not rescue censored events. Initial L10 warm-up is retained as unavailable time.
    A run present at episode step0 is left-boundary uncertain, not an observed onset.
    """
    a, q = np.asarray(alarms, bool), np.asarray(scored, bool)
    require(a.shape == q.shape and a.ndim == 1 and 0 <= start < len(a), 'Bad event range')
    require(next_start is None or start < next_start < len(a), 'Bad next onset')
    deadline = int(start + horizon)
    end = min(deadline, len(a)-1, int(next_start)-1 if next_start is not None else len(a)-1)
    ix = np.arange(int(start), end+1)
    rising, unknown = alarm_edges(a, q)
    any_ix, new_ix = ix[a[ix] & q[ix]], ix[rising[ix]]
    before = int(a[start-1]) if start > 0 and q[start-1] else -1
    due_next = next_start is not None and next_start <= deadline
    due_end = len(a)-1 < deadline
    full = start > 0 and not due_next and not due_end and bool(q[deadline])
    return dict(response_end_step=end, deadline_step=deadline,
        fully_followed=bool(full), left_boundary_onset=bool(start == 0),
        censored_by_next_onset=bool(due_next), censored_by_record_end=bool(due_end),
        scored_response_windows=int(q[ix].sum()), unscored_response_windows=int((~q[ix]).sum()),
        preexisting_alarm=before, any_alarm_observed=bool(len(any_ix)),
        new_alarm_observed=bool(len(new_ix)),
        any_alarm_delay_seconds=int(any_ix[0]-start) if len(any_ix) else None,
        new_alarm_delay_seconds=int(new_ix[0]-start) if len(new_ix) else None,
        unknown_start_high_count=int(unknown[ix].sum()))


def gap_response(alarms, correct_alarms, scored, start, end):
    a, c, q = np.asarray(alarms, bool), np.asarray(correct_alarms, bool), np.asarray(scored, bool)
    require(a.shape == c.shape == q.shape and 0 <= start <= end < len(a), 'Bad gap range')
    ix = np.arange(start, end+1); eligible = ix[q[ix]]; full = bool(q[ix].all())
    longest = max((b-s+1 for s, b in positive_runs(~a[ix])), default=0) if full else None
    return dict(scored_windows=int(len(eligible)), unscored_windows=int(len(ix)-len(eligible)),
                alarm_windows=int(a[eligible].sum()), correct_type_alarm_windows=int(c[eligible].sum()),
                fully_scored=full, longest_no_alarm_run=longest)


def tail_response(alarms, scored, start, width=TAIL_WIDTH, quiet_run=QUIET_RUN):
    """First `width` one-second decision windows: indices [start,start+width).

    Silence confirmation timestamp is the last of `quiet_run` consecutive known
    nonalarm decisions. With five silent decisions beginning at start, delay=4 s.
    No confirmation in the first10 windows -> capped delay=10 s. An initially
    silent model can succeed; this is deliberately NOT 'cleared after detection'.
    """
    a, q = np.asarray(alarms, bool), np.asarray(scored, bool)
    require(a.shape == q.shape and 0 <= start < len(a) and 1 <= quiet_run <= width, 'Bad tail range')
    ix = np.arange(start, min(start+width, len(a))); eligible = ix[q[ix]]
    full = len(ix) == width and bool(q[ix].all())
    delays = [s+quiet_run-1 for s, e in positive_runs(~a[ix] & q[ix]) if e-s+1 >= quiet_run]
    delay = int(delays[0]) if delays else None
    return dict(fully_followed=bool(full), observed_windows_first10=len(ix),
        scored_windows_first10=len(eligible), unscored_windows_first10=int((~q[ix]).sum()),
        right_censored=bool(len(ix) < width),
        alarm_at_tail_start=int(a[start]) if q[start] else -1,
        alarm_before_tail=int(a[start-1]) if start > 0 and q[start-1] else -1,
        observed_alarm_windows=int(a[eligible].sum()),
        alarm_fraction_first10=float(a[ix].mean()) if full else None,
        silence_confirmed=bool(delay is not None) if full else None,
        silence_confirmation_delay_seconds=delay if full else None,
        capped_confirmation_delay_seconds=(delay if delay is not None else width) if full else None)


EP_COLS = ['episode_index','episode_id','label','label_id','start_row','windows','eligible_endpoints',
           'observable_runs','intervention_runs','observable_windows','active_windows','persistent_windows']
EV_COLS = ['event_id','episode_index','episode_id','label','label_id','anchor','event_number',
           'start_step','end_step','duration_windows','start_timestamp','end_timestamp','next_start_step',
           'left_boundary_onset','fully_followed_10s','censored_by_next_onset','censored_by_record_end']
GAP_COLS = ['gap_id','episode_index','episode_id','label','label_id','start_step','end_step',
            'duration_windows','length_bin','active_marked_windows','persistent_marked_windows',
            'fully_scored','eligible_windows','preceding_event_start','following_event_start']
TAIL_COLS = ['tail_id','episode_index','episode_id','label','label_id','kind','available','reason',
             'start_step','end_step','duration_windows','active_marked_windows','persistent_marked_windows',
             'fully_followed_first10','right_censored_first10','warmup_in_first10']


def make_catalogs(windows):
    """Catalogue from audit data only, before any Step65 test scores are opened."""
    episodes, events, gaps, tails = [], [], [], []
    for ei, (eid, ix) in enumerate(windows.groupby('episode_id', sort=False).indices.items()):
        ix = np.asarray(ix); w = windows.iloc[ix]; n = len(w); lab = str(w.label.iloc[0]); li = LABELS.index(lab)
        a = w.audit_attack_active.to_numpy(bool); p = w.audit_persistent_effect_present.to_numpy(bool)
        d = w.audit_observable_effect_present.to_numpy(bool); q = np.arange(n) >= 9
        dr, ar = positive_runs(d), positive_runs(a)
        base = dict(episode_index=ei, episode_id=str(eid), label=lab, label_id=li)
        episodes.append(dict(base, start_row=int(ix[0]), windows=n, eligible_endpoints=int(q.sum()),
            observable_runs=len(dr), intervention_runs=len(ar), observable_windows=int(d.sum()),
            active_windows=int(a.sum()), persistent_windows=int(p.sum())))
        if li == 0: continue
        for anchor, runs in [('observable', dr), ('intervention', ar)]:
            for j, (s, e) in enumerate(runs):
                nxt = runs[j+1][0] if j+1 < len(runs) else None
                info = event_response(np.zeros(n, bool), q, s, nxt)
                events.append(dict(base, event_id=f'{ei}:{anchor}:{j+1}', anchor=anchor,
                    event_number=j+1, start_step=s, end_step=e, duration_windows=e-s+1,
                    start_timestamp=str(w.timestamp.iloc[s]), end_timestamp=str(w.timestamp.iloc[e]),
                    next_start_step=nxt, left_boundary_onset=s == 0,
                    fully_followed_10s=info['fully_followed'],
                    censored_by_next_onset=info['censored_by_next_onset'],
                    censored_by_record_end=info['censored_by_record_end']))
        for j, ((s0, e0), (s1, e1)) in enumerate(zip(dr[:-1], dr[1:])):
            s, e = e0+1, s1-1; length = e-s+1
            gaps.append(dict(base, gap_id=f'{ei}:gap:{j+1}', start_step=s, end_step=e,
                duration_windows=length, length_bin='1-3' if length <= 3 else '4-9' if length <= 9 else '10+',
                active_marked_windows=int(a[s:e+1].sum()), persistent_marked_windows=int(p[s:e+1].sum()),
                fully_scored=bool(q[s:e+1].all()), eligible_windows=int(q[s:e+1].sum()),
                preceding_event_start=s0, following_event_start=s1))
        for kind, flag in [('after_last_observable_change', d), ('after_last_marked_activity', a | p | d)]:
            pos = np.flatnonzero(flag)
            s = int(pos[-1]+1) if len(pos) else None
            available = s is not None and s < n
            reason = 'available_not_verified_recovery' if available else 'no_prior_marked_activity' if s is None else 'no_suffix_before_record_end'
            tails.append(dict(base, tail_id=f'{ei}:{kind}', kind=kind, available=available, reason=reason,
                start_step=s if available else None, end_step=n-1 if available else None,
                duration_windows=n-s if available else 0,
                active_marked_windows=int(a[s:].sum()) if available else 0,
                persistent_marked_windows=int(p[s:].sum()) if available else 0,
                fully_followed_first10=bool(available and s >= 9 and s+TAIL_WIDTH <= n),
                right_censored_first10=bool(available and s+TAIL_WIDTH > n),
                warmup_in_first10=bool(available and s < 9)))
    return (pd.DataFrame(episodes, columns=EP_COLS), pd.DataFrame(events, columns=EV_COLS),
            pd.DataFrame(gaps, columns=GAP_COLS), pd.DataFrame(tails, columns=TAIL_COLS))


def safe_div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def bootstrap_ratios(num, den, labels, class_sets, draws, seed):
    """Class-stratified paired WHOLE-EPISODE bootstrap; repeated runs not iid.

    num[K,E,M] has numerators averaged over fixed heads; den[K,E] is shared across
    heads/methods. Same episode multiplicities across all metrics and methods.
    """
    num, den, labels = np.asarray(num,float), np.asarray(den,float), np.asarray(labels)
    require(num.ndim == 3 and den.shape == num.shape[:2] and labels.shape == (num.shape[1],), 'Bootstrap shape mismatch')
    require(len(class_sets) == len(num) and draws > 0 and all(len(x) for x in class_sets), 'Bad bootstrap specification')
    require(np.isfinite(num).all() and np.isfinite(den).all() and (den >= 0).all(), 'Bad bootstrap components')
    k, _, m = num.shape; classes = sorted({c for group in class_sets for c in group})
    point = np.zeros((k,m)); groups = {}; support = [[] for _ in range(k)]
    for c in classes:
        ids = np.flatnonzero(labels == c); selected = np.flatnonzero([c in cs for cs in class_sets])
        groups[c] = (ids, selected)
        ratios = safe_div(num[:,ids].sum(1), den[:,ids].sum(1)[:,None])
        for j in selected:
            point[j] += ratios[j] / len(class_sets[j])
            support[j].append(int((den[j,ids] > 0).sum()))
    samples = np.zeros((draws,k,m)); rng = np.random.default_rng(seed)
    for start in range(0,draws,256):
        size = min(256,draws-start)
        for c,(ids,selected) in groups.items():
            n = len(ids)
            if not n: samples[start:start+size,selected] = np.nan; continue
            weights = rng.multinomial(n, np.full(n,1./n), size=size)
            bn = (weights @ num[:,ids].transpose(1,0,2).reshape(n,k*m)).reshape(size,k,m)
            bd = weights @ den[:,ids].T
            ratios = safe_div(bn,bd[:,:,None])
            for j in selected: samples[start:start+size,j] += ratios[:,j]/len(class_sets[j])
    status = ['missing_target_class' if min(x)==0 else 'fewer_than_two_eligible_episodes_in_class' if min(x)<2 else
              'undefined_bootstrap_draw' if not np.isfinite(samples[:,j]).all() else 'ok'
              for j,x in enumerate(support)]
    return point,samples,status


def load_step65(run):
    """Verify artifacts and lock; do not deserialize any test-score NPZ yet."""
    run = Path(run).resolve(); manifest_path = run/'run_manifest.json'
    require(manifest_path.is_file(), 'Completed Step65 directory not found: '+str(run)+
            '\nKeep the original outputs folder, or pass --step65-run PATH.')
    manifest = read_json(manifest_path)
    require(manifest.get('status') == 'STEP65_ALARM_MEMORY_COMPLETE', 'Step65 is not complete')
    hashes = manifest.get('output_sha256', {})
    needed = {'protocol.json','thresholds.json','thresholds.csv','threshold_lock.json',
              'endpoint_evidence_test.csv','source/65.alarm_memory_controls.py'}
    needed.update(f'endpoint_predictions/test_scores_{s}.npz' for s in SEEDS)
    require(needed <= set(hashes), 'Step65 manifest lacks required artifacts: '+str(sorted(needed-set(hashes))))
    verified = {str(manifest_path):digest(manifest_path)}
    for rel, sha in hashes.items():
        path = safe_child(run,rel); check_file(path,sha); verified[str(path)] = sha
    p = read_json(run/'protocol.json')
    require(p.get('identity') == obj_hash({k:v for k,v in p.items() if k!='identity'}) == manifest['protocol_identity'],
            'Step65 protocol identity mismatch')
    require(p.get('version') == 'step65_finite_alarm_memory_v1' and p.get('script_sha256') == STEP65_SHA,
            'Use the reviewed Step65 v1 code/results; do not replace pins')
    check_file(run/'source/65.alarm_memory_controls.py',STEP65_SHA)
    require(p.get('dataset_id') == DATASET_ID and p.get('source_step62_identity') == STEP62_ID,
            'Step65 uses a different dataset/Step62 lineage')
    require(tuple(p['methods']) == METHODS and tuple(p['head_seeds']) == SEEDS and
            tuple(p['validation_fpr_targets']) == TARGETS and p['memory_length'] == 10 and
            p['finite_ewma_alpha'] == .25 and p['training'] is False, 'Step65 method/seed contract mismatch')
    lock = read_json(run/'threshold_lock.json'); td = read_json(run/'thresholds.json')
    require(lock['protocol_identity'] == td['protocol_identity'] == p['identity'] and
            lock.get('all_validation_jobs_complete') is True and lock.get('test_scoring_started') is False and
            lock['thresholds_sha256'] == digest(run/'thresholds.json'), 'Step65 threshold lock is invalid')
    rows = pd.DataFrame(td['rows']); csv = read_csv(run/'thresholds.csv')
    require(set(rows.columns) == set(csv.columns) and len(rows) == len(csv) == len(SEEDS)*len(METHODS)*2,
            'Threshold rows differ')
    for col in rows:
        if pd.api.types.is_numeric_dtype(rows[col]):
            require(np.array_equal(rows[col].to_numpy(),csv[col].to_numpy()), 'Threshold CSV/JSON differs: '+col)
        else:
            require(rows[col].astype(str).equals(csv[col].astype(str)), 'Threshold CSV/JSON differs: '+col)
    thresholds = np.full((len(SEEDS),len(METHODS),len(TARGETS)),np.nan); seen = set()
    for r in rows.to_dict('records'):
        key = (r['seed'],r['method'],r['target_normal_endpoint_fpr'])
        require(key not in seen and key[0] in SEEDS and key[1] in METHODS and key[2] in TARGETS, 'Unknown/duplicate threshold key')
        seen.add(key); value = float(r['threshold'])
        require(np.isfinite(value) and 0 <= value <= 1, 'Invalid locked threshold')
        require(0 <= r['validation_alarm_count'] <= r['validation_allowed_alarm_count'], 'Validation budget violation')
        thresholds[SEEDS.index(key[0]),METHODS.index(key[1]),TARGETS.index(key[2])] = value
    require(np.isfinite(thresholds).all(), 'Incomplete locked thresholds')
    ep = read_csv(run/'endpoint_evidence_test.csv')
    require(len(ep)>0 and not ep.duplicated(['episode_id','episode_step']).any(), 'Bad Step65 endpoint table')
    return dict(run65=run,protocol65=p,manifest=manifest,verified=verified,thresholds=thresholds,
                endpoints=ep,threshold_rows=rows)


def source_input_sha(protocol, suffix):
    suffix = suffix.replace('\\','/').lower()
    found = [sha for path,sha in protocol['input_sha256'].items()
             if str(path).replace('\\','/').lower().endswith('/'+suffix)]
    require(len(found)==1, 'Missing/ambiguous recorded source hash for '+suffix)
    return found[0]


def validate_alignment(windows, endpoints):
    require(list(windows.columns) == list(META)+list(AUDIT),
            'Unexpected window audit schema. Recovery flags must not be invented or silently ignored.')
    require(len(windows)>0 and not windows.isna().any().any() and
            not windows.duplicated(['episode_id','episode_step']).any(), 'Missing/duplicate audit windows')
    require(windows.label.isin(LABELS).all(), 'Unknown audit label')
    for col in ('episode_step','window_id')+AUDIT:
        require(pd.api.types.is_integer_dtype(windows[col].dtype), 'Expected integer audit column: '+col)
    for col in (AUDIT[0],AUDIT[2],AUDIT[3]):
        require(windows[col].isin((0,1)).all(), 'Nonbinary audit flag: '+col)
    require(windows[AUDIT[1]].between(0,22).all() and
            np.array_equal(windows[AUDIT[0]].to_numpy(bool),windows[AUDIT[1]].to_numpy()>0), 'Audit count/flag mismatch')
    require((windows.loc[windows.label.eq('NORMAL'),list(AUDIT)]==0).all().all(), 'NORMAL audit has attack effects')
    ts = pd.to_datetime(windows.timestamp,utc=True,errors='raise')
    require(not ts.isna().any(), 'Missing timestamp')
    seconds = (ts-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds().to_numpy()
    ends = []
    for eid,ix in windows.groupby('episode_id',sort=False).indices.items():
        ix=np.asarray(ix); w=windows.iloc[ix]
        require(str(eid).strip() and np.array_equal(ix,np.arange(ix[0],ix[-1]+1)), 'Noncontiguous episode rows')
        require(w.label.nunique()==1 and w.scenario_id.nunique()==1 and w.episode_type.nunique()==1, 'Mixed episode metadata')
        require(np.array_equal(w.episode_step,np.arange(len(ix))), 'Episode steps must start at0 and be consecutive')
        require(np.all(np.diff(seconds[ix])==1), 'Non-one-second within-episode cadence')
        ends.extend(ix[9:])
    ends = np.asarray(ends,dtype=np.int64)
    require(np.array_equal(endpoints.sequence_id,np.arange(len(ends))) and
            np.array_equal(endpoints.sequence_end_row,ends) and
            np.array_equal(endpoints.sequence_start_row,ends-9), 'L10 endpoint coverage/alignment differs')
    require(endpoints.source_split.eq('test').all() and
            np.array_equal(endpoints.label.map(dict(zip(LABELS,range(6)))),endpoints.label_id), 'Endpoint label/split mismatch')
    expected = windows.iloc[ends].reset_index(drop=True)
    for col in META:
        require(np.array_equal(expected[col],endpoints[col]), 'Window/endpoint metadata mismatch: '+col)
    require(np.array_equal(endpoints.sequence_end_step,endpoints.episode_step), 'Endpoint end-step mismatch')
    changed=windows[AUDIT[0]].to_numpy(bool); past=changed[ends[:,None]-np.arange(1,10)]
    count=past.sum(1); current=changed[ends]; lag=np.where(count>0,past.argmax(1)+1,0)
    for col,value in [('current_changed',current.astype(int)),('past_changed_count',count),('most_recent_past_change_lag',lag)]:
        require(np.array_equal(endpoints[col],value), 'Step65 evidence alignment mismatch: '+col)
    strata=np.where(current,'current_changed',np.where(count>0,'history_only_changed','no_change_in_L10'))
    strata=np.where(endpoints.label_id.to_numpy()==0,'NORMAL',strata)
    require(np.array_equal(endpoints.evidence_stratum,strata), 'Step65 evidence strata mismatch')


def load_dataset(ctx, dataset):
    dataset=Path(dataset).resolve(); p=ctx['protocol65']; paths={}
    for rel in ('build_manifest.json','audit/test_window_observable_effects.csv','splits/test_observable.csv'):
        path=dataset/rel; sha=source_input_sha(p,'dataset/shared_nominal_v2/'+rel)
        check_file(path,sha);ctx['verified'][str(path)]=sha;paths[rel]=path
    require(read_json(paths['build_manifest.json'])['dataset_id']==DATASET_ID, 'Build manifest dataset differs')
    w=read_csv(paths['audit/test_window_observable_effects.csv'])
    meta=read_csv(paths['splits/test_observable.csv'],usecols=list(META))
    require(len(meta)==len(w),'Audit/learning window count mismatch')
    for col in META:require(np.array_equal(w[col],meta[col]), 'Audit/learning metadata mismatch: '+col)
    validate_alignment(w,ctx['endpoints'])
    ctx.update(windows=w,dataset=dataset,dataset_paths=paths)


def load_scores(ctx, seed):
    rel=f'endpoint_predictions/test_scores_{seed}.npz'; path=ctx['run65']/rel
    check_file(path,ctx['manifest']['output_sha256'][rel])
    with np.load(path,allow_pickle=False) as z:
        require(tuple(z['methods'].tolist())==METHODS and np.array_equal(z['sequence_ids'],ctx['endpoints'].sequence_id), 'Score IDs/methods mismatch')
        score=z['attack_scores'].copy(); typ=z['attack_types'].copy()
    require(score.shape==typ.shape==(len(METHODS),len(ctx['endpoints'])),'Score shape mismatch')
    require(np.isfinite(score).all() and ((score>=0)&(score<=1)).all(), 'Invalid attack scores')
    require(typ.dtype.kind in 'iu' and np.isin(typ,np.arange(1,6)).all(), 'Invalid predicted attack type')
    return score,typ


def metric_specs():
    specs=[]
    def add(metric,group,target,direction='higher',classes=(1,2,3,4,5)):
        specs.append(dict(metric=metric,group=group,target_fpr=target,direction=direction,target_classes=list(classes)))
    for target in TARGETS:
        for metric in ('normal_endpoint_alarm_fraction','normal_episode_any_alarm_probability','normal_observed_runs_per1000_endpoints'):
            add(metric,'NORMAL',target,'lower',(0,))
        for group in ('all','1-3','4-9','10+'):
            add('gap_alarm_fraction',group,target)
            add('gap_correct_type_alarm_fraction',group,target)
        add('gap_mean_longest_no_alarm_run','all',target,'lower')
        for group in ('after_last_observable_change','after_last_marked_activity'):
            direction = 'descriptive' if group=='after_last_observable_change' else 'lower'
            add('tail_alarm_fraction_first10',group,target,direction)
            add('tail_silence_confirmation_probability_first10',group,target,
                'descriptive' if direction=='descriptive' else 'higher')
            add('tail_capped_confirmation_delay_s',group,target,direction)
        for anchor in ('observable','intervention'):
            for subset in ('all','repeated'):
                group=anchor+':'+subset
                add('event_any_alarm_present_by10s',group,target)
                add('event_known_new_alert_by10s',group,target)
                add('event_any_alarm_in_scored_run',group,target)
                add('event_known_new_alert_in_scored_run',group,target)
                add('event_preexisting_alarm_probability',group,target,'descriptive')
    return specs


def group_records(frame):
    return {int(e):g.to_dict('records') for e,g in frame.groupby('episode_index',sort=False)}


def completed_seed(folder, identity, spec_hash):
    path=folder/'complete.json'
    if not path.exists():return None
    record=read_json(path)
    require(record['identity']==identity and record['spec_hash']==spec_hash, 'Seed belongs to another protocol')
    for rel,sha in record['artifacts'].items():check_file(safe_child(folder,rel),sha)
    with np.load(folder/'components.npz',allow_pickle=False) as z:
        num,den=z['numerators'].copy(),z['denominators'].copy()
    return num,den


def evaluate_seed(ctx, catalogs, si, specs, out, identity):
    seed=SEEDS[si]; folder=Path(out)/f'seed_{seed}'; spec_hash=obj_hash(specs)
    cached=completed_seed(folder,identity,spec_hash)
    if cached is not None:
        print(f'  seed {seed}: verified saved analysis reused',flush=True)
        return cached
    folder.mkdir(parents=True,exist_ok=True)
    scores,types=load_scores(ctx,seed)
    episodes,events,gaps,tails=catalogs
    ev_by,gap_by,tail_by=map(group_records,(events,gaps,tails))
    meta=ctx['endpoints']; ep_indices={str(e):np.asarray(ix) for e,ix in meta.groupby('episode_id',sort=False).indices.items()}
    lookup={(s['metric'],s['group'],s['target_fpr']):i for i,s in enumerate(specs)}
    num=np.zeros((len(specs),len(episodes),len(METHODS))); den=np.zeros_like(num)
    event_rows=[];gap_rows=[];tail_rows=[];normal_rows=[]
    def add(metric,group,target,ei,mi,value,denominator):
        k=lookup[(metric,group,target)];num[k,ei,mi]+=value;den[k,ei,mi]+=denominator
    for er in episodes.to_dict('records'):
        ei=int(er['episode_index']);eid=str(er['episode_id']);n=int(er['windows']);li=int(er['label_id'])
        inds=ep_indices.get(eid,np.empty(0,dtype=int))
        positions=meta.iloc[inds].episode_step.to_numpy(dtype=int)
        q=np.zeros(n,bool);q[positions]=True
        require(np.array_equal(np.flatnonzero(q),np.arange(9,n)), 'Seed endpoint coverage differs')
        for mi,method in enumerate(METHODS):
            for ti,target in enumerate(TARGETS):
                a=np.zeros(n,bool);a[positions]=scores[mi,inds]>ctx['thresholds'][si,mi,ti]
                correct=np.zeros(n,bool);correct[positions]=a[positions] & (types[mi,inds]==li)
                info=dict(seed=seed,method=method,target_fpr=target)
                if li==0:
                    lengths=[e-s+1 for s,e in positive_runs(a & q)]
                    ne=int(q.sum());na=int(a[q].sum());nr=len(lengths)
                    add('normal_endpoint_alarm_fraction','NORMAL',target,ei,mi,na,ne)
                    add('normal_episode_any_alarm_probability','NORMAL',target,ei,mi,int(na>0),int(ne>0))
                    add('normal_observed_runs_per1000_endpoints','NORMAL',target,ei,mi,1000*nr,ne)
                    normal_rows.append(dict(info,episode_id=eid,label=er['label'],label_id=li,
                        scored_windows=ne,alarm_windows=na,observed_alarm_runs=nr,
                        longest_observed_alarm_run=max(lengths,default=0),
                        mean_observed_alarm_run_length=float(np.mean(lengths)) if lengths else None,
                        boundary_truncation_possible=True))
                    continue
                rising, unknown_start = alarm_edges(a,q)
                for ev in ev_by.get(ei,[]):
                    nxt=ev['next_start_step'];nxt=None if nxt is None or pd.isna(nxt) else int(nxt)
                    response=event_response(a,q,int(ev['start_step']),nxt)
                    require(response['fully_followed']==bool(ev['fully_followed_10s']), 'Score-dependent event cohort')
                    run_ix=np.arange(int(ev['start_step']),int(ev['end_step'])+1)
                    run_scored=run_ix[q[run_ix]]
                    response.update(scored_positive_run_windows=len(run_scored),
                        positive_run_alarm_windows=int(a[run_scored].sum()),
                        any_alarm_in_scored_run=bool(a[run_scored].any()),
                        known_new_alert_in_scored_run=bool(rising[run_scored].any()),
                        unknown_start_high_in_scored_run=int(unknown_start[run_scored].sum()))
                    event_rows.append(dict(ev,**info,**response))
                    subsets=['all']+(['repeated'] if ev['event_number']>1 else [])
                    for subset in subsets:
                        group=ev['anchor']+':'+subset;full=response['fully_followed']
                        add('event_any_alarm_present_by10s',group,target,ei,mi,int(full and response['any_alarm_observed']),int(full))
                        add('event_known_new_alert_by10s',group,target,ei,mi,int(full and response['new_alarm_observed']),int(full))
                        has_scored_run=bool(len(run_scored))
                        add('event_any_alarm_in_scored_run',group,target,ei,mi,
                            int(response['any_alarm_in_scored_run']),int(has_scored_run))
                        add('event_known_new_alert_in_scored_run',group,target,ei,mi,
                            int(response['known_new_alert_in_scored_run']),int(has_scored_run))
                        known=response['preexisting_alarm']>=0
                        add('event_preexisting_alarm_probability',group,target,ei,mi,
                            int(known and response['preexisting_alarm']==1),int(known))
                for gap in gap_by.get(ei,[]):
                    response=gap_response(a,correct,q,int(gap['start_step']),int(gap['end_step']))
                    gap_rows.append(dict(gap,**info,**response))
                    for group in ('all',gap['length_bin']):
                        add('gap_alarm_fraction',group,target,ei,mi,response['alarm_windows'],response['scored_windows'])
                        add('gap_correct_type_alarm_fraction',group,target,ei,mi,response['correct_type_alarm_windows'],response['scored_windows'])
                    full=response['fully_scored']
                    add('gap_mean_longest_no_alarm_run','all',target,ei,mi,response['longest_no_alarm_run'] if full else 0,int(full))
                for tail in tail_by.get(ei,[]):
                    if not tail['available']:
                        tail_rows.append(dict(tail,**info));continue
                    response=tail_response(a,q,int(tail['start_step']))
                    require(response['fully_followed']==bool(tail['fully_followed_first10']), 'Score-dependent suffix cohort')
                    tail_rows.append(dict(tail,**info,**response));group=tail['kind'];full=response['fully_followed']
                    add('tail_alarm_fraction_first10',group,target,ei,mi,
                        response['observed_alarm_windows'] if full else 0,TAIL_WIDTH if full else 0)
                    add('tail_silence_confirmation_probability_first10',group,target,ei,mi,int(full and response['silence_confirmed']),int(full))
                    add('tail_capped_confirmation_delay_s',group,target,ei,mi,response['capped_confirmation_delay_seconds'] if full else 0,int(full))
    require(np.array_equal(den,np.broadcast_to(den[:,:,:1],den.shape)), 'Method-dependent denominator detected')
    require(np.isfinite(num).all() and np.isfinite(den).all(),'Nonfinite seed components')
    extra=['seed','method','target_fpr']
    event_columns=list(dict.fromkeys(EV_COLS+extra+list(event_response(np.zeros(30,bool),np.arange(30)>=9,10,None))+
        ['scored_positive_run_windows','positive_run_alarm_windows','any_alarm_in_scored_run',
         'known_new_alert_in_scored_run','unknown_start_high_in_scored_run']))
    gap_columns=list(dict.fromkeys(GAP_COLS+extra+list(gap_response(np.zeros(2,bool),np.zeros(2,bool),np.ones(2,bool),0,1))))
    tail_columns=list(dict.fromkeys(TAIL_COLS+extra+list(tail_response(np.zeros(30,bool),np.ones(30,bool),10))))
    write_csv(folder/'events.csv.gz',pd.DataFrame(event_rows,columns=event_columns))
    write_csv(folder/'gaps.csv.gz',pd.DataFrame(gap_rows,columns=gap_columns))
    write_csv(folder/'tails.csv.gz',pd.DataFrame(tail_rows,columns=tail_columns))
    write_csv(folder/'normal_episodes.csv.gz',pd.DataFrame(normal_rows,columns=extra+[
        'episode_id','label','label_id','scored_windows','alarm_windows','observed_alarm_runs',
        'longest_observed_alarm_run','mean_observed_alarm_run_length','boundary_truncation_possible']))
    write_npz(folder/'components.npz',numerators=num,denominators=den[:,:,0],
              episode_ids=episodes.episode_id.to_numpy(dtype=str),methods=np.array(METHODS),seed=np.array(seed))
    artifacts={name:digest(folder/name) for name in ('events.csv.gz','gaps.csv.gz','tails.csv.gz','normal_episodes.csv.gz','components.npz')}
    write_json(folder/'complete.json',dict(identity=identity,spec_hash=spec_hash,seed=seed,artifacts=artifacts,
               completed_at_utc=now(),event_rows=len(event_rows),gap_rows=len(gap_rows),tail_rows=len(tail_rows)))
    print(f'  seed {seed}: saved {len(event_rows):,} event, {len(gap_rows):,} gap, {len(tail_rows):,} suffix records',flush=True)
    return num,den[:,:,0]


def focal_spec(spec):
    return ((spec['metric']=='gap_correct_type_alarm_fraction' and spec['group']=='all') or
            (spec['metric']=='tail_alarm_fraction_first10' and spec['group']=='after_last_marked_activity'))


def summarize(components,den,episodes,specs,out,draws,seed):
    """Every displayed aggregate can be recomputed from saved episode components."""
    values=np.asarray(components,float); labels=episodes.label_id.to_numpy()
    require(values.shape==(len(SEEDS),len(specs),len(episodes),len(METHODS)), 'Summary component shape mismatch')
    class_sets=[s['target_classes'] for s in specs]
    points,samples,status=bootstrap_ratios(values.mean(0),den,labels,class_sets,draws,seed)
    head_points=np.zeros((len(SEEDS),len(specs),len(METHODS)))
    class_rows=[];head_rows=[];support_rows=[]
    for j,spec in enumerate(specs):
        for c in spec['target_classes']:
            ix=np.flatnonzero(labels==c);dd=den[j,ix].sum()
            cp=safe_div(values[:,j][:,ix].sum(1),dd)
            head_points[:,j]+=cp/len(spec['target_classes'])
            support_rows.append(dict(spec,label=LABELS[c],eligible_episodes=int((den[j,ix]>0).sum()),denominator_total=float(dd)))
            for mi,method in enumerate(METHODS):
                class_rows.append(dict(spec,label=LABELS[c],method=method,estimate=float(cp[:,mi].mean()),
                    fixed_head_sd=float(cp[:,mi].std(ddof=1)),eligible_episodes=int((den[j,ix]>0).sum()),denominator_total=float(dd)))
        for si,head in enumerate(SEEDS):
            for mi,method in enumerate(METHODS):head_rows.append(dict(spec,seed=head,method=method,estimate=head_points[si,j,mi]))
    rows=[];contrasts=[];focals=[];fsamples=[]
    for j,spec in enumerate(specs):
        q=np.quantile(samples[:,j],[.025,.975],axis=0) if status[j]=='ok' else np.full((2,len(METHODS)),np.nan)
        for mi,method in enumerate(METHODS):
            rows.append(dict(spec,method=method,estimate=points[j,mi],ci95_low=q[0,mi],ci95_high=q[1,mi],
                fixed_head_sd=head_points[:,j,mi].std(ddof=1),ci_status=status[j]))
        for mi in range(1,len(METHODS)):
            delta=samples[:,j,0]-samples[:,j,mi];hp=head_points[:,j,0]-head_points[:,j,mi]
            qc=np.quantile(delta,[.025,.975]) if status[j]=='ok' else [np.nan,np.nan]
            record=dict(spec,reference=METHODS[0],comparator=METHODS[mi],
                reference_estimate=points[j,0],comparator_estimate=points[j,mi],difference=points[j,0]-points[j,mi],
                ci95_low=qc[0],ci95_high=qc[1],paired_fixed_head_sd=float(hp.std(ddof=1)),
                positive_head_differences=int((hp>0).sum()),negative_head_differences=int((hp<0).sum()),
                ci_status=status[j],focal=bool(mi==2 and focal_spec(spec)))
            contrasts.append(record)
            if record['focal']:
                qa=np.quantile(delta,[.05/(2*FOCAL_SIZE),1-.05/(2*FOCAL_SIZE)]) if status[j]=='ok' else [np.nan,np.nan]
                focals.append(dict(record,family_size=FOCAL_SIZE,family_adjusted_low=qa[0],family_adjusted_high=qa[1]))
                fsamples.append(delta)
    require(len(focals)==FOCAL_SIZE, 'Focal family changed')
    write_csv(out/'metrics_by_seed.csv',pd.DataFrame(head_rows))
    write_csv(out/'metric_intervals.csv',pd.DataFrame(rows))
    write_csv(out/'class_metrics.csv',pd.DataFrame(class_rows))
    write_csv(out/'metric_support.csv',pd.DataFrame(support_rows))
    write_csv(out/'paired_comparisons.csv',pd.DataFrame(contrasts))
    write_csv(out/'focal_comparisons.csv',pd.DataFrame(focals))
    write_npz(out/'bootstrap_episode_components.npz',numerators_by_seed=values,denominators=den,
              episode_ids=episodes.episode_id.to_numpy(dtype=str),episode_labels=labels,methods=np.array(METHODS),seeds=np.array(SEEDS))
    write_npz(out/'focal_bootstrap_samples.npz',differences=np.stack(fsamples,axis=1),bootstrap_seed=np.array(seed))
    write_json(out/'summary.json',dict(status='STEP66_EVIDENCE_TRANSITIONS_COMPLETE',methods=METHODS,head_seeds=SEEDS,
        focal_comparisons=focals,defined_interval_count=sum(s=='ok' for s in status),metric_specifications=len(specs),
        physical_recovery_evaluated=False,independent_validation=False,new_generation=False,
        interpretation='High gap alarm retention and low audit-quiet suffix alarm occupancy are separate objectives; no automatic winner.',
        warnings=['An observed-change run is not necessarily a new security incident.',
                  'No marked audit effect is not proof of normal operational state.',
                  'Gap correctness retains the EPISODE class; it is not instantaneous-active ground truth.',
                  'All comparisons are post-hoc and conditional on fixed trained heads/encoder/thresholds.',
                  'Unsupported subgroups remain NA. No seeds, conditions or classes are dropped to improve scores.']))
    return focals


def population_table(catalogs):
    episodes,events,gaps,tails=catalogs;rows=[]
    for li,lab in enumerate(LABELS):
        ep=episodes[episodes.label_id==li];ev=events[events.label_id==li];ga=gaps[gaps.label_id==li]
        ta=tails[(tails.label_id==li)&(tails.kind=='after_last_marked_activity')]
        rows.append(dict(label=lab,episodes=len(ep),windows=int(ep.windows.sum()),endpoints=int(ep.eligible_endpoints.sum()),
            observable_runs=int(ep.observable_runs.sum()),intervention_bursts=int(ep.intervention_runs.sum()),
            repeated_observable_runs=int(((ev.anchor=='observable')&(ev.event_number>1)).sum()),
            fully_followed_observable_onsets=int(((ev.anchor=='observable')&ev.fully_followed_10s.astype(bool)).sum()),
            interior_gaps=len(ga),fully_scored_gaps=int(ga.fully_scored.sum()),
            gap_scored_windows=int(ga.eligible_windows.sum()),
            terminal_audit_quiet_suffixes=int(ta.available.sum()),
            fully_followed_audit_quiet_suffixes=int(ta.fully_followed_first10.sum()),
            no_eligible_endpoint_episodes=int((ep.eligible_endpoints==0).sum())))
    return pd.DataFrame(rows)


def protocol_document(ctx,args,specs):
    return dict(version=VERSION,script_sha256=digest(__file__),dataset_id=DATASET_ID,
        source_step65_identity=ctx['protocol65']['identity'],source_step62_identity=STEP62_ID,
        source_input_sha256=ctx['verified'],methods=METHODS,head_seeds=SEEDS,validation_fpr_targets=TARGETS,
        training=False,new_inference=False,new_generator=False,threshold_recalibration=False,post_hoc=True,
        scope='Within-original-episode audit-transition analysis only; no new independent stream or physical recovery experiment.',
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),threads=args.threads),
        source_labels='Original episode/scenario labels retained; not an instantaneous compromised-state detector.',
        segmentation='Maximal one-second positive runs of intervention and observable audit flags; not independent attack incidents.',
        alarm='Strict Step65 score > locked per-method/per-head threshold; all predictions/thresholds unchanged.',
        new_alert='Known consecutive scored OFF->ON only; initial high after unavailable score is unknown, not a new alert.',
        response=dict(horizon_seconds=HORIZON,interval='[onset,onset+10s] inclusive; stop before next same-kind onset',
            attribution='No shared response window for two same-kind onsets; anchor types are separate analyses.',
            dense_bursts='Additional in-scored-run outcomes retain every positive run with >=1 eligible endpoint, regardless of fixed-horizon censoring. Variable duration, not a standardized 10s estimand.',
            denominator='Known within-record onset, next onset after deadline, record/scored deadline available; independent of early success.',
            warmup='First nine decisions unavailable, neither filled nor shifted; start at episode step0 is left-boundary uncertain.',
            censoring='All events retained in catalogue/details; incomplete horizons excluded from fixed-horizon ratios even if early success.'),
        gaps='All strictly interior zero-observable-change runs; not declared normal or active threat. Correct type uses retained episode class.',
        suffixes=dict(observable='After last observable-change flag within this finite record; active/persistent flags can remain.',
            audit_quiet='After last occurrence of ANY active, persistent or observable flag; no operational recovery claim.',
            no_prior_effect='No suffix is invented for episodes without a prior relevant positive flag.',
            width_windows=TAIL_WIDTH,interval='[suffix_start,suffix_start+10) one-second decisions (10 samples)',
            stable_nonalarm_decisions=QUIET_RUN,confirmation='End timestamp of first five consecutive known nonalarms; minimum delay4s.',
            capped_delay='10s if no confirmation in first10 windows; initially silent outputs included, not clearance conditional on detection.',
            full_followup='All first10 suffix windows must exist and be scored; no conditional rescue for early silence.'),
        normal='Only original NORMAL episodes; specific benign-lookalike event labels are not available to this evaluator.',
        statistics=dict(draws=args.bootstrap_draws,seed=args.bootstrap_seed,unit='Class-stratified paired whole episode',
            fixed='Heads, encoder, validation thresholds; no head or calibration resampling.',
            weights='Five attack classes equal; repeated bursts/long gaps contribute event/window counts within class.',
            interval='Percentile95%; suppressed for missing class, <2 eligible episodes in a class, or any undefined bootstrap draw.',
            focal_family=FOCAL_SIZE,focal='ActualHistory-MaxHold10 x 2 targets x gap correct-type fraction and audit-quiet suffix alarm fraction.',
            correction='Individual1-.05/4 percentile intervals for the four focal contrasts; other intervals exploratory.',
            caution='Retention and residual occupancy are separate objectives. No composite winner or post-test tuning.'),
        metric_specifications=specs)


def initialize_output(out,protocol):
    identity=obj_hash(protocol);document=dict(protocol,identity=identity);path=out/'protocol.json'
    if path.exists():
        require(read_json(path)==clean_json(document), 'Output protocol/input/code changed. Use the identical command or an explicit NEW --output-dir; do not overwrite the original run.')
    else:
        require({p.name for p in out.iterdir()} <= {'.step66.lock'}, 'Output contains unrelated files')
        write_json(path,document)
    return identity


def copy_verified(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():check_file(dst,digest(src))
    else:shutil.copyfile(src,dst)


def prepare_catalog_output(ctx,catalogs,specs,out,identity):
    lock_path=out/'catalog_lock.json'
    if lock_path.exists():
        record=read_json(lock_path)
        require(record['protocol_identity']==identity, 'Catalogue lock identity mismatch')
        for rel,sha in record['artifacts'].items():check_file(safe_child(out,rel),sha)
        return
    paths=[]
    for filename,frame in zip(('episode_catalog.csv','event_catalog.csv','gap_catalog.csv','tail_catalog.csv'),catalogs):
        write_csv(out/filename,frame);paths.append(filename)
    write_csv(out/'population_audit.csv',population_table(catalogs));paths.append('population_audit.csv')
    capability=dict(schema_columns=list(ctx['windows'].columns),physical_recovery_evaluation='NOT_SUPPORTED',
        instantaneous_security_state='NOT_AVAILABLE_IN_CURRENT_AUDIT',
        specific_benign_lookalike_annotations='NOT_AVAILABLE_IN_THIS_EVALUATOR',
        continuous_cross_episode_operation='NOT_EVALUATED',
        supported=['within-record intervention bursts','observable-change runs and intervening gaps',
                   'known new alerts versus carried alarms','alarm behavior in terminal audit-quiet suffixes'],
        restrictions=['All-zero audit flags do not certify recovery or benign operational state.',
                      'No episode stitching, reset at ground-truth onset, relabeling or recovery claim.',
                      'Original episode boundaries remain; repeated events in an episode are not independent samples.',
                      'Terminal suffix identification uses future audit within the record, for retrospective evaluation only.'],
        conclusion='Proceed with limited evidence-transition analysis. A separate recovery experiment needs independent state truth.')
    write_json(out/'truth_capability_audit.json',capability);paths.append('truth_capability_audit.json')
    write_json(out/'metric_specifications.json',specs);paths.append('metric_specifications.json')
    for src,name in [(ctx['run65']/'protocol.json','step65_protocol.json'),
                     (ctx['run65']/'run_manifest.json','step65_run_manifest.json'),
                     (ctx['run65']/'thresholds.json','step65_thresholds.json'),
                     (ctx['run65']/'threshold_lock.json','step65_threshold_lock.json'),
                     (ctx['run65']/'thresholds.csv','step65_thresholds.csv'),
                     (ctx['run65']/'endpoint_evidence_test.csv','endpoint_evidence_test.csv'),
                     (ctx['dataset_paths']['audit/test_window_observable_effects.csv'],'test_window_observable_effects.csv'),
                     (Path(__file__),Path(__file__).name)]:
        rel='source/'+name;copy_verified(src,out/rel);paths.append(rel)
    write_json(lock_path,dict(protocol_identity=identity,created_at_utc=now(),test_score_deserialization_started=False,
        artifacts={rel:digest(out/rel) for rel in paths}))


def output_files(out):
    return [p for p in sorted(out.rglob('*')) if p.is_file() and p.name!='run_manifest.json'
            and not p.name.startswith('.') and not p.name.endswith('.tmp') and p.suffix!='.zip']


def package_output(out,status,identity):
    files=output_files(out)
    write_json(out/'run_manifest.json',dict(status=status,version=VERSION,protocol_identity=identity,completed_at_utc=now(),
        training=False,new_inference=False,threshold_recalibration=False,physical_recovery_evaluated=False,
        output_sha256={p.relative_to(out).as_posix():digest(p) for p in files}))
    suffix='_review.zip' if status=='STEP66_EVIDENCE_TRANSITIONS_COMPLETE' else '_diagnostic_review.zip'
    archive=out.parent/(out.name+suffix);tmp=archive.with_name(archive.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[out/'run_manifest.json']:z.write(p,p.relative_to(out).as_posix())
    os.replace(tmp,archive)
    return archive


def verify_finished(out,identity):
    path=out/'run_manifest.json'
    if not path.exists():return False
    r=read_json(path)
    if r.get('status')!='STEP66_EVIDENCE_TRANSITIONS_COMPLETE':return False
    require(r['protocol_identity']==identity,'Completed run identity differs')
    for rel,sha in r['output_sha256'].items():check_file(safe_child(out,rel),sha)
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--step65-run',type=Path,help='Completed original Step65 directory, not ZIP')
    parser.add_argument('--dataset-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--bootstrap-draws',type=int,default=20000)
    parser.add_argument('--bootstrap-seed',type=int,default=2026100866)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:return self_test()
    managed=False;identity=None;out=None
    try:
        require(1<=args.threads<=32 and args.bootstrap_draws>=2000 and 0<=args.bootstrap_seed<2**63,
                'Require threads1..32, bootstrap draws>=2000, valid nonnegative bootstrap seed')
        root=args.project_root.resolve();run=resolve(root,args.step65_run or DEFAULT65)
        data=resolve(root,args.dataset_dir or DEFAULT_DATA);out=resolve(root,args.output_dir or DEFAULT_OUT)
        require(out!=root,'Output must not be the project root')
        for source in (run,data):
            require(out!=source and source not in out.parents and out not in source.parents, 'Output overlaps input directory')
        print('Step66: immutable Step65 scores -> within-episode evidence transitions. No training or new inference.',flush=True)
        print('[1/4] Verify Step65 artifacts, locked thresholds and original test audit ...',flush=True)
        ctx=load_step65(run);load_dataset(ctx,data)
        specs=metric_specs();catalogs=make_catalogs(ctx['windows'])
        protocol=protocol_document(ctx,args,specs)
        with exclusive_lock(out):
            identity=initialize_output(out,protocol);managed=True
            if verify_finished(out,identity):
                archive=out.parent/(out.name+'_review.zip')
                if not archive.exists():archive=package_output(out,'STEP66_EVIDENCE_TRANSITIONS_COMPLETE',identity)
                print('Verified completed Step66; no recomputation.\nReview ZIP: '+str(archive),flush=True)
                return 0
            try:
                prepare_catalog_output(ctx,catalogs,specs,out,identity)
                population=population_table(catalogs)
                write_json(out/'preflight.json',dict(status='STEP66_PREFLIGHT_PASS',protocol_identity=identity,
                    test_windows=len(ctx['windows']),test_endpoints=len(ctx['endpoints']),episodes=len(catalogs[0]),
                    event_runs=len(catalogs[1]),interior_gaps=len(catalogs[2]),
                    fully_followed_audit_quiet_suffixes=int(population.fully_followed_audit_quiet_suffixes.sum()),
                    physical_recovery_evaluation='NOT_SUPPORTED',scope='Existing within-episode evidence transitions only'))
                print('[2/4] Catalogue locked: '+str(len(catalogs[1]))+' runs, '+str(len(catalogs[2]))+
                      ' interior gaps. Physical recovery truth is NOT available; no recovery metric will be invented.',flush=True)
                if args.preflight_only:
                    print('STEP66_PREFLIGHT_PASS\nOutput: '+str(out),flush=True);return 0
                print('[3/4] Apply existing thresholds without resets at event onsets; save each seed ...',flush=True)
                components=[];den=None
                for si in range(len(SEEDS)):
                    nn,dd=evaluate_seed(ctx,catalogs,si,specs,out,identity)
                    if den is None:den=dd
                    else:require(np.array_equal(den,dd),'Head-dependent denominators detected')
                    components.append(nn)
                print('[4/4] Paired whole-episode bootstrap and source recheck ...',flush=True)
                try:
                    from threadpoolctl import threadpool_limits
                    limit=threadpool_limits(limits=args.threads)
                except ImportError:limit=nullcontext()
                with limit:summarize(components,den,catalogs[0],specs,out,args.bootstrap_draws,args.bootstrap_seed)
                for path,sha in ctx['verified'].items():check_file(path,sha)
                require(digest(__file__)==protocol['script_sha256'],'Step66 source changed during execution')
                if (out/'failure.json').exists():(out/'failure.json').unlink()
                archive=package_output(out,'STEP66_EVIDENCE_TRANSITIONS_COMPLETE',identity)
                print('STEP66_EVIDENCE_TRANSITIONS_COMPLETE\nResults: '+str(out)+'\nReview ZIP: '+str(archive),flush=True)
                return 0
            except (Exception,KeyboardInterrupt) as exc:
                write_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                    next_action='Restore corrupted inputs if needed, then rerun the identical command. Verified completed seeds are reused.'))
                archive=package_output(out,'STEP66_INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'STEP66_FAILED',identity)
                print('STOPPED: '+repr(exc)+'\nDiagnostic ZIP: '+str(archive),file=sys.stderr)
                return 130 if isinstance(exc,KeyboardInterrupt) else 1
    except (Exception,KeyboardInterrupt) as exc:
        # Failure before an owned protocol is established never edits another run.
        print('STOPPED: '+repr(exc),file=sys.stderr)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1


def self_test():
    """Fast synthetic edge cases; no project input or model package is opened."""
    assert positive_runs([1,1,0,1])==[(0,1),(3,3)]
    q=np.arange(40)>=9;a=np.ones(40,bool)
    r=event_response(a,q,20,None);assert r['any_alarm_observed'] and not r['new_alarm_observed']
    a=np.zeros(40,bool);a[21:]=True
    r=event_response(a,q,20,None);assert r['new_alarm_delay_seconds']==1
    a[:]=False;a[9:]=True
    r=event_response(a,q,3,None);assert r['any_alarm_delay_seconds']==6 and not r['new_alarm_observed']
    a[:]=False;a[12]=True
    r=event_response(a,q,10,18);assert r['any_alarm_observed'] and not r['fully_followed']
    assert event_response(a,q,0,None)['left_boundary_onset']
    a[:]=False;a[20]=True
    assert event_response(a,q,10,None)['new_alarm_delay_seconds']==10
    a[:]=False;r=tail_response(a,q,10)
    assert r['silence_confirmation_delay_seconds']==4 and r['alarm_fraction_first10']==0
    a[:]=True;assert tail_response(a,q,10)['capped_confirmation_delay_seconds']==10
    assert not tail_response(a[:17],q[:17],10)['fully_followed']
    assert not tail_response(a,q,4)['fully_followed']
    den=np.ones((1,4));num=np.array([[[.8,.5]]*4]);labels=np.array([1,1,2,2])
    p,s,status=bootstrap_ratios(num,den,labels,[(1,2)],100,66)
    assert status==['ok'] and np.allclose(s[:,0,0]-s[:,0,1],.3)
    den[0,2:]=0;num[0,2:]=0
    assert bootstrap_ratios(num,den,labels,[(1,2)],100,66)[2]==['missing_target_class']
    try:safe_child(Path.cwd(),'../escape')
    except ValueError:pass
    else:raise AssertionError('Unsafe path accepted')
    assert sum(focal_spec(s) for s in metric_specs())==FOCAL_SIZE
    print('STEP66_SELF_TEST_PASS: synthetic segmentation, alarm attribution, censoring, silence and bootstrap checks.')
    return 0


if __name__=='__main__':
    sys.exit(main())
