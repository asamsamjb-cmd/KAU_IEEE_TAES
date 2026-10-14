#!/usr/bin/env python3
"""benchmark_audit.py — freeze-gate audit for a regenerated TAES benchmark.

Run on CPU right after Step 7 produces the splits:

    python benchmark_audit.py --splits dataset/splits --out outputs/benchmark_audit
    python benchmark_audit.py --splits dataset/splits --missing-token MISSING

Inputs : dataset/splits/{train,val,test}_observable.csv (METADATA + 22 observable fields)
Outputs: <out>/audit_summary.json, per-check CSVs, and a console gate (PASS / WARN / FAIL).

The script never modifies inputs. It reads labels only to measure how much
label information a field or field group carries; it is a leakage/shortcut
audit, not a model. No GPU, no training beyond small sklearn logistic probes.

Checks
  A  schema, NaN, forbidden raw_ columns, split disjointness, label constancy
  B  missing-value convention (designated token present? rate per field)
  C  context fields: held-out probes, within-episode constancy, per-class divergence
  D  single-field probes (majority mapping, train -> test) for all 22 fields
  E  field-group probes (logistic, train -> test): context / message / receiver / trace / all
  F  row-signature duplicates across labels; unseen-signature fraction; lookup-table baseline
  G  cross-label identical episode trajectories
  H  episode-length distribution per class and length-only probe
  I  temporal sanity (1-second spacing, contiguous steps)
  J  class-exclusive vocabulary tokens per field
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
warnings.filterwarnings('ignore')
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

FIELDS = (
    'observation_stage', 'path_type', 'direction', 'source_role', 'destination_role',
    'protocol_context', 'operational_phase',
    'message_type', 'service_context', 'command_intent', 'payload_or_parameter_context',
    'sequence_freshness_status', 'control_security_state',
    'signal_quality_state', 'tracking_stability', 'navigation_consistency', 'timing_consistency',
    'decoder_status', 'command_execution_result', 'retry_timeout_state', 'session_state',
    'state_transition_consistency',
)
GROUPS = {
    'context': FIELDS[:7],
    'message_control': FIELDS[7:13],
    'receiver': FIELDS[13:17],
    'trace': FIELDS[17:],
}
METADATA = ('scenario_id', 'episode_id', 'episode_type', 'episode_step', 'window_id', 'timestamp', 'label')
LABELS = ('NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING', 'UNAUTHORIZED_TELECOMMAND_INJECTION',
          'TELECOMMAND_REPLAY', 'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
MISSING_LIKE = {'', 'UNK', 'MISSING', 'MISSING_FIELD', 'N/A', 'NA', 'NONE', 'NULL', 'UNAVAILABLE',
                'NOT_OBSERVED', 'UNOBSERVED', 'nan', 'NaN'}

# gate thresholds (edit here if the design pre-specifies other values)
CHANCE = 1.0 / 6
CTX_PROBE_WARN = CHANCE + 0.03      # context-only macro-F1 above this -> WARN
CTX_PROBE_FAIL = CHANCE + 0.10      # -> FAIL
CTX_CRAMERS_V_WARN = 0.05           # per-class divergence of a context field
SINGLE_FIELD_STRONG = 0.50          # single evidence field macro-F1 above this is reported
LOOKUP_WARN = 0.90                  # train->test lookup-table macro-F1 above this -> WARN
LENGTH_PROBE_WARN = CHANCE + 0.05


# ----------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = 6) -> float:
    f = []
    for c in range(k):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def majority_probe(train_key: pd.Series, y_train: np.ndarray, test_key: pd.Series, y_test: np.ndarray,
                   default: int) -> float:
    """Map each key value to its majority training label; unseen keys -> default."""
    table = pd.crosstab(train_key, y_train)
    mapping = table.idxmax(axis=1).to_dict()
    pred = test_key.map(mapping).fillna(default).astype(int).to_numpy()
    return macro_f1(y_test, pred)


def logistic_probe(train_df: pd.DataFrame, y_train: np.ndarray, test_df: pd.DataFrame, y_test: np.ndarray,
                   cols, max_rows: int = 60000, seed: int = 0) -> float | None:
    if not HAVE_SKLEARN:
        return None
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_df))
    if len(idx) > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    enc = OneHotEncoder(handle_unknown='ignore')
    xtr = enc.fit_transform(train_df.iloc[idx][list(cols)].astype(str))
    xte = enc.transform(test_df[list(cols)].astype(str))
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(xtr, y_train[idx])
    return macro_f1(y_test, clf.predict(xte))


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def load_split(path: Path) -> pd.DataFrame:
    # Step 7 writes UTF-8 with BOM (utf-8-sig); read accordingly and strip any stray BOM from headers.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
    return df


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='dataset/splits')
    ap.add_argument('--out', default='outputs/benchmark_audit')
    ap.add_argument('--missing-token', default=None,
                    help='Designated missing-value token of the new design (e.g. MISSING). '
                         'If given, check B reports its presence per field and per split.')
    ap.add_argument('--context-episode-constant', action='store_true',
                    help='Design expects context fields to be constant within an episode; '
                         'check C then fails if they are not.')
    ap.add_argument('--max-probe-rows', type=int, default=60000)
    args = ap.parse_args()

    split_dir = Path(args.splits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gate: list[tuple[str, str, str]] = []   # (check, status, message)

    def add(check: str, status: str, message: str) -> None:
        gate.append((check, status, message))
        print(f'[{status:4s}] {check}: {message}', flush=True)

    frames = {}
    for split in ('train', 'val', 'test'):
        p = split_dir / f'{split}_observable.csv'
        if not p.is_file():
            add('A.files', 'FAIL', f'missing {p}')
            return finish(out, gate, {})
        frames[split] = load_split(p)

    summary: dict = {'rows': {}, 'episodes': {}}
    label_to_id = {l: i for i, l in enumerate(LABELS)}

    # ---------------- A. schema / splits -----------------------------------
    for split, df in frames.items():
        cols = list(df.columns)
        forbidden = [c for c in cols if c.startswith('raw_') or c in ('phase', 'severity', 'case_type', 'label_id', 'text')]
        if forbidden:
            add('A.schema', 'FAIL', f'{split}: forbidden columns {forbidden}')
        missing_cols = [c for c in METADATA + FIELDS if c not in cols]
        if missing_cols:
            add('A.schema', 'FAIL', f'{split}: missing columns {missing_cols}')
            return finish(out, gate, summary)
        extra = [c for c in cols if c not in METADATA + FIELDS]
        if extra:
            add('A.schema', 'WARN', f'{split}: extra columns {extra} (must not reach serialization)')
        if (df[list(FIELDS)] == '').any().any() and args.missing_token != '':
            n_empty = int((df[list(FIELDS)] == '').sum().sum())
            add('A.nan', 'WARN', f'{split}: {n_empty} empty-string cells in observable fields')
        unknown_labels = set(df['label']) - set(LABELS)
        if unknown_labels:
            add('A.labels', 'FAIL', f'{split}: unknown labels {sorted(unknown_labels)}')
        summary['rows'][split] = int(len(df))
        summary['episodes'][split] = int(df['episode_id'].nunique())
        # episode-level label constancy
        n_lab = df.groupby('episode_id')['label'].nunique()
        if (n_lab > 1).any():
            add('A.episode_label', 'WARN', f'{split}: {int((n_lab > 1).sum())} episodes with >1 label (transitions)')
    ep_sets = {s: set(df['episode_id']) for s, df in frames.items()}
    leak = (ep_sets['train'] & ep_sets['val']) | (ep_sets['train'] & ep_sets['test']) | (ep_sets['val'] & ep_sets['test'])
    add('A.split', 'FAIL' if leak else 'PASS', f'episodes shared across splits: {len(leak)}')
    per_class = {s: frames[s].groupby('label').size().to_dict() for s in frames}
    summary['rows_per_class'] = per_class
    add('A.counts', 'PASS', f"rows train/val/test = {summary['rows']['train']}/{summary['rows']['val']}/{summary['rows']['test']}; "
                            f"episodes {summary['episodes']['train']}/{summary['episodes']['val']}/{summary['episodes']['test']}")

    train, val, test = frames['train'], frames['val'], frames['test']
    y_tr = train['label'].map(label_to_id).to_numpy()
    y_te = test['label'].map(label_to_id).to_numpy()
    y_va = val['label'].map(label_to_id).to_numpy()
    default_label = int(np.bincount(y_tr).argmax())

    # ---------------- B. missing-value convention --------------------------
    rows_b = []
    for f in FIELDS:
        rec = {'field': f}
        for split, df in frames.items():
            vals = df[f]
            rec[f'{split}_missing_like_rate'] = float(vals.isin(MISSING_LIKE).mean())
            if args.missing_token is not None:
                rec[f'{split}_token_rate'] = float((vals == args.missing_token).mean())
        rows_b.append(rec)
    b = pd.DataFrame(rows_b)
    b.to_csv(out / 'B_missing_convention.csv', index=False)
    if args.missing_token is not None:
        tr_rate = b['train_token_rate']
        if (tr_rate == 0).all():
            add('B.missing', 'FAIL', f"designated token '{args.missing_token}' never appears in train; "
                                     'the encoder would never see it (same situation as Step 47 UNK)')
        else:
            add('B.missing', 'PASS', f"designated token present in train: fields with rate>0 = {int((tr_rate > 0).sum())}/22, "
                                     f"mean rate {tr_rate.mean():.4f}")
    ml = b['train_missing_like_rate']
    add('B.missing_like', 'PASS' if ml.max() == 0 or args.missing_token else 'WARN',
        f'missing-like tokens in train: max field rate {ml.max():.4f} (values in {sorted(MISSING_LIKE)[:6]}...)')

    # ---------------- C. context fields ------------------------------------
    ctx = list(GROUPS['context'])
    key_tr = train[ctx].astype(str).agg('|'.join, axis=1)
    key_te = test[ctx].astype(str).agg('|'.join, axis=1)
    ctx_major = majority_probe(key_tr, y_tr, key_te, y_te, default_label)
    ctx_logit = logistic_probe(train, y_tr, test, y_te, ctx, args.max_probe_rows)
    msg = f'context-only macro-F1 (train->test): majority {ctx_major:.3f}' + (f', logistic {ctx_logit:.3f}' if ctx_logit is not None else '') + f'; chance {CHANCE:.3f}'
    worst = max(ctx_major, ctx_logit or 0)
    add('C.context_probe', 'FAIL' if worst > CTX_PROBE_FAIL else 'WARN' if worst > CTX_PROBE_WARN else 'PASS', msg)
    rows_c = []
    for f in ctx:
        const = train.groupby('episode_id')[f].nunique().eq(1).mean()
        v = cramers_v(train[f], train['label'])
        rows_c.append({'field': f, 'episode_constant_fraction': float(const), 'cramers_v_vs_label': v,
                       'n_values': int(train[f].nunique())})
    c = pd.DataFrame(rows_c)
    c.to_csv(out / 'C_context_fields.csv', index=False)
    if args.context_episode_constant:
        bad = c[c['episode_constant_fraction'] < 0.999]
        add('C.context_constancy', 'FAIL' if len(bad) else 'PASS',
            f'context fields constant within episode: {len(ctx) - len(bad)}/{len(ctx)}')
    else:
        add('C.context_constancy', 'PASS', 'episode-constant fraction per context field: ' +
            ', '.join(f"{r.field}={r.episode_constant_fraction:.2f}" for r in c.itertuples()))
    vmax = c['cramers_v_vs_label'].max()
    add('C.context_divergence', 'WARN' if vmax > CTX_CRAMERS_V_WARN else 'PASS',
        f"max Cramer's V(context field, label) = {vmax:.3f} ({c.loc[c['cramers_v_vs_label'].idxmax(), 'field']})")

    # ---------------- D. single-field probes --------------------------------
    rows_d = []
    for f in FIELDS:
        s = majority_probe(train[f], y_tr, test[f], y_te, default_label)
        rows_d.append({'field': f, 'group': next(g for g, fs in GROUPS.items() if f in fs),
                       'single_field_majority_macro_f1': s, 'cramers_v_vs_label': cramers_v(train[f], train['label']),
                       'n_values_train': int(train[f].nunique())})
    d = pd.DataFrame(rows_d).sort_values('single_field_majority_macro_f1', ascending=False)
    d.to_csv(out / 'D_single_field_probes.csv', index=False)
    strong = d[d['single_field_majority_macro_f1'] > SINGLE_FIELD_STRONG]
    add('D.single_field', 'WARN' if len(strong) else 'PASS',
        'fields with single-field macro-F1 > %.2f: ' % SINGLE_FIELD_STRONG +
        (', '.join(f"{r.field}={r.single_field_majority_macro_f1:.2f}" for r in strong.itertuples()) or 'none') +
        f"; best = {d.iloc[0]['field']} {d.iloc[0]['single_field_majority_macro_f1']:.3f}")

    # ---------------- E. field-group probes ---------------------------------
    rows_e = []
    for g, fs in list(GROUPS.items()) + [('all_22', FIELDS)]:
        k_tr = train[list(fs)].astype(str).agg('|'.join, axis=1)
        k_te = test[list(fs)].astype(str).agg('|'.join, axis=1)
        maj = majority_probe(k_tr, y_tr, k_te, y_te, default_label)
        lg = logistic_probe(train, y_tr, test, y_te, fs, args.max_probe_rows)
        rows_e.append({'group': g, 'n_fields': len(fs), 'majority_macro_f1': maj,
                       'logistic_macro_f1': lg if lg is not None else np.nan})
    e = pd.DataFrame(rows_e)
    e.to_csv(out / 'E_group_probes.csv', index=False)
    add('E.groups', 'PASS', 'single-window logistic macro-F1 by group: ' +
        ', '.join(f"{r.group}={r.logistic_macro_f1:.3f}" if not np.isnan(r.logistic_macro_f1) else f"{r.group}=na" for r in e.itertuples()))

    # ---------------- F. row signatures / lookup ----------------------------
    sig_tr = train[list(FIELDS)].astype(str).agg('|'.join, axis=1)
    sig_te = test[list(FIELDS)].astype(str).agg('|'.join, axis=1)
    sig_label = pd.crosstab(sig_tr, train['label'])
    multi = int((sig_label.gt(0).sum(axis=1) > 1).sum())
    unseen = float((~sig_te.isin(set(sig_tr))).mean())
    lookup = majority_probe(sig_tr, y_tr, sig_te, y_te, default_label)
    add('F.signatures', 'WARN' if lookup > LOOKUP_WARN else 'PASS',
        f'22-field signatures shared across >1 label (train): {multi}; test signatures unseen in train: {unseen:.3f}; '
        f'lookup-table baseline macro-F1: {lookup:.3f}')
    summary['F'] = {'multi_label_signatures': multi, 'unseen_test_signature_fraction': unseen, 'lookup_macro_f1': lookup}

    # ---------------- G. identical episode trajectories across labels ------
    traj = defaultdict(set)
    for split, df in frames.items():
        for eid, g in df.sort_values(['episode_id', 'episode_step']).groupby('episode_id'):
            h = hash(tuple(g[list(FIELDS)].astype(str).agg('|'.join, axis=1)))
            traj[h].add(g['label'].iloc[0])
    collisions = sum(1 for v in traj.values() if len(v) > 1)
    add('G.trajectories', 'FAIL' if collisions else 'PASS', f'identical full-episode trajectories across labels: {collisions}')

    # ---------------- H. episode lengths ------------------------------------
    lengths = train.groupby(['episode_id', 'label']).size().reset_index(name='n')
    h = lengths.groupby('label')['n'].agg(['mean', 'std', 'min', 'max', 'count'])
    h.to_csv(out / 'H_episode_lengths.csv')
    lengths_te = test.groupby(['episode_id', 'label']).size().reset_index(name='n')
    bucket_tr = pd.cut(lengths['n'], bins=[0, 20, 40, 60, 80, 100, 150, 10**6]).astype(str)
    bucket_te = pd.cut(lengths_te['n'], bins=[0, 20, 40, 60, 80, 100, 150, 10**6]).astype(str)
    lp = majority_probe(bucket_tr, lengths['label'].map(label_to_id).to_numpy(), bucket_te,
                        lengths_te['label'].map(label_to_id).to_numpy(), default_label)
    add('H.length_probe', 'WARN' if lp > LENGTH_PROBE_WARN else 'PASS',
        f'episode-length-only probe macro-F1 (episode level): {lp:.3f}; mean length per class: ' +
        ', '.join(f"{l[:8]}={m:.1f}" for l, m in h['mean'].items()))

    # ---------------- I. temporal sanity ------------------------------------
    bad_steps = 0
    for split, df in frames.items():
        st = df['episode_step'].astype(int)
        g = df.assign(st=st).sort_values(['episode_id', 'st']).groupby('episode_id')['st']
        contiguous = g.apply(lambda s: bool((np.diff(s.to_numpy()) == 1).all()) if len(s) > 1 else True)
        bad_steps += int((~contiguous).sum())
    add('I.temporal', 'FAIL' if bad_steps else 'PASS', f'episodes with non-contiguous steps: {bad_steps}')

    # ---------------- J. class-exclusive tokens ------------------------------
    rows_j = []
    for f in FIELDS:
        tab = pd.crosstab(train[f], train['label'])
        exclusive = tab[(tab.gt(0).sum(axis=1) == 1)]
        for val, row in exclusive.iterrows():
            lab = row.idxmax()
            rows_j.append({'field': f, 'value': val, 'only_in_label': lab, 'train_count': int(row.max()),
                           'share_of_label_rows': float(row.max() / (train['label'] == lab).sum())})
    j = pd.DataFrame(rows_j).sort_values('train_count', ascending=False) if rows_j else pd.DataFrame(
        columns=['field', 'value', 'only_in_label', 'train_count', 'share_of_label_rows'])
    j.to_csv(out / 'J_class_exclusive_tokens.csv', index=False)
    big = j[j['share_of_label_rows'] > 0.05] if len(j) else j
    add('J.exclusive_tokens', 'WARN' if len(big) else 'PASS',
        f'class-exclusive field values in train: {len(j)} (covering >5% of a class: {len(big)}); '
        + (', '.join(f"{r.field}={r.value}->{r.only_in_label[:8]} ({r.share_of_label_rows:.2f})" for r in big.head(6).itertuples()) if len(big) else 'none'))

    summary['gate'] = [{'check': c_, 'status': s_, 'message': m_} for c_, s_, m_ in gate]
    return finish(out, gate, summary)


def finish(out: Path, gate, summary) -> int:
    status = 'FAIL' if any(s == 'FAIL' for _, s, _ in gate) else 'WARN' if any(s == 'WARN' for _, s, _ in gate) else 'PASS'
    summary['overall'] = status
    (out / 'audit_summary.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print(f'\nOVERALL: {status}   (report: {out}/audit_summary.json)')
    print('FAIL = do not freeze; WARN = design decision needed and must be documented; PASS = no issue found by this audit.')
    return 0 if status != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())
