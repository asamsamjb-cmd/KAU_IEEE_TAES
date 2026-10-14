#!/usr/bin/env python3
"""fingerprint_audit.py — generator-fingerprint acceptance test for the TAES benchmark.

Question answered: can windows (or 10-window sequences) that contain NO observed
attack evidence be told apart by class?  If yes, the generators leave a class
signature in benign windows, and part of any detector's score is generator
identification rather than attack detection.

Usage (CPU, a few minutes):
    python fingerprint_audit.py --dataset-dir dataset --out outputs/fingerprint_audit
    python fingerprint_audit.py --dataset-dir dataset --out outputs/fingerprint_audit \
        --l10-predictions outputs/revised_makers_v1/tst_training_.../L10/predictions/test_predictions_L10.csv

Inputs: the six *_with_raw.csv maker outputs (raw audit flags define "observed evidence").
Outputs: <out>/fingerprint_summary.json, per-class tables, token asymmetry list, and a gate.

Acceptance (shared benign substrate): with class-balanced logistic probes,
sequence-level macro-F1 on evidence-free 10-window runs <= 0.25 and NORMAL
recall <= 0.35 (chance is ~0.17 for both); with --l10-predictions,
evidence-free attack endpoints are predicted NORMAL at a rate comparable to
NORMAL's own evidence-free rate (command-related classes at 0% is a failure).
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

FIELDS = ('observation_stage', 'path_type', 'direction', 'source_role', 'destination_role', 'protocol_context',
          'operational_phase', 'message_type', 'service_context', 'command_intent', 'payload_or_parameter_context',
          'sequence_freshness_status', 'control_security_state', 'signal_quality_state', 'tracking_stability',
          'navigation_consistency', 'timing_consistency', 'decoder_status', 'command_execution_result',
          'retry_timeout_state', 'session_state', 'state_transition_consistency')
CLS = ['NORMAL', 'RF', 'GNSS', 'INJ', 'REPLAY', 'GND']
FILES = {'NORMAL': 'normal_observable_dataset_with_raw.csv',
         'RF': 'rf_link_jamming_observable_dataset_with_raw.csv',
         'GNSS': 'gnss_spoofing_observable_dataset_with_raw.csv',
         'INJ': 'unauthorized_telecommand_injection_observable_dataset_with_raw.csv',
         'REPLAY': 'telecommand_replay_observable_dataset_with_raw.csv',
         'GND': 'ground_segment_ttc_control_path_compromise_observable_dataset_with_raw.csv'}
LABEL_ID = {'NORMAL': 0, 'RF_LINK_JAMMING': 1, 'GNSS_SPOOFING': 2, 'UNAUTHORIZED_TELECOMMAND_INJECTION': 3,
            'TELECOMMAND_REPLAY': 4, 'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE': 5}
SEQ_F1_MAX, NORMAL_RECALL_MAX = 0.25, 0.35


REQUIRED_FLAGS = {'NORMAL': [], 'RF': ['raw_attack_active', 'raw_observation_on_target'],
                  'GNSS': ['raw_effect_applied', 'raw_residual_effect_active'], 'INJ': ['raw_observed_injection'],
                  'REPLAY': ['raw_observed_replay'], 'GND': ['raw_observed_compromise']}


def evidence(df: pd.DataFrame, k: str) -> np.ndarray:
    missing = [c for c in REQUIRED_FLAGS[k] if c not in df.columns]
    if missing:
        sys.exit(f"[FAIL] {k}: raw evidence flag column(s) missing: {missing}. "
                 "The makers must keep these raw audit column names (see shared_substrate_spec section 2-5).")
    g = lambda c: (pd.to_numeric(df[c], errors='coerce').fillna(0).to_numpy() > 0)
    if k == 'NORMAL': return np.zeros(len(df), bool)
    if k == 'RF': return g('raw_attack_active') & g('raw_observation_on_target')
    if k == 'GNSS': return g('raw_effect_applied') | g('raw_residual_effect_active')
    if k == 'INJ': return g('raw_observed_injection')
    if k == 'REPLAY': return g('raw_observed_replay')
    return g('raw_observed_compromise')


def macro_f1(y, p, k=6):
    f = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), np.array(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default='dataset'); ap.add_argument('--out', default='outputs/fingerprint_audit')
    ap.add_argument('--l10-predictions', default=None); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); summary = {}
    parts = []
    for k, fn in FILES.items():
        path = Path(a.dataset_dir) / fn
        if not path.is_file():
            sys.exit(f"[FAIL] missing maker output: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
        miss = [c for c in FIELDS + ('episode_id', 'episode_step') if c not in df.columns]
        if miss:
            sys.exit(f"[FAIL] {k}: columns missing {miss}")
        sub = df[list(FIELDS) + ['episode_id', 'episode_step']].copy()
        sub['cls'] = k; sub['y'] = CLS.index(k); sub['evidence'] = evidence(df, k); sub['step'] = sub['episode_step'].astype(int)
        parts.append(sub)
        summary[f'evidence_rows_{k}'] = int(sub['evidence'].sum())
        print(f"[load   ] {k:7s} rows={len(df):6d} observed-evidence rows={int(sub['evidence'].sum()):6d} ({sub['evidence'].mean():.3f})")
        if k != 'NORMAL' and sub['evidence'].sum() == 0:
            sys.exit(f"[FAIL] {k}: no observed-evidence rows; the raw flags are not populated as expected.")
    w = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(a.seed); eps = np.array(sorted(w['episode_id'].unique())); rng.shuffle(eps)
    test_eps = set(eps[:int(0.3 * len(eps))]); w['test'] = w['episode_id'].isin(test_eps)
    ef = w[~w.evidence]
    enc = OneHotEncoder(handle_unknown='ignore'); Xtr = enc.fit_transform(ef[~ef.test][list(FIELDS)]); Xte = enc.transform(ef[ef.test][list(FIELDS)])
    ytr = ef[~ef.test]['y'].to_numpy(); yte = ef[ef.test]['y'].to_numpy()
    p = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr).predict(Xte); m, per = macro_f1(yte, p)
    summary['window_level'] = {'n_test': int(len(yte)), 'accuracy': float(np.mean(p == yte)), 'macro_f1': m, 'per_class_f1': dict(zip(CLS, per.round(3).tolist()))}
    print(f"[window ] evidence-free windows: acc={np.mean(p == yte):.3f} macro-F1={m:.3f} per-class={summary['window_level']['per_class_f1']}")
    rows, ys, tests = [], [], []
    for eid, g in ef.sort_values(['episode_id', 'step']).groupby('episode_id'):
        steps = g['step'].to_numpy(); X = enc.transform(g[list(FIELDS)]); yv = int(g['y'].iloc[0]); t = bool(g['test'].iloc[0]); i = 0
        while i + 10 <= len(steps):
            if steps[i + 9] - steps[i] == 9: rows.append(np.asarray(X[i:i + 10].sum(0)).ravel()); ys.append(yv); tests.append(t); i += 5
            else: i += 1
    Xs = sparse.csr_matrix(np.vstack(rows)); ys = np.array(ys); tests = np.array(tests)
    p2 = LogisticRegression(max_iter=3000, class_weight='balanced').fit(Xs[~tests], ys[~tests]).predict(Xs[tests]); m2, per2 = macro_f1(ys[tests], p2)
    rec_norm = float(np.mean(p2[ys[tests] == 0] == 0)) if (ys[tests] == 0).any() else float('nan')
    bal_acc = float(np.mean([np.mean(p2[ys[tests] == c] == c) for c in range(6) if (ys[tests] == c).any()]))
    summary['sequence_level'] = {'n_test': int(tests.sum()), 'accuracy': float(np.mean(p2 == ys[tests])), 'balanced_accuracy': bal_acc, 'macro_f1': m2, 'normal_recall': rec_norm, 'per_class_f1': dict(zip(CLS, per2.round(3).tolist()))}
    cm = pd.crosstab(pd.Series(ys[tests]).map(dict(enumerate(CLS))), pd.Series(p2).map(dict(enumerate(CLS))), normalize='index') * 100
    cm.to_csv(out / 'sequence_level_confusion.csv')
    print(f"[seq    ] 10 evidence-free windows (bag of tokens, class-balanced): acc={np.mean(p2 == ys[tests]):.3f} balanced-acc={bal_acc:.3f} (chance 0.167) macro-F1={m2:.3f} NORMAL recall={rec_norm:.3f}")
    print(cm.round(1).to_string())
    asym = []
    for f in FIELDS[7:]:
        tab = pd.crosstab(ef[f], ef['cls']).reindex(columns=CLS, fill_value=0)
        for v, r in tab.iterrows():
            present = [c for c in CLS if r[c] > 0]
            if r.sum() >= 50 and len(present) <= 3: asym.append({'field': f, 'value': v, 'n': int(r.sum()), 'classes': {c: int(r[c]) for c in present}})
    asym_df = pd.DataFrame(asym, columns=['field', 'value', 'n', 'classes'])
    if len(asym_df):
        asym_df = asym_df.sort_values('n', ascending=False)
    asym_df.to_csv(out / 'evidence_free_token_asymmetry.csv', index=False)
    summary['token_asymmetries'] = len(asym)
    print(f"[tokens ] class-exclusive/near-exclusive tokens in evidence-free windows: {len(asym)}")
    if a.l10_predictions:
        w2 = w.sort_values(['episode_id', 'step']); w2['cum'] = w2.groupby('episode_id')['evidence'].cumsum()
        cum = {(e, s): c for e, s, c in zip(w2['episode_id'].to_numpy(), w2['step'].to_numpy(), w2['cum'].to_numpy())}
        pr = pd.read_csv(a.l10_predictions, encoding='utf-8-sig')
        need = ['episode_id', 'sequence_end_step', 'sequence_start_step', 'true_label_id', 'pred_label_id']
        miss = [c for c in need if c not in pr.columns]
        if miss:
            sys.exit(f"[FAIL] prediction file lacks columns {miss}")
        known = set(w['episode_id'].unique()); found = pr['episode_id'].isin(known).mean()
        if found < 0.99:
            sys.exit(f"[FAIL] only {found:.1%} of prediction episode_ids exist in the maker outputs; episode ids were renamed or the files do not match.")
        e = pr['episode_id'].to_numpy(); s = pr['sequence_end_step'].astype(int).to_numpy(); s0 = pr['sequence_start_step'].astype(int).to_numpy()
        ctx = np.array([cum.get((ee, ss), 0) - (cum.get((ee, a0 - 1), 0) if a0 > 0 else 0) > 0 for ee, ss, a0 in zip(e, s, s0)])
        y = pr['true_label_id'].to_numpy(); yh = pr['pred_label_id'].to_numpy(); tab = {}
        for c, name in enumerate(CLS):
            sub = (y == c); efm = sub & ~ctx; evm = sub & ctx
            tab[name] = {'n': int(sub.sum()), 'evidence_free_fraction': float(efm.sum() / sub.sum()), 'acc_evidence_free': float(np.mean(yh[efm] == c)) if efm.any() else None,
                         'acc_with_evidence': float(np.mean(yh[evm] == c)) if evm.any() else None,
                         'evidence_free_predicted_as': {CLS[k]: float(np.mean(yh[efm] == k)) for k in range(6)} if efm.any() else None}
        orc = np.where(ctx, y, 0); summary['l10'] = {'per_class': tab, 'oracle_bound_macro_f1': macro_f1(y, orc)[0], 'model_macro_f1': macro_f1(y, yh)[0]}
        print(f"[L10    ] oracle bound={summary['l10']['oracle_bound_macro_f1']:.4f} model={summary['l10']['model_macro_f1']:.4f}")
        for name, t in tab.items():
            print(f"          {name:7s} EF={t['evidence_free_fraction']:.3f} acc(EF)={t['acc_evidence_free']}  acc(ev)={t['acc_with_evidence']}")
    gate = 'PASS' if (m2 <= SEQ_F1_MAX and (np.isnan(rec_norm) or rec_norm <= NORMAL_RECALL_MAX)) else 'FAIL'
    summary['gate'] = gate; (out / 'fingerprint_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(f"\nGATE: {gate}  (sequence-level macro-F1 <= {SEQ_F1_MAX} and NORMAL recall <= {NORMAL_RECALL_MAX} required)")


if __name__ == '__main__':
    main()
