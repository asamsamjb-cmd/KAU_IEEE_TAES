#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TAES A-prime / Step 15: paired, class-stratified episode bootstrap.

사용법: 이 파일을 10~14번 .py 파일과 같은 프로젝트 폴더에 두고 실행합니다.
필수 패키지: numpy, pandas. GPU / torch / 재학습 / 추가 데이터 생성 불필요.
기본 실행: 5 architectures x 10 saved head seeds, L10, 20,000 resamples.

Statistical contract (v1.0, set before computing interval results):
* Estimand = arithmetic mean of the 10 per-seed, sequence-level Macro-F1s.
  Never pool seeds' confusion matrices before F1, or average probabilities.
* Resampling unit = whole episode. Within each true class, sample the same
  number of episodes with replacement. Retain ALL sequences of each selected
  episode, including its multiplicity. Longer episodes retain their original
  larger contribution to the sequence-level metric (not episode-mean F1).
* Apply one common episode draw to all models AND all fixed head seeds.
* Primary CI = two-sided percentile 95% interval, conditional on the observed
  trained runs, fixed encoder, training data, generator realization/protocol,
  and class-specific episode counts. Seeds are NOT resampled. Their sample SD
  is reported separately and is NOT a confidence interval.
* Four predeclared contrasts: Mean/Max/MeanMax/LSTM minus TST. Also report
  Bonferroni marginal 98.75% percentile intervals (nominal family coverage 95%,
  subject to bootstrap approximation). Do not infer equivalence from a CI
  containing zero. No post-hoc practical-equivalence margin or p-values.
* Episode independence/exchangeability within class is an assumption. This
  analysis does not establish generalization to a new encoder or generator.
* Original predictions are read-only. Each execution creates a new run folder.
  Missing/invalid inputs STOP the run; no partial-seed 'formal' result is made.

CLI examples (optional; PyCharm Run needs no arguments):
  python "15.paired_episode_bootstrap.py"
  python "15.paired_episode_bootstrap.py" --audit-only
  python "15.paired_episode_bootstrap.py" --project-root "D:/research/project"

Outputs in outputs/15_paired_episode_bootstrap/run_<UTC timestamp>/:
  protocol.json, input_audit.csv, metrics_by_seed.csv, model_summary.csv,
  paired_comparisons.csv, bootstrap_distribution.npz,
  seed_average_row_normalized_confusion.csv, episode_inventory.csv,
  analysis_summary.md, run_manifest.json (status must be COMPLETE).

Reference for paired index resampling and percentile intervals:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
Implementation uses NumPy episode-confusion sufficient statistics to avoid
materializing resampled prediction rows. No SciPy dependency is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ======================== USER CONFIG ========================
BASE_DIR = Path(__file__).resolve().parent
SEEDS = tuple(range(20260610, 20260620))
N_BOOTSTRAP = 20_000
BOOTSTRAP_RANDOM_SEED = 20260926
BOOTSTRAP_BATCH_SIZE = 100
DATASET_VERSION = 'shared_nominal_v2'      # benchmark version; all inputs must come from this tree
OUT_ROOT = f'outputs/{DATASET_VERSION}'
EXPECTED_TEST_L10_COUNT = None             # resolved from the Step-10 report index (dataset guard), never hard-coded
MODELS = ('TST', 'Mean_MLP', 'Max_MLP', 'MeanMax_MLP', 'LSTM')
LABELS = (
    'NORMAL', 'RF_LINK_JAMMING', 'GNSS_SPOOFING',
    'UNAUTHORIZED_TELECOMMAND_INJECTION', 'TELECOMMAND_REPLAY',
    'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE',
)
METADATA_REL = f'{OUT_ROOT}/tst_sequences_finetuned_main/test_tst_sequence_metadata_L10.csv'
SEQUENCE_INDEX_REL = f'{OUT_ROOT}/tst_sequences_finetuned_main/sequence_build_report_index.json'
MULTISEED_VALIDATION_REL = f'{OUT_ROOT}/tst_training_finetuned_multiseed/multiseed_validation.json'
MULTISEED_MANIFEST_REL = f'{OUT_ROOT}/tst_training_finetuned_multiseed/multiseed_run_manifest.json'
POOLING_SUMMARY_NAME = '39_pooling_mlp_summary.json'
LSTM_SUMMARY_NAME = 'lstm_finetuned_distilbert_l10_summary.json'
OUTPUT_REL = f'{OUT_ROOT}/15_paired_episode_bootstrap'
# Source directories retain the names actually used by Steps 12, 13 and 14.
# Do not rename the existing '39_pooling...' directory.
PATH_TEMPLATES = {
    'TST': OUT_ROOT + '/tst_training_finetuned_multiseed/seed_{seed}/L10/predictions/test_predictions_L10.csv',
    'Mean_MLP': OUT_ROOT + '/39_pooling_mlp_finetuned_main/predictions/mean_seed{seed}_test_predictions.csv',
    'Max_MLP': OUT_ROOT + '/39_pooling_mlp_finetuned_main/predictions/max_seed{seed}_test_predictions.csv',
    'MeanMax_MLP': OUT_ROOT + '/39_pooling_mlp_finetuned_main/predictions/meanmax_seed{seed}_test_predictions.csv',
    'LSTM': OUT_ROOT + '/lstm_finetuned_distilbert_l10/predictions/seed{seed}_test_predictions.csv',
}
KEY_COLUMNS = ('episode_id', 'sequence_start_row', 'sequence_end_row',
               'sequence_start_step', 'sequence_end_step')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def integers(series, context):
    x = pd.to_numeric(series, errors='raise').to_numpy(dtype=np.float64)
    require(np.isfinite(x).all() and np.equal(x, np.floor(x)).all()
            and (np.abs(x) < 2**53).all(), f'{context}: invalid integer values')
    return x.astype(np.int64)


def read_csv(path):
    return pd.read_csv(path, encoding='utf-8-sig', dtype={'episode_id': str})


def load_metadata(path, expected_count):
    require(isinstance(expected_count, int) and expected_count > 0, 'expected_count must be resolved from the dataset contract')
    d = read_csv(path)
    needed = {'sequence_id', 'label_id', 'label', 'sequence_length',
              'source_split', *KEY_COLUMNS}
    require(needed <= set(d), f'{path}: missing metadata columns {sorted(needed-set(d))}')
    require(len(d) == expected_count, f'{path}: expected {expected_count} rows, got {len(d)}')
    require(not d[list(needed)].isna().any().any(), f'{path}: missing metadata values')
    for col in ('sequence_id', 'label_id', 'sequence_length', *KEY_COLUMNS[1:]):
        d[col] = integers(d[col], f'{path}:{col}')
    require(d.sequence_id.is_unique, f'{path}: duplicate sequence_id')
    d = d.sort_values('sequence_id').reset_index(drop=True)
    require(np.array_equal(d.sequence_id, np.arange(len(d))), f'{path}: sequence_id must be 0..N-1')
    require(set(d.label_id) == set(range(len(LABELS))), f'{path}: expected all six classes')
    require(np.array_equal(d.label, np.array(LABELS)[d.label_id]), f'{path}: label mapping mismatch')
    require((d.source_split == 'test').all(), f'{path}: non-test metadata')
    require((d.sequence_length == 10).all(), f'{path}: non-L10 sequences')
    require((d.sequence_end_step - d.sequence_start_step == 9).all(), f'{path}: non-L10 steps')
    require((d[list(KEY_COLUMNS[1:])] >= 0).all().all(), f'{path}: negative positions')
    require((d.sequence_end_row >= d.sequence_start_row).all(), f'{path}: invalid row interval')
    require(d.episode_id.str.strip().ne('').all(), f'{path}: empty episode_id')
    require(not d.duplicated(['episode_id', 'sequence_end_step']).any(), f'{path}: duplicate endpoint')
    require(d.groupby('episode_id').label_id.nunique().eq(1).all(),
            f'{path}: mixed-label episodes: class-stratified protocol is not applicable')
    if 'sequence_num_label_changes' in d:
        require((integers(d.sequence_num_label_changes, str(path)) == 0).all(),
                f'{path}: label transitions present; review protocol')
    return d


def load_prediction(path, ref):
    before = sha256_file(path)
    d = read_csv(path)
    needed = {'sequence_id', 'true_label_id', 'pred_label_id', 'true_label',
              'pred_label', *KEY_COLUMNS}
    require(needed <= set(d), f'{path}: missing columns {sorted(needed-set(d))}')
    require(len(d) == len(ref), f'{path}: row count {len(d)} != {len(ref)}')
    require(not d[list(needed)].isna().any().any(), f'{path}: missing prediction values')
    for col in ('sequence_id', 'true_label_id', 'pred_label_id', *KEY_COLUMNS[1:]):
        d[col] = integers(d[col], f'{path}:{col}')
    require(d.sequence_id.is_unique, f'{path}: duplicate sequence_id')
    reordered = not np.array_equal(d.sequence_id, ref.sequence_id)
    d = d.sort_values('sequence_id').reset_index(drop=True)
    for col in ('sequence_id', *KEY_COLUMNS):
        require(np.array_equal(d[col], ref[col]), f'{path}: {col} mismatch against Step 10')
    for col in ('sequence_start_timestamp', 'sequence_end_timestamp'):
        if col in ref and col in d:
            require(np.array_equal(d[col], ref[col]), f'{path}: {col} mismatch')
    y = d.true_label_id.to_numpy()
    p = d.pred_label_id.to_numpy()
    require(np.array_equal(y, ref.label_id), f'{path}: true_label_id mismatch')
    require(((p >= 0) & (p < len(LABELS))).all(), f'{path}: predicted label outside 0..5')
    require(np.array_equal(d.true_label, np.array(LABELS)[y]), f'{path}: true_label mismatch')
    require(np.array_equal(d.pred_label, np.array(LABELS)[p]), f'{path}: pred_label mismatch')
    if 'correct' in d:
        require(np.array_equal(integers(d.correct, str(path)), (y == p).astype(int)),
                f'{path}: correct column mismatch')
    probcols = ['prob_' + x.lower() for x in LABELS]
    require(set(probcols) <= set(d), f'{path}: missing class probability columns')
    probs = d[probcols].to_numpy(dtype=float)
    require(np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all(),
            f'{path}: invalid probabilities')
    require(np.allclose(probs.sum(1), 1, rtol=0, atol=2e-6), f'{path}: probability sums != 1')
    require(np.array_equal(probs.argmax(1), p), f'{path}: probability argmax != prediction')
    if 'confidence' in d:
        require(np.allclose(d.confidence.to_numpy(float), probs.max(1), rtol=0, atol=2e-6),
                f'{path}: confidence mismatch')
    require(before == sha256_file(path), f'{path}: changed while being read')
    return p, {'path': str(Path(path).resolve()), 'sha256': before,
               'rows': len(d), 'episodes': int(d.episode_id.nunique()),
               'rows_reordered': bool(reordered), 'alignment': 'PASS'}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _find_one(directory, name):
    hits = sorted(Path(directory).rglob(name))
    require(len(hits) >= 1, f'{directory}: {name} not found (run the producing step under {OUT_ROOT})')
    return hits[-1]


def resolve_dataset_contract(root):
    """Read the dataset_id of every producer (Steps 10, 12, 13, 14) and require agreement.

    This replaces the hard-coded A-prime sequence count: the expected number of
    L=10 test endpoints is taken from the Step-10 report index of the same tree.
    """
    root = Path(root)
    index = _read_json(root/SEQUENCE_INDEX_REL)
    dataset_id = index.get('dataset_id')
    require(isinstance(dataset_id, str) and len(dataset_id) >= 16, f'{SEQUENCE_INDEX_REL}: dataset_id missing')
    counts = index.get('common_endpoint_report', {}).get('splits', {})
    expected = counts.get('test', {}).get('common_endpoint_count')
    require(isinstance(expected, int) and expected > 0, f'{SEQUENCE_INDEX_REL}: test common_endpoint_count missing')
    sources = {'step10_sequence_index': dataset_id}
    ms = _read_json(root/MULTISEED_VALIDATION_REL)
    sources['step12_multiseed_validation'] = ms.get('source_contract', {}).get('dataset_id')
    pool = _read_json(_find_one(root/OUT_ROOT/'39_pooling_mlp_finetuned_main', POOLING_SUMMARY_NAME))
    sources['step13_pooling_summary'] = pool.get('manifest', {}).get('revised_pipeline_provenance', {}).get('dataset_id')
    lstm = _read_json(_find_one(root/OUT_ROOT/'lstm_finetuned_distilbert_l10', LSTM_SUMMARY_NAME))
    sources['step14_lstm_summary'] = lstm.get('manifest', {}).get('revised_pipeline_provenance', {}).get('dataset_id')
    bad = {k: v for k, v in sources.items() if v != dataset_id}
    require(not bad, 'dataset_id mismatch across producers; do not mix artifacts: ' + json.dumps(bad))
    return {'dataset_version': DATASET_VERSION, 'dataset_id': dataset_id,
            'expected_test_l10_count': int(expected), 'sources': sources}


def tst_prediction_paths(root, seeds):
    """TST paths come from the Step-12 manifest: seed 20260610 is 'reused_main' and lives in
    tst_training_finetuned_main, the other seeds in tst_training_finetuned_multiseed/seed_*."""
    manifest = _read_json(Path(root)/MULTISEED_MANIFEST_REL)
    runs = {int(r['seed']): r for r in manifest.get('runs', []) if 'seed' in r}
    out = {}
    for seed in seeds:
        run = runs.get(int(seed))
        require(run is not None, f'Step-12 manifest has no run for seed {seed}')
        require(run.get('status') in ('completed', 'reused_main'),
                f'Step-12 run for seed {seed} has status {run.get("status")}; all ten seeds must be complete')
        out_root = Path(str(run['output_root']))
        if not out_root.is_absolute():
            out_root = Path(root)/out_root
        out[int(seed)] = out_root/'L10'/'predictions'/'test_predictions_L10.csv'
    return out


def check_inputs(root, seeds):
    tst_paths = tst_prediction_paths(root, seeds)
    paths = [(model, seed, (tst_paths[int(seed)] if model == 'TST'
                            else Path(root)/PATH_TEMPLATES[model].format(seed=seed)))
             for model in MODELS for seed in seeds]
    missing = [str(path) for _, _, path in paths if not path.is_file()]
    meta = Path(root)/METADATA_REL
    if not meta.is_file():
        missing.append(str(meta))
    if missing:
        raise FileNotFoundError(
            f'Need all {len(paths)} prediction CSVs and Step-10 metadata. '
            f'Missing {len(missing)} file(s); no bootstrap performed:\n' + '\n'.join(missing))
    return meta, paths


def macro_f1(cm):
    """Macro-F1 over last two confusion axes (true, predicted)."""
    cm = np.asarray(cm)
    den = cm.sum(axis=-1) + cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    return np.divide(2.0 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(axis=-1)


def episode_confusions(y, predictions, episode_codes, n_episodes, n_classes):
    """Return [episode, run, true, predicted]; run order is model-major."""
    predictions = np.asarray(predictions)
    counts = np.zeros((n_episodes, len(predictions), n_classes, n_classes), dtype=np.int64)
    for r, p in enumerate(predictions):
        codes = (episode_codes * n_classes + y) * n_classes + p
        counts[:, r] = np.bincount(codes, minlength=n_episodes*n_classes*n_classes).reshape(
            n_episodes, n_classes, n_classes)
    return counts


def draw_episode_weights(strata, batch, rng):
    """Multinomial multiplicities exactly represent within-class cluster draws."""
    weights = np.zeros((batch, len(strata)), dtype=np.int64)
    for c in np.unique(strata):
        ix = np.flatnonzero(strata == c)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=batch)
    return weights


def bootstrap_scores(counts, strata, n_seeds, n_models, n_resamples,
                     random_seed, batch_size, progress=True):
    require(counts.shape[1] == n_seeds*n_models, 'run/seed dimension mismatch')
    require(len(strata) == counts.shape[0], 'episode stratum dimension mismatch')
    require(n_resamples > 0 and batch_size > 0, 'positive resample and batch counts required')
    rng = np.random.default_rng(random_seed)
    # F1 requires TP, true support and predicted support, not all K*K entries.
    sufficient = np.stack((np.diagonal(counts, axis1=-2, axis2=-1),
                           counts.sum(-1), counts.sum(-2)), axis=-2).astype(float)
    flat = sufficient.reshape(len(strata), -1)
    scores = np.empty((n_resamples, n_models), dtype=float)
    start = time.perf_counter()
    for lo in range(0, n_resamples, batch_size):
        hi = min(lo+batch_size, n_resamples)
        w = draw_episode_weights(strata, hi-lo, rng)
        sampled = (w.astype(float) @ flat).reshape(hi-lo, n_models, n_seeds, 3, counts.shape[-1])
        den = sampled[..., 1, :] + sampled[..., 2, :]
        per_class = np.divide(2*sampled[..., 0, :], den,
                              out=np.zeros_like(den), where=den > 0)
        # First class macro-average, THEN seed average. Never an ensemble.
        scores[lo:hi] = per_class.mean(-1).mean(-1)
        if progress and (lo == 0 or hi == n_resamples or hi % 1000 == 0):
            print(f'  Bootstrap {hi:,}/{n_resamples:,} | {time.perf_counter()-start:.1f}s', flush=True)
    return scores


def interval_direction(lo, hi):
    if lo > 0:
        return 'comparator_higher'
    if hi < 0:
        return 'TST_higher'
    return 'zero_included_not_equivalence'


def save_results(out, metrics, counts, scores, seeds):
    n_seeds = len(seeds)
    cm = counts.sum(0).reshape(len(MODELS), n_seeds, len(LABELS), len(LABELS))
    f1 = macro_f1(cm)
    pd.DataFrame(metrics).to_csv(out/'metrics_by_seed.csv', index=False, encoding='utf-8-sig')
    rows = []
    for i, name in enumerate(MODELS):
        lo, hi = np.quantile(scores[:, i], [.025, .975])
        rows.append({'model': name, 'n_seeds': n_seeds, 'macro_f1_mean': float(f1[i].mean()),
                     'macro_f1_seed_sd_ddof1': float(f1[i].std(ddof=1)),
                     'episode_ci95_low': float(lo), 'episode_ci95_high': float(hi)})
    pd.DataFrame(rows).to_csv(out/'model_summary.csv', index=False, encoding='utf-8-sig')
    comparisons = []
    contrasts = scores[:, 1:] - scores[:, [0]]
    for j, name in enumerate(MODELS[1:]):
        delta = f1[j+1].mean()-f1[0].mean()
        lo, hi = np.quantile(contrasts[:, j], [.025, .975])
        # 4 contrasts: alpha/(2*4) per tail => marginal coverage 98.75%.
        blo, bhi = np.quantile(contrasts[:, j], [.00625, .99375])
        comparisons.append({'comparator': name, 'reference': 'TST',
            'delta_macro_f1': float(delta), 'delta_percentage_points': float(100*delta),
            'ci95_low': float(lo), 'ci95_high': float(hi),
            'bonferroni_ci98_75_low': float(blo), 'bonferroni_ci98_75_high': float(bhi),
            'ci95_interpretation': interval_direction(lo, hi),
            'familywise_interpretation': interval_direction(blo, bhi)})
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv', index=False, encoding='utf-8-sig')
    norm = cm / cm.sum(-1, keepdims=True)
    norm = norm.mean(1)  # normalize each seed's true row, then average seeds
    confusion_rows = [{'model': model, 'true_label': LABELS[t], 'pred_label': LABELS[p],
                       'seed_mean_row_normalized': float(norm[i,t,p])}
                      for i, model in enumerate(MODELS)
                      for t in range(len(LABELS)) for p in range(len(LABELS))]
    pd.DataFrame(confusion_rows).to_csv(out/'seed_average_row_normalized_confusion.csv',
                                       index=False, encoding='utf-8-sig')
    np.savez_compressed(out/'bootstrap_distribution.npz', model_names=np.array(MODELS),
                        head_seeds=np.array(seeds), seed_mean_macro_f1=scores,
                        comparator_minus_tst=contrasts)
    text = ['# Step 15: paired episode bootstrap', '',
            'Delta = comparator minus TST. Values are Macro-F1 on the 0–1 scale.', '',
            '| Comparator | Delta | 95% percentile CI | Bonferroni 98.75% CI |',
            '|---|---:|---|---|']
    for r in comparisons:
        text.append(f"| {r['comparator']} | {r['delta_macro_f1']:+.8f} | "
                    f"[{r['ci95_low']:+.8f}, {r['ci95_high']:+.8f}] | "
                    f"[{r['bonferroni_ci98_75_low']:+.8f}, {r['bonferroni_ci98_75_high']:+.8f}] |")
    text += ['', 'Intervals condition on the ten observed trained heads and fixed encoder/data.',
             'Head seeds are averaged, not resampled. Seed SD is reported separately.',
             'An interval containing zero does not establish equivalence.',
             'Statistical direction does not by itself establish practical importance.',
             'The four adjusted intervals target nominal 95% family coverage under the bootstrap approximation.',
             'CSV alignment and hashes do not independently certify checkpoint provenance.',
             'No operational satellite deployment or generator-shift claim follows from this analysis.']
    (out/'analysis_summary.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print('\n'+pd.DataFrame(comparisons)[['comparator','delta_macro_f1','ci95_low','ci95_high']].to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE_DIR)
    parser.add_argument('--audit-only', action='store_true', help='Validate all 50 CSVs without bootstrap')
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    require(len(SEEDS) == 10 and len(set(SEEDS)) == 10, 'Formal protocol requires ten unique seeds')
    require(N_BOOTSTRAP >= 20_000, 'Protocol requires at least 20,000 bootstrap replicates')
    contract = resolve_dataset_contract(root)
    expected_count = contract['expected_test_l10_count']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = root/OUTPUT_REL/f'run_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        'version': '1.1', 'created_utc': stamp, 'project_root': str(root),
        'dataset_contract': contract,
        'models': list(MODELS), 'head_seeds': list(SEEDS), 'L': 10,
        'expected_sequences': expected_count, 'bootstrap_replicates': N_BOOTSTRAP,
        'bootstrap_rng_seed': BOOTSTRAP_RANDOM_SEED, 'batch_size': BOOTSTRAP_BATCH_SIZE,
        'resampling': 'class-stratified whole episodes with replacement; shared across all runs',
        'estimand': 'mean of per-head-seed sequence-level six-class Macro-F1',
        'seed_treatment': 'fixed trained runs; arithmetic mean, not ensemble; no seed resampling',
        'ci_method': 'percentile; numpy quantile default linear interpolation',
        'primary_ci': .95, 'contrast_sign': 'comparator minus TST',
        'contrast_count': 4, 'bonferroni_marginal_ci': .9875,
        'seed_sd_ddof': 1, 'practical_equivalence_margin': None,
        'conditioning': 'fixed training data, encoder, heads, generation regime and class episode counts',
        'assumption': 'episodes independent/exchangeable within class',
        'numpy': np.__version__, 'pandas': pd.__version__, 'python': platform.python_version(),
        'script_sha256': sha256_file(Path(__file__)),
        'source_templates': PATH_TEMPLATES, 'audit_only': args.audit_only,
    }
    write_json(out/'protocol.json', protocol)
    manifest = {'status': 'RUNNING', 'output_dir': str(out), 'protocol_sha256': sha256_file(out/'protocol.json')}
    write_json(out/'run_manifest.json', manifest)
    audits = []
    try:
        metadata_path, paths = check_inputs(root, SEEDS)
        metadata_hash = sha256_file(metadata_path)
        ref = load_metadata(metadata_path, expected_count)
        require(metadata_hash == sha256_file(metadata_path), 'Metadata changed during read')
        codes, episode_ids = pd.factorize(ref.episode_id, sort=True)
        inventory = ref.groupby('episode_id', sort=True).agg(
            label_id=('label_id','first'), label=('label','first'), sequences=('sequence_id','size'))
        require(np.array_equal(inventory.index.to_numpy(), episode_ids.to_numpy()), 'Episode ordering mismatch')
        inventory.to_csv(out/'episode_inventory.csv', encoding='utf-8-sig')
        manifest.update({'metadata_sha256': metadata_hash, 'metadata_path': str(metadata_path),
                         'n_sequences': len(ref), 'n_episodes': len(inventory),
                         'class_episode_counts': inventory.groupby('label').size().to_dict()})
        predictions, metrics = [], []
        print(f'Project: {root}\nOutput: {out}\nValidating all {len(paths)} prediction CSVs...', flush=True)
        y = ref.label_id.to_numpy(dtype=np.int64)
        for i, (model, seed, path) in enumerate(paths, 1):
            pred, audit = load_prediction(path, ref)
            audit.update(model=model, seed=seed)
            audits.append(audit)
            predictions.append(pred)
            cm = np.bincount(y*len(LABELS)+pred, minlength=len(LABELS)**2).reshape(len(LABELS),len(LABELS))
            metrics.append({'model': model, 'seed': seed, 'macro_f1': float(macro_f1(cm)),
                            'accuracy': float((y==pred).mean()), 'errors': int((y!=pred).sum())})
            print(f'  [{i:02d}/{len(paths)}] {model} seed={seed}: PASS | F1={metrics[-1]["macro_f1"]:.8f}', flush=True)
        pd.DataFrame(audits).to_csv(out/'input_audit.csv', index=False, encoding='utf-8-sig')
        if args.audit_only:
            pd.DataFrame(metrics).to_csv(out/'metrics_by_seed.csv', index=False, encoding='utf-8-sig')
            manifest['status'] = 'AUDIT_ONLY_COMPLETE'
        else:
            counts = episode_confusions(y, np.stack(predictions), codes, len(inventory), len(LABELS))
            require(np.array_equal(counts.sum((0,2,3)), np.full(len(paths),len(ref))), 'Confusion count mismatch')
            scores = bootstrap_scores(counts, inventory.label_id.to_numpy(), len(SEEDS), len(MODELS),
                                      N_BOOTSTRAP, BOOTSTRAP_RANDOM_SEED, BOOTSTRAP_BATCH_SIZE)
            save_results(out, metrics, counts, scores, SEEDS)
            manifest['status'] = 'COMPLETE'
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['output_sha256'] = {p.name: sha256_file(p) for p in sorted(out.iterdir())
                                     if p.is_file() and p.name != 'run_manifest.json'}
        write_json(out/'run_manifest.json', manifest)
        print(f'\n{manifest["status"]}\nResults: {out}', flush=True)
        return 0
    except Exception as exc:
        manifest.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        write_json(out/'run_manifest.json', manifest)
        if audits:
            pd.DataFrame(audits).to_csv(out/'partial_input_audit.csv', index=False, encoding='utf-8-sig')
        print(f'\nFAILED: {exc}\nDiagnostic folder: {out}\nDo not rerun training; resolve the reported input issue.',
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
