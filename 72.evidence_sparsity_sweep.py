#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 72: evidence-sparsity sweep with frozen detectors (no training, no selection).

Run from the project code directory (same .venv / GPU environment as Steps 47-49):
    python "72.evidence_sparsity_sweep.py" --preflight-only
    python "72.evidence_sparsity_sweep.py" --generate-only      (optional; no neural models)
    python "72.evidence_sparsity_sweep.py"

Question. Does the benefit of cross-window integration follow the probability that
the context contains observable attack evidence? Earlier evidence for this rests on
one generator operating point (Steps 12-15, 47, 52, 62) and a post-hoc stratification
(Step 63). Step 49's parameter regimes changed four factors jointly. Here ONE factor
changes: the attack event retention of the reviewed Step 49 generator.

Design (fixed before any score is read).
* Levels R100/R060/R035/R020 = event_retention 1.0/0.6/0.35/0.2. Benign-condition odds,
  receiver dispersion and attack severity stay at 1.0. The reviewed Step 49 code
  (sha bb75f26d...) generates every level with the original makers and the unmodified
  common mapper; its root seed is reused, so R100 must reproduce Step 49 P_reference
  byte-for-byte (checked when that run is present) and all levels share one nominal
  timeline. RF/GNSS retention uses fixed per-window uniform draws shared across levels;
  injection retains attempts from a fixed stream; replay/ground scale their existing
  recurrence/attempt strength. Realized evidence per class and level is measured and
  reported; the nominal multiplier is not used as the x-axis.
* Detectors, all frozen: the Step 49 L10 families (TST, Mean/Max/MeanMax MLP, LSTM x10
  head seeds; ProbabilityMean, MajorityVote, EWMA; Rule_L10), the Step 12 TST heads at
  L=1/2/3/5 (10 seeds each, same Step 11 runs as the L10 heads), and the Step 8 window
  classifier on the endpoint window. L<10 heads read the last L windows of the same
  common L10 endpoints.
* Evidence-availability reference at L=1/2/3/5/10: the true label if any of the last L
  windows differs from its paired nominal projection, otherwise NORMAL.
* NORMAL records are identical across levels by construction (checked); only attack
  endpoints change.
* Statistics: whole shared nominal episodes (all six class replicas) resampled jointly
  across levels and models, 20,000 draws. Estimand: mean of per-head six-class
  Macro-F1. Prespecified contrasts and prediction-evaluation rules are in protocol.json.

Output: outputs/shared_nominal_v2/72_evidence_sparsity/run_evidence_sparsity_v1 and a
sibling *_review.zip. Rerun the same command to reuse verified completed stages.
Scope: within-generator, frozen-model sensitivity analysis; not external validation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT_REL = 'outputs/shared_nominal_v2/72_evidence_sparsity/run_evidence_sparsity_v1'
STEP49_RUN_REL = 'outputs/shared_nominal_v2/49_parameter_stress/run_parameter_stress_v1'
VERSION = 'shared_nominal_evidence_sparsity_sweep_v1'
STEP49_SHA = 'bb75f26db4da3ada807e27e4505658efc8aa98b075fe6efe4d6092206b1292b2'
STEP48_SHA = '7ef7d833fcf5c3cbe39675a583093f5b41f7dfe1c82ae393d298ca303cb56a39'
STEP47_SHA = 'd41f481693a05a8181c69a2d8064259732c145218f67358007faff0af22ee716'
LEVELS = (('R100', 1.0), ('R060', 0.6), ('R035', 0.35), ('R020', 0.2))
CONDITIONS = tuple(name for name, _ in LEVELS)
PARAMETERS = {name: dict(benign_odds_multiplier=1., receiver_std_multiplier=1., severity_scale=1., event_retention=r)
              for name, r in LEVELS}
SUB_LENGTHS = (1, 2, 3, 5)
LENGTHS = SUB_LENGTHS + (10,)
DRAWS = 20000
BOOTSTRAP_SEED = 2026100972
BATCH = 200
TRACKING_MARGIN = 0.03
POOL_MARGIN = 0.01
METRICS = ('macro_f1', 'attack_macro_recall', 'normal_false_alarm_rate')
STRATA = ('current_changed', 'history_only_changed', 'no_observable_change')
m49 = e = m48 = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_step49(root, explicit=None):
    candidates = [Path(explicit)] if explicit else [p for p in root.glob('*.py') if re.match(r'^49[. _-]', p.name)]
    unique = {sha256(p): p.resolve() for p in candidates if p.is_file()}
    require(len(unique) == 1, f'Expected one current Step 49 script in {root}; found {len(unique)} versions')
    path = next(iter(unique.values()))
    require(sha256(path) == STEP49_SHA, 'Step 49 code differs from the reviewed parameter-stress version (bb75f26d...); '
            'Step 72 reuses its generator internals and must not run on a different version')
    return path


# --------------------------------------------------------------------------- run grid
def families72(selected_pool):
    return ([f'Reference_L{L}' for L in LENGTHS] + ['WindowClassifier_L1'] + [f'TST_L{L}' for L in LENGTHS]
            + ['Mean_MLP', 'Max_MLP', 'MeanMax_MLP', 'LSTM', 'ProbabilityMean', 'MajorityVote', 'EWMA', 'Rule_L10'])


def runs72():
    return ([(f'Reference_L{L}', None) for L in LENGTHS] + [('WindowClassifier_L1', None)]
            + [(f'TST_L{L}', s) for L in SUB_LENGTHS for s in e.SEEDS]
            + [('TST_L10' if f == 'TST' else f, s) for f, s in m49.RUNS])


# --------------------------------------------------------------------------- generator configuration
def configure_sweep(module):
    """Patch the reviewed Step 49 module so its unchanged generator produces the retention-only levels."""
    require(len(CONDITIONS) == len(module.CONDITIONS) == 4, 'Step 49 statistics assume four conditions')
    module.CONDITIONS = CONDITIONS
    module.PARAMETERS = PARAMETERS
    original = module.generator_protocol

    def generator_protocol():
        g = original()
        g.update(design='single-factor event-retention sweep (Step 72); benign odds, receiver dispersion and severity fixed at 1.0',
                 severity='Severity scale fixed at 1.0 at every level; only event_retention varies.',
                 scope='Within-generator evidence-sparsity sensitivity with frozen models; not external validation.',
                 nested_draws='RF/GNSS retention uses the fixed per-window uniform draws of the Step 49 perturbation stream, '
                              'shared across levels; injection uses the fixed attempt stream; replay/ground scale strength variables.')
        return g
    module.generator_protocol = generator_protocol


# --------------------------------------------------------------------------- L-sweep heads (file level; no torch)
def audit_lsweep_heads(root, context):
    helper = context['helper']; observed = context['observed']; heads = {}
    for seed in e.SEEDS:
        manifest = Path(context['expected'][('TST', seed)]['source_manifest']); folder = manifest.parent
        run = helper.load_json(manifest, observed)
        l10 = context['heads'][e.run_key('TST', seed)]
        require(Path(l10['checkpoint']).resolve() == (folder / 'L10/checkpoints/best_tst_model_L10.pt').resolve(),
                f'TST seed {seed}: L10 head is not in the Step 11 run of its source manifest')
        artifacts = run.get('artifacts') or {}
        for L in SUB_LENGTHS:
            rel = dict(checkpoint=f'L{L}/checkpoints/best_tst_model_L{L}.pt', report=f'L{L}/reports/evaluation_metrics_L{L}.json',
                       val=f'L{L}/predictions/val_predictions_L{L}.csv')
            require(all(n in artifacts for n in rel.values()), f'TST seed {seed}: Step 11 run lacks L{L} artifacts')
            helper.verify_selected(folder, artifacts, list(rel.values()), observed)
            report = helper.load_json(folder / rel['report'], observed)
            mc, tc = report['model_config'], report['train_config']
            require(int(report.get('sequence_length', -1)) == L and mc['sequence_length'] == L and mc['num_classes'] == 6
                    and mc['input_dim'] == 768 and tc['random_seed'] == seed and report.get('label_order') == list(e.LABELS),
                    f'TST seed {seed}: L{L} report schema/seed differs')
            require(tc.get('use_missing_aware_training') is False and not tc.get('use_robust_eval_as_primary_for_long_contexts', False),
                    f'TST seed {seed}: L{L} head is not a clean-trained head')
            require(report.get('provenance') == l10['provenance'], f'TST seed {seed}: L{L} provenance differs from its L10 head')
            a = {k: v for k, v in mc.items() if k not in ('sequence_length', 'temporal_context_seconds')}
            b = {k: v for k, v in l10['model_config'].items() if k not in ('sequence_length', 'temporal_context_seconds')}
            require(a == b, f'TST seed {seed}: L{L} architecture differs from its L10 head')
            heads[f'TST_L{L}/seed_{seed}'] = dict(seed=seed, length=L, folder=str(folder),
                checkpoint=str(folder / rel['checkpoint']), checkpoint_sha256=artifacts[rel['checkpoint']],
                val_predictions=str(folder / rel['val']), val_predictions_sha256=artifacts[rel['val']],
                model_config=mc, train_config=tc, provenance=report['provenance'],
                runtime=l10['runtime'], settings=l10['settings'])
    source = context['source_contract']
    names = [f'val_tst_sequences_L{L}.npz' for L in SUB_LENGTHS]
    require(all(n in source['step10_artifacts'] for n in names), 'Step 10 contract lacks validation L-sequences')
    helper.verify_selected(root / e.SEQ_REL, source['step10_artifacts'], names, observed)
    return heads


def audit_step49_reference(root, explicit):
    folder = (Path(explicit) if explicit else root / STEP49_RUN_REL).resolve()
    names = ('evaluation_records.csv', 'evaluation_window_observable_effects.csv', 'evaluation_endpoints_L10.csv')
    files = {n: folder / 'generation/P_reference' / n for n in names}
    if not all(p.is_file() for p in files.values()):
        return dict(available=False, folder=str(folder), rule='R100 reproduction check skipped: Step 49 P_reference files not found')
    return dict(available=True, folder=str(folder), sha256={n: sha256(p) for n, p in files.items()},
                rule='R100 must reproduce these Step 49 P_reference CSV files byte-for-byte')


# --------------------------------------------------------------------------- protocol
def make_protocol72(context, previous, rule_info, paths, lsweep, step49_reference, original_sources):
    protocol = m49.make_protocol(context, previous, rule_info, paths)
    protocol['step49_style_statistics'] = protocol.pop('statistics')
    protocol.update(version=VERSION, script_sha256=sha256(Path(__file__)), step49_code_sha256=STEP49_SHA,
        original_maker_sources=original_sources, step49_reference=step49_reference,
        stress_used_for_selection=False, test_used_for_selection=False, training=False,
        primary='Gain from cross-window integration (TST_L10 minus TST_L1 Macro-F1) at each evidence-sparsity level, '
                'its change relative to R100, and the change of the evidence-availability gap (Reference_L10 minus TST_L10)',
        lsweep_heads={k: {kk: vv for kk, vv in v.items() if kk not in ('runtime', 'settings')} for k, v in lsweep.items()},
        runs72=[list(r) for r in runs72()],
        evidence_sparsity=dict(
            manipulated='event_retention', levels=dict(LEVELS), fixed=dict(benign_odds_multiplier=1.0, receiver_std_multiplier=1.0, severity_scale=1.0),
            class_semantics=dict(RF_LINK_JAMMING='per-window effect kept with probability r (fixed draws, nested across levels)',
                GNSS_SPOOFING='per-window effect kept with probability r (fixed draws, nested across levels)',
                UNAUTHORIZED_TELECOMMAND_INJECTION='each sampled attempt window kept with probability r (fixed attempt stream)',
                TELECOMMAND_REPLAY='episode_recurrence_strength multiplied by r', GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE='attempt_strength multiplied by r'),
            realized_measures='attack-active windows, observable-change windows, endpoint evidence strata and evidence presence within the last L windows, per class and level',
            lengths=list(LENGTHS),
            reference='Reference_L: true label if any of the last L windows differs from its paired nominal projection, else NORMAL; audit flags only, never model inputs',
            window_classifier='Step 8 classifier head on the endpoint window CLS (current window only)',
            lsweep_inputs='L<10 heads read the last L windows of the same common L10 endpoints, as in the Step 11/12 common-endpoint evaluation',
            invariants=['NORMAL records identical across levels', 'endpoints, labels, blocks and context identical across levels',
                        'R100 reproduces Step 49 P_reference CSVs when that run is present']),
        statistics=dict(draws=DRAWS, seed=BOOTSTRAP_SEED, unit='whole shared nominal episode containing all six class replicas; same draw for all levels and models',
            estimand='mean of per-head six-class Macro-F1 (also attack macro recall and NORMAL false-alarm rate); deterministic families have one run',
            intervals='95% linear percentile, conditional on this generator realization and the frozen models; no head/encoder/generator-seed resampling',
            primary_families=dict(
                A='gain_s = TST_L10 - TST_L1 at each of four levels; 95% and four-level Bonferroni intervals',
                B='gain_s - gain_R100 at R060/R035/R020; 95% and three-level Bonferroni intervals',
                C=f'gap_s - gap_R100 with gap = Reference_L10 - TST_L10 at R060/R035/R020; 95% and three-level Bonferroni intervals; '
                  f'tracking margin +/-{TRACKING_MARGIN} Macro-F1 prespecified'),
            secondary='per-L gaps and their changes; selected pool minus TST_L10 (margin +/-0.01); window classifier minus TST_L1; '
                      'available integration gain (Reference_L10 minus Reference_L1) and realized-minus-available gain; '
                      'per-family changes vs R100; stratum accuracy; Step 49-style TST comparisons. Descriptive.',
            equivalence_claim=False),
        predictions_before_running=[
            dict(id='P1', statement='Reference values fall as retention falls, and fall more for shorter contexts',
                 rule='consistent if Reference_L is non-increasing over R100>R060>R035>R020 for every L and the R100-R020 drop is larger for L=1 than for L=10; otherwise inconsistent'),
            dict(id='P2', statement='Detectors track the reference: the L10 gap stays approximately constant across levels',
                 rule=f'consistent if every 95% CI of gap_s-gap_R100 lies within +/-{TRACKING_MARGIN}; inconsistent if any CI lies entirely outside; otherwise inconclusive'),
            dict(id='P3', statement='The L10-minus-L1 gain grows at intermediate sparsity (R060 vs R100)',
                 rule='consistent if the 95% CI of gain_R060-gain_R100 is above 0; inconsistent if below 0; otherwise inconclusive; all levels reported'),
            dict(id='P4', statement='The validation-selected order-free pool and TST_L10 stay close at every level',
                 rule=f'consistent if every 95% CI of pool-TST_L10 lies within +/-{POOL_MARGIN}; inconsistent if any CI lies entirely outside; otherwise inconclusive'),
            dict(id='report', statement='All results are reported regardless of direction; no level, model or threshold is changed after scores are read')],
        scope='Retrospective within-generator evidence-sparsity sensitivity with frozen A-trained models. Retention semantics differ by class '
              '(frequency for RF/GNSS/injection, strength scaling for replay/ground); realized evidence is reported. No external validity, '
              'physical calibration, retraining or factor-specific causal claim beyond this generator.')
    return protocol


# --------------------------------------------------------------------------- frozen L-sweep inference
def restore_lsweep(modules, head, context):
    import torch
    path = Path(head['checkpoint'])
    require(sha256(path) == head['checkpoint_sha256'], 'L-sweep checkpoint changed: ' + str(path))
    c = e.read_checkpoint(path); cfg, tc = c['model_config'], c['train_config']
    require(cfg == head['model_config'] and tc == head['train_config'] and c.get('provenance') == head['provenance']
            and c.get('source_contract') == context['source_contract'] and c['label_order'] == list(e.LABELS)
            and np.isfinite(c['best_val_summary']['macro_f1']), 'L-sweep checkpoint contract differs: ' + str(path))
    module = modules['TST']
    torch.set_num_threads(head['runtime'].get('torch_num_threads', head['runtime'].get('threads')))
    torch.backends.cudnn.deterministic = bool(head['settings']['DETERMINISTIC_CUDNN'])
    module.set_all_seeds(head['seed'])
    model = module.TimeSeriesTransformerClassifier(module.ModelConfig(**cfg))
    model.load_state_dict(c['model_state_dict'], strict=True); model.eval().to('cuda')
    for p in model.parameters():
        p.requires_grad_(False)
    return model, c


def lsweep_probabilities(modules, head, context, embeddings, indices):
    import torch
    model, c = restore_lsweep(modules, head, context)
    try:
        return e.predict_head(model, c, 'TST', embeddings, indices)
    finally:
        del model
        torch.cuda.empty_cache()


def lsweep_canaries(root, out, identity, context, data, modules, lsweep):
    folder = out / 'cache' / 'lsweep_replay'; folder.mkdir(parents=True, exist_ok=True)
    contract = e.object_hash(dict(protocol=identity, stage='lsweep_canaries'))
    done = e.read_completion(folder, 'complete.json', contract)
    if done is not None:
        return done
    d = data['val']; rows = []
    for key, head in lsweep.items():
        L = head['length']
        with np.load(root / e.SEQ_REL / f'val_tst_sequences_L{L}.npz', allow_pickle=False) as z:
            ix = np.asarray(z['sequence_indices'], dtype=np.int64); y = np.asarray(z['y'], dtype=np.int64)
        require(ix.ndim == 2 and ix.shape[1] == L and ix.min() >= 0 and ix.max() < len(d['embeddings']), 'Invalid Step 10 val L-sequences')
        n = e.canary_count(len(ix), int(head['train_config']['batch_size']))
        probs = lsweep_probabilities(modules, head, context, d['embeddings'], ix[:n])
        require(sha256(head['val_predictions']) == head['val_predictions_sha256'], 'Saved L-sweep predictions changed')
        saved = pd.read_csv(head['val_predictions'], keep_default_na=False).sort_values('sequence_id').reset_index(drop=True)
        require(np.array_equal(saved.sequence_id.to_numpy()[:n], np.arange(n)) and np.array_equal(saved.true_label_id.to_numpy()[:n], y[:n]),
                key + ': saved validation predictions do not align with Step 10 sequences')
        ref = saved[['prob_' + label.lower() for label in e.LABELS]].to_numpy(float)[:n]
        error = float(np.max(np.abs(probs - ref)))
        require(np.array_equal(probs.argmax(1), ref.argmax(1)), key + ': validation decisions do not reproduce; check source/runtime/precision')
        require(np.allclose(probs, ref, rtol=1e-4, atol=2e-5), f'{key}: validation probabilities differ, max_abs={error}; do not tune tolerances')
        rows.append(dict(head=key, rows=n, max_abs_probability_error=error, prediction_mismatches=0))
    result = dict(status='PASS', heads=len(rows), rows=rows, role='Step 11 saved validation predictions at original batch boundaries; no sweep data used')
    e.write_json(folder / 'evidence.json', result)
    e.write_completion(folder, 'complete.json', contract, result, ['evidence.json'])
    return result


def evaluate_lsweep(out, name, identity, context, modules, lsweep):
    folder = out / 'evaluation' / name; folder.mkdir(parents=True, exist_ok=True)
    contract = e.object_hash(dict(protocol=identity, stage='lsweep', condition=name))
    if e.read_completion(folder, 'lsweep_complete.json', contract) is not None:
        return
    cohort = m48.load_fresh(out / 'generation' / name)
    cache = out / 'cache' / name
    ec = e.object_hash(dict(parent=e.object_hash(dict(protocol=identity, stage='evaluate', name=name)), stage='embeddings'))
    require(e.read_completion(cache, 'embeddings_complete.json', ec) is not None, 'Missing verified embeddings for ' + name)
    embeddings = np.load(cache / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    require(embeddings.shape == (len(cohort['frame']), 768) and embeddings.dtype == np.float32, 'Invalid cached embeddings: ' + name)

    def missing():
        raise ValueError('Window probabilities missing for ' + name + '; the neural evaluation must finish first')
    wp = e.cached_probs(out, f'cache/{name}/window_probabilities', identity, missing, expected_rows=len(cohort['frame']))
    window = wp[cohort['indices'][:, -1]].argmax(1).astype(np.int8)
    predictions = []; keys = []
    for L in SUB_LENGTHS:
        idx = np.ascontiguousarray(cohort['indices'][:, -L:])
        for seed in e.SEEDS:
            head = lsweep[f'TST_L{L}/seed_{seed}']
            print(f'Evaluate {name}: TST_L{L}, seed={seed}', flush=True)
            p = e.cached_probs(out, f'cache/{name}/lsweep/TST_L{L}/seed_{seed}', identity,
                               lambda h=head, i=idx: lsweep_probabilities(modules, h, context, embeddings, i), expected_rows=len(cohort['y']))
            predictions.append(p.argmax(1).astype(np.int8)); keys.append((f'TST_L{L}', seed))
    P = np.stack(predictions)
    e.write_array(folder / 'lsweep_predictions.npz', predictions=P, window_classifier=window, y=cohort['y'],
                  sequence_id=cohort['meta'].sequence_id.to_numpy(), families=np.array([k[0] for k in keys]), seeds=np.array([k[1] for k in keys]))
    rows = [dict(condition=name, family=f, seed=s, **{k: v for k, v in e.metrics(cohort['y'], p).items() if k in ('macro_f1', 'accuracy')})
            for (f, s), p in zip(keys + [('WindowClassifier_L1', None)], list(P) + [window])]
    e.write_csv(folder / 'lsweep_metrics_by_run.csv', rows)
    e.write_completion(folder, 'lsweep_complete.json', contract, dict(condition=name, endpoints=len(cohort['y']), runs=len(keys) + 1),
                       ['lsweep_predictions.npz', 'lsweep_metrics_by_run.csv'])


# --------------------------------------------------------------------------- generation invariants
def generation_invariants(out, identity, step49_reference):
    folder = out / 'audit' / 'invariants'; folder.mkdir(parents=True, exist_ok=True)
    contract = e.object_hash(dict(protocol=identity, stage='invariants'))
    done = e.read_completion(folder, 'complete.json', contract)
    if done is not None:
        return done
    cohorts = {n: m48.load_fresh(out / 'generation' / n) for n in CONDITIONS}
    base = cohorts[CONDITIONS[0]]; normal = base['frame'].label.eq('NORMAL').to_numpy()
    blocks0 = pd.read_csv(out / 'generation' / CONDITIONS[0] / 'endpoint_blocks.csv')
    report = dict(levels=list(CONDITIONS))
    for n in CONDITIONS[1:]:
        c = cohorts[n]
        require(np.array_equal(c['indices'], base['indices']) and np.array_equal(c['y'], base['y']), 'Endpoints/labels differ across levels: ' + n)
        m48.same_frame(c['meta'], base['meta'], 'Endpoint metadata differs across levels: ' + n)
        m48.same_frame(pd.read_csv(out / 'generation' / n / 'endpoint_blocks.csv'), blocks0, 'Endpoint blocks differ across levels: ' + n)
        m48.same_frame(c['frame'].loc[normal], base['frame'].loc[normal], 'NORMAL records differ across levels: ' + n)
    report['normal_records_identical_across_levels'] = True
    report['endpoints_labels_blocks_identical_across_levels'] = True
    if step49_reference['available']:
        ours = {k: sha256(out / 'generation' / CONDITIONS[0] / k) for k in step49_reference['sha256']}
        require(ours == step49_reference['sha256'], 'R100 does not reproduce Step 49 P_reference; makers, shared modules or environment differ. Stop.')
        report['R100_reproduces_step49_P_reference'] = True
    else:
        report['R100_reproduces_step49_P_reference'] = 'not_checked_step49_run_absent'
    e.write_json(folder / 'audit.json', report)
    e.write_completion(folder, 'complete.json', contract, report, ['audit.json'])
    return report


# --------------------------------------------------------------------------- statistics helpers
def run_metrics(z):
    """z[..., 3, 6] = (true positives, support, predicted) -> [..., 3] metrics."""
    tp, sup, pred = z[..., 0, :], z[..., 1, :], z[..., 2, :]
    den = sup + pred
    f1 = np.divide(2. * tp, den, out=np.zeros(tp.shape, dtype=float), where=den > 0).mean(-1)
    rec = np.divide(tp, sup, out=np.zeros(tp.shape, dtype=float), where=sup > 0)
    return np.stack([f1, rec[..., 1:].mean(-1), 1. - rec[..., 0]], axis=-1)


def family_reduce(values, groups):
    """values[..., R, 3] -> [..., F, 3] (mean over the runs of each family)."""
    return np.stack([values[..., g, :].mean(-2) for g in groups], axis=-2)


def sufficient_statistics(y, codes, n_clusters, predictions):
    """predictions[C, R, n] -> [K, C, R, 3, 6]."""
    C, R, _ = predictions.shape
    out = np.zeros((n_clusters, C, R, 3, 6), dtype=np.int64)
    for c in range(C):
        for r in range(R):
            cm = np.bincount(codes * 36 + 6 * y + predictions[c, r].astype(np.int64), minlength=n_clusters * 36).reshape(n_clusters, 6, 6)
            out[:, c, r, 0] = np.diagonal(cm, axis1=1, axis2=2); out[:, c, r, 1] = cm.sum(2); out[:, c, r, 2] = cm.sum(1)
    return out


def bootstrap_family(suff, groups, draws=DRAWS, seed=BOOTSTRAP_SEED, batch=BATCH, progress=True):
    K = suff.shape[0]; C, R = suff.shape[1], suff.shape[2]
    flat = suff.reshape(K, -1).astype(np.float64)
    rng = np.random.Generator(np.random.PCG64(seed)); out = np.empty((draws, C, len(groups), 3), dtype=np.float32)
    for lo in range(0, draws, batch):
        n = min(batch, draws - lo)
        w = rng.multinomial(K, np.full(K, 1. / K), size=n).astype(np.float64)
        out[lo:lo + n] = family_reduce(run_metrics((w @ flat).reshape(n, C, R, 3, 6)), groups)
        if progress and (lo == 0 or (lo + n) % 2000 == 0 or lo + n == draws):
            print(f'Sweep bootstrap {lo + n:,}/{draws:,}', flush=True)
    return out


def quantiles(samples, alpha):
    lo, hi = np.quantile(np.asarray(samples, dtype=np.float64), [alpha / 2, 1 - alpha / 2], method='linear')
    return float(lo), float(hi)


def contrast_row(family, label, point, samples, comparisons=1, margin=None, **extra):
    lo, hi = quantiles(samples, .05)
    blo, bhi = quantiles(samples, .05 / comparisons) if comparisons > 1 else (lo, hi)
    row = dict(family=family, contrast=label, estimate=float(point), ci95_low=lo, ci95_high=hi,
               bonferroni_comparisons=comparisons, bonferroni_low=blo, bonferroni_high=bhi,
               interpretation='positive' if lo > 0 else 'negative' if hi < 0 else 'zero_included_not_equivalence', **extra)
    if margin is not None:
        row.update(margin=margin, within_margin=bool(-margin <= lo and hi <= margin),
                   outside_margin=bool(lo > margin or hi < -margin))
    return row


def margin_verdict(rows):
    if all(r['within_margin'] for r in rows):
        return 'consistent'
    if any(r['outside_margin'] for r in rows):
        return 'inconsistent'
    return 'inconclusive'


def reference_predictions(flags, indices, y):
    return {L: np.where(flags[indices[:, -L:]].any(1), y, 0).astype(np.int8) for L in LENGTHS}


def evaluate_predictions(point, samples, fam, rows_c, pool_rows):
    """Apply the prespecified rules P1-P4 (point[C, F, 3], samples[B, C, F, 3])."""
    ref = {L: point[:, fam[f'Reference_L{L}'], 0] for L in LENGTHS}
    monotone = all(np.all(np.diff(ref[L]) <= 1e-12) for L in LENGTHS)
    drop1, drop10 = ref[1][0] - ref[1][-1], ref[10][0] - ref[10][-1]
    gain_change = samples[:, 1, fam['TST_L10'], 0] - samples[:, 1, fam['TST_L1'], 0] - (samples[:, 0, fam['TST_L10'], 0] - samples[:, 0, fam['TST_L1'], 0])
    lo, hi = quantiles(gain_change, .05)
    return [
        dict(id='P1', verdict='consistent' if monotone and drop1 > drop10 else 'inconsistent',
             evidence=f'Reference non-increasing for all L: {monotone}; R100-R020 drop L1={drop1:.4f}, L10={drop10:.4f}'),
        dict(id='P2', verdict=margin_verdict(rows_c), evidence='; '.join(f"{r['contrast']}: {r['estimate']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]" for r in rows_c)),
        dict(id='P3', verdict='consistent' if lo > 0 else 'inconsistent' if hi < 0 else 'inconclusive',
             evidence=f'gain_R060-gain_R100 95% CI [{lo:.4f}, {hi:.4f}]'),
        dict(id='P4', verdict=margin_verdict(pool_rows), evidence='; '.join(f"{r['contrast']}: {r['estimate']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]" for r in pool_rows))]


# --------------------------------------------------------------------------- sweep statistics
def load_level(out, name, runs):
    cohort = m48.load_fresh(out / 'generation' / name)
    y = cohort['y']; ix = cohort['indices']
    blocks = pd.read_csv(out / 'generation' / name / 'endpoint_blocks.csv')
    m48.same_frame(blocks[['sequence_id', 'episode_id']], cohort['meta'][['sequence_id', 'episode_id']], 'Unaligned blocks: ' + name)
    audit = pd.read_csv(out / 'generation' / name / 'evaluation_window_observable_effects.csv', keep_default_na=False)
    m48.same_frame(audit[list(e.META)], cohort['frame'][list(e.META)], 'Window effects do not align: ' + name)
    flags = audit.audit_observable_effect_present.to_numpy(dtype=np.int64)
    with np.load(out / 'generation' / name / 'measurement_audit.npz', allow_pickle=False) as z:
        active = np.asarray(z['attack_active'], dtype=np.int64)
    require(len(active) == len(flags) == len(cohort['frame']), 'Measurement audit length differs: ' + name)
    with np.load(out / 'evaluation' / name / 'combined_predictions.npz', allow_pickle=False) as z:
        require(np.array_equal(z['y'], y) and np.array_equal(z['sequence_id'], cohort['meta'].sequence_id)
                and np.array_equal(z['families'], [f for f, s in m49.RUNS])
                and np.array_equal(z['seeds'], [s if s is not None else -1 for f, s in m49.RUNS]), 'Step 49-style predictions differ: ' + name)
        combined = np.asarray(z['predictions'], dtype=np.int8)
    with np.load(out / 'evaluation' / name / 'lsweep_predictions.npz', allow_pickle=False) as z:
        expected = [(f'TST_L{L}', s) for L in SUB_LENGTHS for s in e.SEEDS]
        require(np.array_equal(z['y'], y) and np.array_equal(z['families'], [f for f, s in expected])
                and np.array_equal(z['seeds'], [s for f, s in expected]), 'L-sweep predictions differ: ' + name)
        lsweep = np.asarray(z['predictions'], dtype=np.int8); window = np.asarray(z['window_classifier'], dtype=np.int8)
    refs = reference_predictions(flags, ix, y)
    by_run = {}
    for L in LENGTHS:
        by_run[(f'Reference_L{L}', None)] = refs[L]
    by_run[('WindowClassifier_L1', None)] = window
    for (f, s), p in zip([(f'TST_L{L}', s) for L in SUB_LENGTHS for s in e.SEEDS], lsweep):
        by_run[(f, s)] = p
    for (f, s), p in zip(m49.RUNS, combined):
        by_run[('TST_L10' if f == 'TST' else f, s)] = p
    P = np.stack([by_run[r] for r in [tuple(x) for x in runs]]).astype(np.int8)
    return dict(cohort=cohort, y=y, indices=ix, blocks=blocks.block_id.to_numpy(), flags=flags, active=active, predictions=P)


def realized_evidence(name, level):
    frame = level['cohort']['frame']; y = level['y']; ix = level['indices']; flags = level['flags']; active = level['active']
    strata = level['cohort']['effects'].audit_effect_stratum.to_numpy()
    rows = []
    for k, label in enumerate(e.LABELS):
        w = frame.label.eq(label).to_numpy(); endpoint = y == k
        row = dict(level=name, retention=PARAMETERS[name]['event_retention'], label=label, windows=int(w.sum()),
                   attack_active_windows=int(active[w].sum()), observable_change_windows=int(flags[w].sum()),
                   observable_change_fraction=float(flags[w].mean()), endpoints=int(endpoint.sum()))
        for s in STRATA:
            row['endpoints_' + s] = int((strata[endpoint] == s).sum())
        for L in LENGTHS:
            row[f'evidence_present_L{L}'] = float(flags[ix[endpoint][:, -L:]].any(1).mean()) if endpoint.any() else None
        rows.append(row)
    return rows


def stratum_rows(name, level, runs, families):
    y = level['y']; strata = level['cohort']['effects'].audit_effect_stratum.to_numpy(); P = level['predictions']; rows = []
    for family in families:
        idx = [i for i, (f, s) in enumerate(runs) if f == family]
        for s in STRATA:
            for k in range(1, 6):
                mask = (strata == s) & (y == k)
                if not mask.any():
                    continue
                rows.append(dict(level=name, family=family, stratum=s, label=e.LABELS[k], endpoints=int(mask.sum()),
                                 mean_run_accuracy=float((P[idx][:, mask] == k).mean()),
                                 mean_run_predicted_normal=float((P[idx][:, mask] == 0).mean())))
    return rows


def sweep_statistics(out, protocol):
    folder = out / 'sweep_statistics'; folder.mkdir(parents=True, exist_ok=True); identity = protocol['identity']
    contract = e.object_hash(dict(protocol=identity, stage='sweep_statistics'))
    if e.read_completion(folder, 'complete.json', contract) is not None:
        return
    selected = protocol['frozen_selection']['selected_pool']
    runs = [tuple(r) for r in protocol['runs72']]; families = families72(selected)
    groups = [[i for i, (f, s) in enumerate(runs) if f == fam] for fam in families]
    require(all(groups) and sum(len(g) for g in groups) == len(runs), 'Run grid and families differ')
    fam = {f: i for i, f in enumerate(families)}
    levels = {n: load_level(out, n, runs) for n in CONDITIONS}
    base = levels[CONDITIONS[0]]
    for n in CONDITIONS[1:]:
        require(np.array_equal(levels[n]['y'], base['y']) and np.array_equal(levels[n]['blocks'], base['blocks']), 'Unpaired levels: ' + n)
    codes, names = pd.factorize(base['blocks'], sort=True)
    for i in range(len(names)):
        counts = np.bincount(base['y'][codes == i], minlength=6)
        require((counts > 0).all(), 'Incomplete paired six-class block')
    P = np.stack([levels[n]['predictions'] for n in CONDITIONS])
    suff = sufficient_statistics(base['y'], codes.astype(np.int64), len(names), P)
    run_point = run_metrics(suff.sum(0).astype(np.float64))
    point = family_reduce(run_point, groups)
    samples = bootstrap_family(suff, groups)
    # Family summaries.
    summary = []
    for ci, n in enumerate(CONDITIONS):
        for fi, f in enumerate(families):
            row = dict(level=n, retention=PARAMETERS[n]['event_retention'], family=f, runs=len(groups[fi]))
            for mi, metric in enumerate(METRICS):
                lo, hi = quantiles(samples[:, ci, fi, mi], .05)
                row[metric] = float(point[ci, fi, mi]); row[metric + '_ci95_low'] = lo; row[metric + '_ci95_high'] = hi
                row[metric + '_seed_sd'] = float(run_point[ci, groups[fi], mi].std(ddof=1)) if len(groups[fi]) > 1 else None
            summary.append(row)
    perclass = []
    tp, sup, pred = [suff.sum(0)[..., j, :].astype(float) for j in range(3)]
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * tp, sup + pred, out=np.zeros_like(tp), where=(sup + pred) > 0)
    for ci, n in enumerate(CONDITIONS):
        for fi, f in enumerate(families):
            for k, label in enumerate(e.LABELS):
                perclass.append(dict(level=n, family=f, label=label, mean_run_recall=float(rec[ci, groups[fi], k].mean()),
                                     mean_run_f1=float(f1[ci, groups[fi], k].mean())))
    # Primary contrasts.
    S = samples[..., 0].astype(np.float64); T = point[..., 0]
    def gain(x, ci): return x[..., ci, fam['TST_L10']] - x[..., ci, fam['TST_L1']]
    def gap(x, ci, L=10): return x[..., ci, fam[f'Reference_L{L}']] - x[..., ci, fam[f'TST_L{L}']]
    primary = [contrast_row('A', f'gain_{n}', gain(T, ci), gain(S, ci), 4) for ci, n in enumerate(CONDITIONS)]
    primary += [contrast_row('B', f'gain_{n}-gain_R100', gain(T, ci) - gain(T, 0), gain(S, ci) - gain(S, 0), 3) for ci, n in enumerate(CONDITIONS) if ci]
    rows_c = [contrast_row('C', f'gap10_{n}-gap10_R100', gap(T, ci) - gap(T, 0), gap(S, ci) - gap(S, 0), 3, margin=TRACKING_MARGIN)
              for ci, n in enumerate(CONDITIONS) if ci]
    primary += rows_c
    # Secondary contrasts (descriptive).
    secondary = []
    for L in LENGTHS:
        for ci, n in enumerate(CONDITIONS):
            secondary.append(contrast_row('D_gap', f'gap_L{L}_{n}', gap(T, ci, L), gap(S, ci, L)))
            if ci:
                secondary.append(contrast_row('D_gap_change', f'gap_L{L}_{n}-gap_L{L}_R100', gap(T, ci, L) - gap(T, 0, L), gap(S, ci, L) - gap(S, 0, L)))
    pool_rows = []
    for ci, n in enumerate(CONDITIONS):
        pool_rows.append(contrast_row('E_pool', f'{selected}-TST_L10_{n}', T[ci, fam[selected]] - T[ci, fam['TST_L10']],
                                      S[:, ci, fam[selected]] - S[:, ci, fam['TST_L10']], margin=POOL_MARGIN))
        secondary.append(contrast_row('F_window', f'WindowClassifier_L1-TST_L1_{n}', T[ci, fam['WindowClassifier_L1']] - T[ci, fam['TST_L1']],
                                      S[:, ci, fam['WindowClassifier_L1']] - S[:, ci, fam['TST_L1']]))
        secondary.append(contrast_row('F_pool_gain', f'{selected}-TST_L1_{n}', T[ci, fam[selected]] - T[ci, fam['TST_L1']],
                                      S[:, ci, fam[selected]] - S[:, ci, fam['TST_L1']]))
        ref_gain_T = T[ci, fam['Reference_L10']] - T[ci, fam['Reference_L1']]; ref_gain_S = S[:, ci, fam['Reference_L10']] - S[:, ci, fam['Reference_L1']]
        secondary.append(contrast_row('H_available_gain', f'Reference_L10-Reference_L1_{n}', ref_gain_T, ref_gain_S))
        secondary.append(contrast_row('H_realized_minus_available', f'gain_{n}-available_gain_{n}', gain(T, ci) - ref_gain_T, gain(S, ci) - ref_gain_S))
        if ci:
            for f in families:
                secondary.append(contrast_row('G_change', f'{f}_{n}-{f}_R100', T[ci, fam[f]] - T[0, fam[f]], S[:, ci, fam[f]] - S[:, 0, fam[f]]))
    secondary = pool_rows + secondary
    verdicts = evaluate_predictions(point, samples, fam, rows_c, pool_rows)
    # Realized evidence, mechanism and collapse data.
    realized = [r for n in CONDITIONS for r in realized_evidence(n, levels[n])]
    strata = [r for n in CONDITIONS for r in stratum_rows(n, levels[n], runs, ['WindowClassifier_L1', 'TST_L1', 'TST_L10', selected, 'Reference_L10'])]
    collapse = []
    for ci, n in enumerate(CONDITIONS):
        attack = base['y'] > 0
        for L in LENGTHS:
            r_lo, r_hi = quantiles(S[:, ci, fam[f'Reference_L{L}']], .05); t_lo, t_hi = quantiles(S[:, ci, fam[f'TST_L{L}']], .05)
            collapse.append(dict(level=n, retention=PARAMETERS[n]['event_retention'], length=L,
                evidence_present_attack_endpoints=float(levels[n]['flags'][base['indices'][attack][:, -L:]].any(1).mean()),
                reference_macro_f1=float(T[ci, fam[f'Reference_L{L}']]), reference_ci95_low=r_lo, reference_ci95_high=r_hi,
                tst_macro_f1=float(T[ci, fam[f'TST_L{L}']]), tst_ci95_low=t_lo, tst_ci95_high=t_hi,
                gap=float(gap(T, ci, L))))
    cp = pd.DataFrame(collapse)
    collapse_summary = dict(points=len(cp), pearson_reference_vs_tst=float(np.corrcoef(cp.reference_macro_f1, cp.tst_macro_f1)[0, 1]),
                            mean_gap=float(cp.gap.mean()), max_abs_gap=float(cp.gap.abs().max()), scope='descriptive only')
    # NORMAL prediction stability (inputs are identical across levels by construction).
    normal = base['y'] == 0; stability = []
    for ri, (f, s) in enumerate(runs):
        stability.append(dict(family=f, seed=s, **{f'normal_prediction_changes_{n}_vs_R100': int((P[ci, ri, normal] != P[0, ri, normal]).sum())
                                                   for ci, n in enumerate(CONDITIONS) if ci}))
    # Write.
    tables = {'family_summary.csv': summary, 'per_class.csv': perclass, 'primary_contrasts.csv': primary,
              'secondary_contrasts.csv': secondary, 'prediction_evaluation.csv': verdicts, 'realized_evidence.csv': realized,
              'stratum_accuracy.csv': strata, 'collapse_points.csv': collapse, 'normal_prediction_stability.csv': stability}
    for filename, rows in tables.items():
        e.write_csv(folder / filename, rows)
    e.write_array(folder / 'bootstrap_family_metrics.npz', samples=samples, point=point.astype(np.float32),
                  levels=np.array(CONDITIONS), families=np.array(families), metrics=np.array(METRICS))
    e.write_array(folder / 'block_sufficient_statistics.npz', counts=suff, blocks=np.asarray(names),
                  runs=np.array([f for f, s in runs]), seeds=np.array([s if s is not None else -1 for f, s in runs]))
    summary_json = dict(levels=dict(LEVELS), selected_pool=selected, clusters=len(names), draws=DRAWS, seed=BOOTSTRAP_SEED,
        primary={r['contrast']: dict(estimate=r['estimate'], ci95=[r['ci95_low'], r['ci95_high']],
                                     bonferroni=[r['bonferroni_low'], r['bonferroni_high']]) for r in primary},
        predictions={v['id']: v['verdict'] for v in verdicts}, collapse=collapse_summary,
        normal_prediction_changes_total=int(sum(v for r in stability for k, v in r.items() if k.startswith('normal_prediction_changes'))),
        interpretation='Conditional on this generator realization, the frozen models and the realized evidence per level; '
                       'retention semantics differ by class; descriptive outside the prespecified contrasts.')
    e.write_json(folder / 'sweep_summary.json', summary_json)
    files = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != 'complete.json' and not p.name.endswith('.tmp'))
    e.write_completion(folder, 'complete.json', contract, dict(draws=DRAWS, clusters=len(names), files=len(files)), files)


# --------------------------------------------------------------------------- completion and archive
def required_stages(identity):
    stages = [('cache/original_replay', 'complete.json', dict(protocol=identity, stage='numerical_canaries')),
              ('cache/lsweep_replay', 'complete.json', dict(protocol=identity, stage='lsweep_canaries')),
              ('audit/invariants', 'complete.json', dict(protocol=identity, stage='invariants')),
              ('statistics', 'complete.json', dict(protocol=identity, stage='statistics')),
              ('sweep_statistics', 'complete.json', dict(protocol=identity, stage='sweep_statistics'))]
    for n in CONDITIONS:
        stages += [(f'generation/{n}', 'complete.json', dict(protocol=identity, stage='generation', condition=n)),
                   (f'evaluation/{n}', 'complete.json', dict(protocol=identity, stage='evaluate', name=n)),
                   (f'evaluation/{n}', 'transfer_complete.json', dict(protocol=identity, stage='combined_rule_and_neural', condition=n)),
                   (f'evaluation/{n}', 'lsweep_complete.json', dict(protocol=identity, stage='lsweep', condition=n))]
    return stages


def finish_archive(out, identity):
    path = out / 'run_manifest.json'
    if path.exists():
        manifest = e.read_json(path)
        require(manifest.get('status') == 'EVIDENCE_SPARSITY_COMPLETE' and manifest.get('protocol_identity') == identity, 'Wrong completed manifest')
        require(manifest.get('identity') == e.object_hash({k: v for k, v in manifest.items() if k != 'identity'}), 'Manifest changed')
        for name, sha in manifest['output_sha256'].items():
            require(e.safe_artifact(out, name).is_file() and e.digest(out / name) == sha, 'Completed output changed/missing: ' + name)
    else:
        for relative, filename, contract in required_stages(identity):
            require(e.read_completion(out / relative, filename, e.object_hash(contract)) is not None, f'Missing stage {relative}/{filename}')
        m48.verify_completion_tree(out)
        files = [p.relative_to(out).as_posix() for p in sorted(out.rglob('*')) if p.is_file() and p.name != '.run.lock' and not p.name.endswith('.tmp')]
        review = [n for n in files if not n.startswith('cache/') or n.endswith(('complete.json', 'audit.json', 'evidence.json'))]
        manifest = dict(status='EVIDENCE_SPARSITY_COMPLETE', protocol_identity=identity, completed_at_utc=e.utc(), levels=dict(LEVELS),
                        trained_l10_heads=50, lsweep_heads=40, deterministic_controls=4, references=len(LENGTHS), training=False,
                        sparsity_used_for_selection=False, output_sha256={n: e.digest(out / n) for n in files}, review_files=review,
                        review_archive_omits='Embedding and probability caches stay on the PC; every retained file is bound here.')
        manifest['identity'] = e.object_hash(manifest); e.write_json(path, manifest)
    return m48.package_review(out, manifest)


def execute(root, out, protocol, context, data, previous, rule, rule_info, paths, lsweep, step49_reference, generate_only=False):
    protocol = json.loads(json.dumps(protocol, allow_nan=False)); protocol['identity'] = e.object_hash(protocol)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'protocol.json').exists():
        require(not any(p.name != '.run.lock' for p in out.iterdir()), 'Foreign files in the output folder; choose an empty --output-root')
    e.save_matching_json(out / 'protocol.json', protocol); identity = protocol['identity']
    if (out / 'run_manifest.json').exists():
        return finish_archive(out, identity)
    m48.verify_completion_tree(out)
    e.save_matching_json(out / 'input_audit.json', context['observed'])
    e.save_matching_json(out / 'validation_selection.json', previous['selection'])
    e.save_matching_json(out / 'rule_source.json', rule_info)
    source = out / 'source'; source.mkdir(exist_ok=True)
    for path in [Path(__file__), *paths.values(), context['helper_path'], *context['code_paths'].values(), Path(rule_info['code'])]:
        dest = source / Path(path).name
        if dest.exists():
            require(e.digest(dest) == e.digest(path), 'Saved source changed: ' + dest.name)
        else:
            shutil.copyfile(path, dest)
    if generate_only:
        m49.generate_all(out, protocol); m49.audit_train_overlap(root, out, identity)
        generation_invariants(out, identity, step49_reference)
        context['helper'].recheck_inputs(context['observed'])
        require(sha256(Path(__file__)) == protocol['script_sha256'], 'Step 72 changed during generation')
        return None
    runtime = e.evaluation_runtime(context)
    require(runtime == protocol['expected_inference_runtime'], 'Use the same inference environment/device/settings as the completed Step 47')
    e.save_matching_json(out / 'inference_runtime.json', runtime)
    modules = e.import_models(context)
    for family, seed in e.RUNS[:50]:
        e.validate_checkpoint(e.read_checkpoint(e.paths_for(family, seed, context)[0]), family, seed, context)
    tokenizer, encoder = e.load_encoder(root, context)
    canary = m48.replay_canaries(root, out, identity, context, data, modules, tokenizer, encoder)
    lcanary = lsweep_canaries(root, out, identity, context, data, modules, lsweep)
    e.save_matching_json(out / 'canary_audit.json', dict(neural=canary, lsweep=lcanary, rule_L10=rule_info['original_replay']))
    m49.generate_all(out, protocol)
    generation_invariants(out, identity, step49_reference)
    m49.audit_token_budget(out, tokenizer, protocol, context); m49.audit_train_overlap(root, out, identity)
    for name in CONDITIONS:
        m49.evaluate_condition(root, out, name, protocol, context, tokenizer, encoder, modules, rule, rule_info)
    del encoder; gc.collect()
    for name in CONDITIONS:
        evaluate_lsweep(out, name, identity, context, modules, lsweep)
    context['helper'].recheck_inputs(context['observed'])
    require(sha256(Path(__file__)) == protocol['script_sha256'], 'Step 72 source changed during execution')
    m49.finish_statistics(out, protocol, previous, rule_info)
    sweep_statistics(out, protocol)
    return finish_archive(out, identity)


def main(argv=None):
    global m49, e, m48
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--step49-code', type=Path); parser.add_argument('--step47-code', type=Path); parser.add_argument('--step48-code', type=Path)
    parser.add_argument('--step47-root', type=Path); parser.add_argument('--step49-run', type=Path, help='Completed Step 49 parameter-stress run (for the R100 reproduction check)')
    parser.add_argument('--shared-code-dir', type=Path)
    parser.add_argument('--maker-code-dir', type=Path); parser.add_argument('--maker-data-dir', type=Path)
    parser.add_argument('--step15-code', type=Path); parser.add_argument('--step16-code', type=Path); parser.add_argument('--rule-run', type=Path)
    parser.add_argument('--tst-root', type=Path); parser.add_argument('--pooling-root', type=Path); parser.add_argument('--lstm-root', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true', help='Verify sources, makers, Step 47/16 and L-sweep artifacts; no generation, no neural models')
    mode.add_argument('--generate-only', action='store_true', help='Generate and check the four levels; no neural models')
    args = parser.parse_args(argv); root = args.project_root.resolve()
    path49 = resolve_step49(root, args.step49_code)
    m49 = import_script(path49, 'step72_step49')
    paths = m49.load_dependencies(root, args); e = m49.e; m48 = m49.m48
    require(e.digest(paths['47']) == STEP47_SHA and e.digest(paths['48']) == STEP48_SHA,
            'Step 47/48 code differs from the versions used by the reviewed Step 49 run')
    paths['49'] = path49
    configure_sweep(m49)
    print('Step72: ' + VERSION + ' | event-retention sweep; frozen models; no retraining or selection.', flush=True)
    context = e.audit_sources(root, args.step15_code, args.tst_root, args.pooling_root, args.lstm_root)
    for p in paths.values():
        context['helper'].track_file(p, context['observed'])
    makers, splitter, maker_paths = m48.load_generators(root, (args.maker_code_dir or args.shared_code_dir or root).resolve(),
                                                        context['build'], context['observed'], context['helper'])
    m49.makers = makers; m49.nominal = sys.modules['shared_nominal']; m49.context_module = sys.modules['shared_context_v4']
    paths.update(maker_paths)
    for key, expected in m49.GENERATOR_SOURCE_SHA256.items():
        require(e.digest(paths[key]) == expected, 'Maker/shared source differs from the audited generator: ' + key)
    original_sources = m48.audit_original_makers(root, (args.maker_data_dir or root / 'dataset').resolve(), context, splitter)
    require(len(set(m49._SEEDS)) == 8 and not set(m49.MAKER_SEEDS).intersection(s['seed'] for s in original_sources),
            'Sweep seeds overlap original maker seeds')
    previous = m48.audit_step47(root, (args.step47_root or root / e.OUT_REL).resolve(), context, paths['47'])
    data = e.load_cohorts(root, context)
    rule_folder = m49.resolve_rule_run(root, args.rule_run, context['source_contract'])
    rule, rule_info = m49.audit_rule(root, rule_folder, context, data, args.step16_code)
    lsweep = audit_lsweep_heads(root, context)
    step49_reference = audit_step49_reference(root, args.step49_run)
    protocol = make_protocol72(context, previous, rule_info, paths, lsweep, step49_reference, original_sources)
    context['helper'].recheck_inputs(context['observed'])
    if args.preflight_only:
        print(f'PREFLIGHT PASS. L-sweep heads verified: {len(lsweep)}; Step 49 reference available: {step49_reference["available"]}. '
              'No generation or neural backend loaded.', flush=True)
        return 0
    out = (args.output_root or root / OUT_REL).resolve()
    protected = [root / 'outputs/shared_nominal_v2/49_generator_b_transfer', root / 'outputs/shared_nominal_v2/49_parameter_stress',
                 root / 'dataset', root / e.ENC_REL, root / e.EMB_REL, root / e.SEQ_REL, Path(previous['folder']), rule_folder,
                 *[Path(v) for v in context['roots'].values()]]
    m48.validate_output_location(out, protected)
    out.mkdir(parents=True, exist_ok=True)
    with e.RunLock(out / '.run.lock'):
        archive = execute(root, out, protocol, context, data, previous, rule, rule_info, paths, lsweep, step49_reference, args.generate_only)
    if archive is None:
        print('GENERATION ONLY COMPLETE. Rerun without --generate-only for the frozen-model evaluation.', flush=True)
    else:
        print(f'COMPLETE. Review archive: {archive}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'\nSTOPPED: {type(exc).__name__}: {exc}\nResolve the cause and rerun the same command; completed stages are verified and reused.',
              file=sys.stderr, flush=True)
        raise
