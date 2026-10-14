#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 48: shared_nominal_v2 fresh realizations, using the frozen Step 47 models.

Run AFTER the revised Step 47 completes:
    python "48.fresh_realizations_evaluation.py"
Keep current 1--7, 11, 13--15, 47, shared_nominal.py and shared_context_v4.py
beside this file. No support ZIP is required. Original maker CSV/raw companions
remain in dataset/ (override --maker-data-dir if necessary).

--preflight-only: file/source/cohort checks, no generation or neural loading.
--generate-only: original replay + three fresh realizations, no neural loading.
The latter also requires completed Step 47 so settings are frozen beforehand.
Rerun the same command to resume. No training or fresh-data model selection.

Same generator rules, new independent default nominal/attack random streams.
This measures within-generator realization sensitivity, NOT independence from
synthetic rules, external validity, NOS3, or an R1-3 independent environment.
All eligible causal stride-1 L10 endpoints remain, including quiet contexts.
Raw counterfactual annotations are diagnostics only, never model inputs.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import gc
import importlib.util
from pathlib import Path
import platform
import re
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT_REL = 'outputs/shared_nominal_v2/48_fresh_realizations/run_fresh_v2'
VERSION = 'shared_nominal_fresh_realizations_v2'
ENGINE_VERSION = 'shared_nominal_uniform_missing8_aggregation_v2'
ROOT_SEEDS = (2026101101, 2026101102, 2026101103)
N_BOOTSTRAP = 20000
BUILD_FUNCTIONS = ('build_normal_dataset', 'build_rf_jamming_dataset',
    'build_gnss_spoofing_dataset', 'build_tc_injection_dataset',
    'build_tc_replay_dataset', 'build_ttc_compromise_dataset')
e = None  # Load the current numbered Step 47 only AFTER parsing CLI / --help.


def load_engine(root, explicit=None):
    paths = [Path(explicit)] if explicit else sorted(root.glob('47*.py'))
    import hashlib
    unique = {hashlib.sha256(p.read_bytes()).hexdigest(): p.resolve()
              for p in paths if p.is_file()}
    if len(unique) != 1:
        raise ValueError('Place one current 47.missing8_aggregation_evaluation.py beside Step 48, or use --step47-code.')
    path = next(iter(unique.values()))
    spec = importlib.util.spec_from_file_location('step48_current47', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, 'PROTOCOL_VERSION', None) != ENGINE_VERSION:
        raise ValueError('Step 48 requires the revised shared_nominal_v2 Step 47, not the old support-folder engine.')
    return module, path


def realization_specs():
    result = []
    for i, seed in enumerate(ROOT_SEEDS, 1):
        values = [int(s.generate_state(1, dtype=np.uint32)[0])
                  for s in np.random.SeedSequence(seed).spawn(8)]
        result.append(dict(name=f'fresh_{i}', root_seed=seed, maker_seeds=values[:6],
                           split_seed=values[6], bootstrap_seed=values[7]))
    return result


def require_seed_separation(original, specs):
    seeds = [s for item in specs for s in item['maker_seeds']]
    e.require(len(set(seeds)) == len(seeds) and not set(seeds).intersection(original),
              'Fresh maker seeds overlap each other or original maker seeds; do not silently replace/select seeds.')
    for field in ('root_seed', 'split_seed', 'bootstrap_seed'):
        values = [s[field] for s in specs]
        e.require(len(set(values)) == len(values), 'Repeated realization ' + field)


def same_frame(actual, expected, message):
    a = actual.reset_index(drop=True).astype(str)
    b = expected.reset_index(drop=True).astype(str)
    e.require(a.equals(b), message)


def compact_raw_columns(splitter):
    return (list(splitter.CANONICAL_COLUMNS)
        + ['baseline_observable_' + f for f in e.FIELDS]
        + ['raw_observable_effect_present', 'raw_observable_difference_count',
           'raw_observable_difference_fields', 'raw_persistent_effect_present',
           'raw_attack_active', 'raw_attacks_enabled', 'raw_paired_audit_background',
           'raw_background_stream'])


def audit_step47(root, folder, context, engine_path):
    """Authenticate the completed current run without rewriting its files/archive."""
    helper = context['helper']; observed = context['observed']
    manifest = helper.load_json(folder / 'run_manifest.json', observed)
    e.require(manifest.get('status') == 'MISSING8_AGGREGATION_COMPLETE',
              'Finish the revised Step 47 before Step 48 (including --generate-only).')
    files = manifest.get('output_sha256', {})
    required = set(e.FINAL_FILES)
    e.require(required.issubset(files), 'Step 47 completion misses required evidence')
    helper.verify_selected(folder, files, list(files), observed)
    protocol = e.read_json(folder / 'protocol.json')
    body = {k: v for k, v in protocol.items() if k != 'identity'}
    e.require(protocol.get('identity') == e.object_hash(body) == manifest.get('protocol_identity'),
              'Step 47 protocol identity differs')
    e.require(protocol.get('version') == ENGINE_VERSION and protocol.get('script_sha256') == e.digest(engine_path),
              'Current Step 47 code does not match its completed run')
    e.require(protocol['source_contract'] == context['source_contract']
              and protocol['checkpoint_contracts'] == context['heads']
              and protocol['source_hashes'] == context['code_hashes']
              and protocol['runs'] == [list(r) for r in e.RUNS],
              'Step 47 used different data/models/head seeds')
    for name, sha in protocol['input_audit'].items():
        path = Path(name)
        e.require(path.is_file() and e.digest(path) == sha, 'Step 47 input changed: ' + name)
        observed[str(path.resolve())] = sha
    selection = e.read_json(folder / 'validation_selection.json')
    validation = pd.read_csv(folder / 'validation_metrics.csv')
    pool = e.choose_pool(validation.to_dict('records'))
    scores = {float(r['alpha']): float(r['macro_f1']) for r in selection['ewma_validation_scores']}
    e.require(set(scores) == set(e.ALPHAS) and all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values()),
              'Invalid Step 47 validation EWMA grid')
    e.require(selection['selected_pool'] == pool and selection['ewma_alpha'] == e.choose_alpha(scores)
              and selection.get('all_ten_head_seeds_retained') is True,
              'Step 47 validation selection changed')
    summary = pd.read_csv(folder / 'model_summary.csv')
    e.require(set(summary.family) == set(e.FAMILIES) and not summary.family.duplicated().any(),
              'Invalid Step 47 original summary')
    return dict(folder=str(folder), protocol=protocol, selection=selection,
                clean_scores=summary.set_index('family').clean_macro_f1.to_dict())


def load_generators(root, maker_dir, build, observed, helper):
    paths = {str(i): resolve_maker(maker_dir, i) for i in range(1, 7)}
    paths['7'] = e.resolve_code(root, 7, build['builder_sha256'])
    for name in ('shared_context_v4', 'shared_nominal'):
        path = (maker_dir / (name + '.py')).resolve()
        e.require(path.is_file(), f'Missing {path.name}; keep BOTH shared Python modules beside makers 1--6.')
        paths[name] = path
        # Explicit local import prevents an unrelated module on sys.path shadowing it.
        e.import_script(path, name)
    nominal = sys.modules['shared_nominal']
    makers = [e.import_script(paths[str(i)], f'step48_maker_{i}') for i in range(1, 7)]
    splitter = e.import_script(paths['7'], 'step48_splitter7')
    e.require(nominal.CODE_VERSION == '2.0-shared-nominal-substrate'
              and tuple(nominal.OBSERVABLE_FIELDS) == e.FIELDS
              and tuple(nominal.METADATA_FIELDS) == e.META, 'Wrong shared nominal implementation')
    for i, module in enumerate(makers, 1):
        e.require(module.MAKER_NUMBER == i and module.CLASS_LABEL == e.LABELS[i-1]
                  and tuple(module.OBSERVABLE_FIELDS) == e.FIELDS
                  and module.nominal is nominal and callable(getattr(module, BUILD_FUNCTIONS[i-1], None)),
                  f'Maker {i} does not use the current common nominal engine')
    policy = build['policy']
    e.require(policy['nominal_substrate'] == 'shared_nominal_v2'
              and policy['window_seconds'] == 1 and policy['columns'] == list(e.META + e.FIELDS)
              and policy['label_order'] == list(e.LABELS)
              and policy['expected_rows_per_class'] == splitter.EXPECTED_ROWS_PER_CLASS
              and policy['ratios'] == [splitter.TRAIN_RATIO, splitter.VAL_RATIO, splitter.TEST_RATIO]
              and policy['seed'] == splitter.DEFAULT_RANDOM_SEED
              and tuple(splitter.OBSERVABLE_FIELDS) == e.FIELDS
              and tuple(splitter.LABEL_ORDER) == e.LABELS,
              'Step 7 generation/split policy differs from the trained corpus')
    for path in paths.values(): helper.track_file(path, observed)
    return makers, splitter, paths


def resolve_maker(folder, number):
    # Numeric boundary matters: maker 1 must not match 11/13/14/15, nor 4 match 47/48.
    candidates = [p.resolve() for p in folder.glob('*.py')
                  if re.match(r'^' + str(number) + r'[. _-]', p.name)]
    unique = {e.digest(p): p for p in candidates}
    e.require(len(unique) == 1, f'Place one current {number}. maker in {folder}; found {len(unique)} distinct versions')
    return next(iter(unique.values()))


def validate_output_location(out, protected):
    out = out.resolve()
    for directory in protected:
        path = Path(directory).resolve()
        e.require(out != path and out not in path.parents and path not in out.parents,
                  'Output root overlaps protected inputs: ' + str(path))


def audit_original_makers(root, data_dir, context, splitter):
    """Recover actual seeds from the authenticated raw stream, not maker defaults."""
    helper = context['helper']; observed = context['observed']; sources = []
    build = context['build']
    e.require([s['label'] for s in build['sources']] == list(e.LABELS), 'Original maker order differs')
    for i, item in enumerate(build['sources'], 1):
        path = e.safe_artifact(data_dir, item['file'])
        raw = e.safe_artifact(data_dir, item['raw_audit']['file'])
        for p, record in ((path, item), (raw, item['raw_audit'])):
            e.require(p.is_file() and e.digest(p) == record['sha256'],
                      'Original maker source missing/changed; select its folder with --maker-data-dir: ' + str(p))
            observed[str(p.resolve())] = record['sha256']
        frame, _ = splitter.validate_schema(e.read_records(path), item['label'], path)
        _, audit = splitter.load_observability_audit(frame, path)
        parts = audit['background_stream'].split(':')
        e.require(len(parts) == 3 and parts[1:] == [str(0x4E4F4D), str(i)],
                  'Original data does not use independent default nominal streams')
        seed = int(parts[0]); start = str(frame.timestamp.iloc[0])
        e.require(seed >= 0, 'Negative original seed')
        sources.append(dict(number=i, label=item['label'], learning=str(path), raw=str(raw),
            seed=seed, rows=len(frame), start_time=start, background_stream=audit['background_stream']))
    require_seed_separation([s['seed'] for s in sources], realization_specs())
    return sources


def make_protocol(context, previous, code_paths, sources):
    return dict(version=VERSION, script_sha256=e.digest(Path(__file__)),
        engine_sha256=e.digest(Path(e.__file__)), source_contract=context['source_contract'],
        checkpoint_contracts=context['heads'], input_audit=context['observed'],
        generator_source_sha256={k: e.digest(v) for k, v in code_paths.items()},
        generation_runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        original_makers=sources, split_policy=context['build']['policy'],
        original_step47_identity=previous['protocol']['identity'], frozen_selection=previous['selection'],
        realizations=realization_specs(), families=list(e.FAMILIES), runs=[list(r) for r in e.RUNS],
        training=False, fresh_validation_used=False, test_used_for_selection=False, missing_fields=False,
        generation=dict(background_seed=None, attacks_enabled=True,
            original_replay='all original learning and compact counterfactual values, all original splits/effect sidecars',
            source_limit='Step 7 records CSV/raw hashes, not original maker code; replay checks observed outputs, not equivalence for every possible seed',
            population='same per-class counts/start times and Step 7 whole-episode split; all test L10 endpoints',
            seed_policy='three prespecified root seeds; separate six maker, split and bootstrap child seeds; no retries/selection',
            overlap='64-bit row-fingerprint diagnostics only; natural repeated vectors retained; no disjoint-feature claim'),
        inference=previous['protocol']['inference'], expected_inference_runtime=previous['protocol']['runtime'],
        primary='TST minus ORIGINAL validation-selected pooling family; mean head-seed F1, then equal mean of realization F1',
        diagnostics='counterfactual projection-change strata; not an oracle ceiling or proof of no information',
        statistics=dict(draws=N_BOOTSTRAP, unit='class-stratified whole episode; paired all 53 runs within realization',
            realization_rng='independent child RNG per realization; average independent bootstrap draws across the three fixed realizations',
            head_seed_resampling=False, realization_resampling=False,
            ci='95% percentile conditional on these three datasets, frozen encoder and trained heads; NOT uncertainty over all possible generator seeds',
            multiplicity='one prespecified primary contrast; seven-comparator Bonferroni supplementary intervals',
            original_comparison='descriptive only; original and fresh episodes are not paired', equivalence_claim=False),
        reviewer_scope='within-generator realization sensitivity; does NOT resolve generator-rule dependence or establish external validity/NOS3/R1-3')


def build_endpoints(frame, window_effects):
    e.require(list(frame) == list(e.META + e.FIELDS), 'Unexpected fresh model input columns')
    same_frame(frame[list(e.META)], window_effects[list(e.META)], 'Fresh effects do not align with records')
    indices = []
    for _, positions in frame.groupby('episode_id', sort=False).indices.items():
        positions = np.asarray(positions)
        steps = frame.iloc[positions].episode_step.to_numpy(dtype=np.int64)
        e.require(np.array_equal(steps, np.arange(len(steps))), 'Fresh episode is incomplete/out of order')
        if len(positions) >= 10: indices.extend(np.lib.stride_tricks.sliding_window_view(positions, 10))
    ix = np.asarray(indices, dtype=np.int64).reshape(-1, 10)
    y = frame.label.map(dict(zip(e.LABELS, range(6)))).to_numpy(dtype=np.int64)[ix[:, -1]]
    e.validate_indices(frame, ix, y)
    e.require(set(y) == set(range(6)), 'Fresh L10 test cohort must retain every class')
    end = frame.iloc[ix[:, -1]].reset_index(drop=True)
    meta = pd.DataFrame(dict(sequence_id=np.arange(len(ix)), episode_id=end.episode_id,
        sequence_start_row=ix[:, 0], sequence_end_row=ix[:, -1],
        sequence_start_step=frame.episode_step.to_numpy()[ix[:, 0]], sequence_end_step=end.episode_step,
        label=end.label, label_id=y, sequence_length=10, source_split='test'))
    flags = window_effects.audit_observable_effect_present.to_numpy(dtype=np.int64)[ix]
    current = flags[:, -1]; history = flags[:, :-1].sum(1)
    effects = pd.DataFrame(dict(audit_current_observable_effect=current,
        audit_history_effect_window_count=history, audit_context_effect_window_count=flags.sum(1),
        audit_effect_stratum=np.where(current != 0, 'current_changed',
            np.where(history != 0, 'history_only_changed', 'no_observable_change'))))
    return ix, y, meta, effects


def row_fingerprints(frame):
    return pd.util.hash_pandas_object(frame[list(e.FIELDS)], index=False).to_numpy(dtype=np.uint64)


def generate_one(root, out, spec, protocol, makers, splitter, reference_hashes):
    name = spec['name']; folder = out / 'generation' / name
    folder.mkdir(parents=True, exist_ok=True)
    contract = e.object_hash(dict(protocol=protocol['identity'], stage='generate', spec=spec))
    done = e.read_completion(folder, 'complete.json', contract)
    if done is not None: return done
    originals = protocol['original_makers']; original = name == 'original_replay'
    frames = []; audits = []; shard_files = []
    raw_columns = compact_raw_columns(splitter)
    for i, (module, fn, source, seed) in enumerate(zip(makers, BUILD_FUNCTIONS, originals, spec['maker_seeds']), 1):
        shard = folder / f'class_{i}'
        shard.mkdir(exist_ok=True)
        learning = shard / 'observable.csv'; raw = shard / 'observable_with_raw.csv'
        sub_contract = e.object_hash(dict(parent=contract, number=i, seed=seed))
        record = e.read_completion(shard, 'complete.json', sub_contract)
        if record is None:
            print(f'Generate {name}: {source["label"]}, rows={source["rows"]:,}, seed={seed}', flush=True)
            full = getattr(module, fn)(num_rows=source['rows'], window_seconds=1,
                start_time=datetime.fromisoformat(source['start_time']), seed=seed,
                return_raw=True, background_seed=None, attacks_enabled=True)
            module.validate_raw_audit_dataset(full, expected_rows=source['rows'], expected_window_seconds=1)
            frame, _ = splitter.validate_schema(full[splitter.CANONICAL_COLUMNS], source['label'], learning)
            compact = full[raw_columns].copy()
            e.require(compact.raw_background_stream.astype(str).eq(f'{seed}:{0x4E4F4D}:{i}').all(),
                      'Maker used an unexpected background stream')
            if original:
                same_frame(frame, e.read_records(Path(source['learning'])), 'Original maker learning replay differs: ' + source['label'])
                old = pd.read_csv(source['raw'], usecols=raw_columns, dtype=str, keep_default_na=False)
                same_frame(compact, old[raw_columns], 'Original counterfactual replay differs: ' + source['label'])
            e.write_csv(learning, frame); e.write_csv(raw, compact)
            del full, compact; gc.collect()
            # Validate the persisted handoff through the ACTUAL current Step 7.
            saved = e.read_records(learning)
            same_frame(saved, frame, 'Generated learning CSV round-trip differs')
            audit, info = splitter.load_observability_audit(saved, learning)
            e.write_completion(shard, 'complete.json', sub_contract, info,
                               ['observable.csv', 'observable_with_raw.csv'])
        else:
            frame = e.read_records(learning)
            frame, _ = splitter.validate_schema(frame, source['label'], learning)
            audit, info = splitter.load_observability_audit(frame, learning)
        frames.append(frame); audits.append(audit)
        shard_files += [f'class_{i}/' + n for n in ('observable.csv', 'observable_with_raw.csv', 'complete.json')]
    merged = splitter.add_internal_global_episode_id(pd.concat(frames, ignore_index=True))
    merged = splitter.rewrite_duplicate_episode_ids_if_needed(merged)
    assigned, manifest = splitter.create_episode_level_split(merged, seed=spec['split_seed'])
    splitter.validate_split_integrity(assigned, manifest)
    aligned = splitter.align_observability_audit(assigned, pd.concat(audits, ignore_index=True))
    manifest = manifest.sort_values(['split', 'label', 'global_episode_id']).reset_index(drop=True)
    e.write_csv(folder / 'episode_split_manifest.csv', manifest)
    partitions = {}; effect_partitions = {}; files = shard_files + ['episode_split_manifest.csv']
    order = ['label', 'episode_id', 'episode_step', 'window_id']
    for split in ('train', 'val', 'test'):
        frame = assigned.loc[assigned.__split.eq(split), splitter.CANONICAL_COLUMNS].sort_values(order).reset_index(drop=True)
        audit = aligned.loc[aligned.__split.eq(split), list(e.META) + splitter.AUDIT_FIELDS].sort_values(order).reset_index(drop=True)
        same_frame(frame[list(e.META)], audit[list(e.META)], 'Split/audit ordering differs')
        if original:
            same_frame(frame, e.read_records(root / e.DATA_REL / f'splits/{split}_observable.csv'), 'Original Step 7 split replay differs: ' + split)
            old = pd.read_csv(root / e.DATA_REL / f'audit/{split}_window_observable_effects.csv', keep_default_na=False)
            same_frame(audit, old, 'Original Step 7 observability replay differs: ' + split)
        e.write_csv(folder / f'{split}_observable.csv', frame)
        e.write_csv(folder / f'{split}_window_observable_effects.csv', audit)
        files += [f'{split}_observable.csv', f'{split}_window_observable_effects.csv']
        partitions[split] = frame; effect_partitions[split] = audit
    report = dict(name=name, maker_seeds=spec['maker_seeds'], split_seed=spec['split_seed'],
        original_replay_passed=original, population={s: dict(rows=len(f), episodes=int(f.episode_id.nunique())) for s, f in partitions.items()},
        fresh_train_val_used_for_fitting_or_selection=False, filtering=False)
    if not original:
        frame = partitions['test'].copy(); audit = effect_partitions['test'].copy()
        hashes = row_fingerprints(frame)
        report['observable_row_overlap'] = {key: dict(rows_matching=int(np.isin(hashes, h).sum()), total_rows=len(frame))
                                           for key, h in reference_hashes.items()}
        # Namespace AFTER the split; IDs stay metadata and never enter serialization.
        frame['episode_id'] = name + '::' + frame.episode_id.astype(str)
        audit['episode_id'] = name + '::' + audit.episode_id.astype(str)
        ix, y, meta, effects = build_endpoints(frame, audit)
        e.write_csv(folder / 'evaluation_records.csv', frame)
        e.write_csv(folder / 'evaluation_window_observable_effects.csv', audit)
        e.write_csv(folder / 'evaluation_endpoints_L10.csv', meta)
        e.write_csv(folder / 'evaluation_context_observable_effects_L10.csv', pd.concat([meta, effects], axis=1))
        e.write_array(folder / 'sequences_L10.npz', indices=ix, y=y)
        files += ['evaluation_records.csv', 'evaluation_window_observable_effects.csv',
                  'evaluation_endpoints_L10.csv', 'evaluation_context_observable_effects_L10.csv', 'sequences_L10.npz']
        report.update(endpoints=len(y), eligible_episodes=int(meta.episode_id.nunique()),
                      effect_strata=effects.audit_effect_stratum.value_counts().to_dict())
    e.write_json(folder / 'generation_audit.json', report)
    e.write_completion(folder, 'complete.json', contract, report, files + ['generation_audit.json'])
    return report


def generate_all(root, out, protocol, makers, splitter):
    references = {'original_train': np.unique(row_fingerprints(e.read_records(root / e.DATA_REL / 'splits/train_observable.csv')))}
    original = dict(name='original_replay', maker_seeds=[s['seed'] for s in protocol['original_makers']],
                    split_seed=protocol['split_policy']['seed'])
    reports = [generate_one(root, out, original, protocol, makers, splitter, references)]
    for spec in protocol['realizations']:
        reports.append(generate_one(root, out, spec, protocol, makers, splitter, references))
        frame = e.read_records(out / 'generation' / spec['name'] / 'evaluation_records.csv')
        references[spec['name'] + '_test'] = np.unique(row_fingerprints(frame))
    e.save_matching_json(out / 'generation_summary.json', dict(realizations=reports,
        overlap_scope='64-bit fingerprint diagnostic, not proof of exact disjointness; repeated observable values remain'))


def replay_canaries(root, out, identity, context, data, modules, tokenizer, encoder):
    folder = out / 'cache' / 'original_replay'; folder.mkdir(parents=True, exist_ok=True)
    contract = e.object_hash(dict(protocol=identity, stage='numerical_canaries'))
    done = e.read_completion(folder, 'complete.json', contract)
    if done is not None: return done
    d = data['val']; n = e.canary_count(len(d['frame']), context['encoder_batch_size'])
    embeddings, info = e.encode_records(d['frame'].iloc[:n], tokenizer, encoder, context)
    reference = d['embeddings'][:n]
    e.require(embeddings.shape == reference.shape and np.allclose(embeddings, reference, rtol=1e-5, atol=2e-5),
              'Step 9 encoder replay differs; stop before fresh evaluation')
    rows = []
    for family, seed in e.RUNS[:50]:
        head = context['heads'][e.run_key(family, seed)]
        batch = head['train_config']['batch_size'] * (2 if family in e.POOLS else 1)
        count = e.canary_count(len(d['y']), batch)
        cohort = dict(d, indices=d['indices'][:count], y=d['y'][:count],
                      references={k: v[:count] for k, v in d['references'].items()})
        probs = e.cached_probs(out, 'cache/original_replay/heads/' + e.run_key(family, seed), identity,
            lambda: e.compute_head_probabilities(root, context, modules, family, seed, cohort, d['embeddings']),
            lambda p: e.verify_clean_replay(p, cohort, family, seed), expected_rows=count)
        rows.append(dict(family=family, seed=seed, rows=count, max_abs_probability_error=float(
            np.abs(probs - cohort['references'][e.run_key(family, seed)]).max())))
    result = dict(status='PASS', encoder_rows=n, encoder_max_abs_error=float(np.abs(embeddings-reference).max()),
        encoding=info, heads=rows, role='whole original batch boundaries; no fresh data for model selection')
    e.write_json(folder / 'evidence.json', result)
    e.write_completion(folder, 'complete.json', contract, result, ['evidence.json'])
    return result


def load_fresh(folder):
    frame = e.read_records(folder / 'evaluation_records.csv')
    audit = pd.read_csv(folder / 'evaluation_window_observable_effects.csv', keep_default_na=False)
    ix, y, meta, effects = build_endpoints(frame, audit)
    with np.load(folder / 'sequences_L10.npz', allow_pickle=False) as z:
        e.require(np.array_equal(z['indices'], ix) and np.array_equal(z['y'], y), 'Fresh sequence payload differs')
    same_frame(meta, pd.read_csv(folder / 'evaluation_endpoints_L10.csv', keep_default_na=False), 'Fresh endpoint file differs')
    same_frame(pd.concat([meta, effects], axis=1), pd.read_csv(folder / 'evaluation_context_observable_effects_L10.csv', keep_default_na=False), 'Fresh context effects differ')
    return dict(frame=frame, indices=ix, y=y, meta=meta, effects=effects)


def evaluate_realization(root, out, spec, protocol, context, tokenizer, encoder, modules):
    name = spec['name']; identity = protocol['identity']; folder = out / 'evaluation' / name
    folder.mkdir(parents=True, exist_ok=True)
    contract = e.object_hash(dict(protocol=identity, stage='evaluate', name=name))
    if e.read_completion(folder, 'complete.json', contract) is not None: return
    cohort = load_fresh(out / 'generation' / name)
    cache = out / 'cache' / name; cache.mkdir(parents=True, exist_ok=True)
    ec = e.object_hash(dict(parent=contract, stage='embeddings'))
    if e.read_completion(cache, 'embeddings_complete.json', ec) is None:
        embeddings, info = e.encode_records(cohort['frame'], tokenizer, encoder, context,
            lambda n, t: print(f'Encode {name}: {n:,}/{t:,}', flush=True))
        e.write_array(cache / 'embeddings.npy', value=embeddings)
        e.write_completion(cache, 'embeddings_complete.json', ec, info, ['embeddings.npy'])
    else:
        embeddings = np.load(cache / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    e.require(embeddings.shape == (len(cohort['frame']), 768) and embeddings.dtype == np.float32
              and np.isfinite(embeddings).all(), 'Invalid fresh embeddings')
    predictions = []
    for family, seed in e.RUNS[:50]:
        print(f'Evaluate {name}: {family}, seed={seed}', flush=True)
        p = e.cached_probs(out, f'cache/{name}/heads/' + e.run_key(family, seed), identity,
            lambda: e.compute_head_probabilities(root, context, modules, family, seed, cohort, embeddings),
            expected_rows=len(cohort['y']))
        predictions.append(p.argmax(1).astype(np.int8))
    wp = e.cached_probs(out, f'cache/{name}/window_probabilities', identity,
        lambda: e.encoder_probabilities(encoder, embeddings), expected_rows=len(cohort['frame']))
    for method in ('mean', 'vote', 'ewma'):
        predictions.append(e.aggregate_probabilities(wp, cohort['indices'], method,
            protocol['frozen_selection']['ewma_alpha']).argmax(1).astype(np.int8))
    p = np.stack(predictions)
    e.write_array(folder / 'predictions.npz', predictions=p, y=cohort['y'],
        sequence_id=cohort['meta'].sequence_id.to_numpy(),
        families=np.array([f for f, s in e.RUNS]), seeds=np.array([s if s is not None else -1 for f, s in e.RUNS]))
    rows = [dict(realization=name, family=f, seed=s, **e.metrics(cohort['y'], pred))
            for (f, s), pred in zip(e.RUNS, p)]
    e.write_csv(folder / 'metrics_by_run.csv', rows)
    e.write_completion(folder, 'complete.json', contract,
        dict(realization=name, records=len(cohort['frame']), endpoints=len(cohort['y']), runs=len(e.RUNS)),
        ['predictions.npz', 'metrics_by_run.csv'])


def realization_statistics(y, episodes, predictions, seed, n_resamples=N_BOOTSTRAP, batch=100, progress=True):
    p = np.asarray(predictions)
    e.require(p.shape == (len(e.RUNS), len(y)) and set(np.unique(y)) == set(range(6)), 'Incomplete fresh prediction population')
    counts, strata = e.episode_counts(y, episodes, p[None])
    point = e.family_scores(e.macro_f1(counts.sum(0)))[0]
    sampled = e.family_scores(e.bootstrap_scores(counts, strata, n_resamples, seed, batch, progress))[:, 0]
    return point, sampled, counts, strata


def summarize_realizations(points, samples, selected_pool):
    points = np.asarray(points); samples = np.asarray(samples)
    e.require(points.shape == (3, len(e.FAMILIES)) and samples.ndim == 3
              and samples.shape[1:] == points.shape and np.isfinite(samples).all(), 'Need three complete realizations')
    means = points.mean(0); boot = samples.mean(1); summary = []; comparisons = []
    for fi, family in enumerate(e.FAMILIES):
        lo, hi = np.quantile(boot[:, fi], [.025, .975], method='linear')
        summary.append(dict(family=family, fresh_mean_macro_f1=float(means[fi]),
            between_realization_sd=float(points[:, fi].std(ddof=1)), min_realization_f1=float(points[:, fi].min()),
            max_realization_f1=float(points[:, fi].max()), conditional_ci95_low=float(lo), conditional_ci95_high=float(hi),
            realizations=3, head_seeds=10 if family in e.TRAINED else 1))
        if fi: comparisons.append(e.interval_row('TST', family, means[0]-means[fi], boot[:, 0]-boot[:, fi], family == selected_pool))
    return summary, comparisons


def finish_statistics(out, protocol, previous):
    folder = out / 'statistics'; folder.mkdir(exist_ok=True)
    contract = e.object_hash(dict(protocol=protocol['identity'], stage='statistics'))
    if e.read_completion(folder, 'complete.json', contract) is not None: return
    points = []; samples = []; by_realization = []; perclass = []; diagnostics = []
    for spec in protocol['realizations']:
        name = spec['name']; cohort = load_fresh(out / 'generation' / name)
        with np.load(out / 'evaluation' / name / 'predictions.npz', allow_pickle=False) as z:
            p = z['predictions']
            e.require(np.array_equal(z['y'], cohort['y']) and np.array_equal(z['sequence_id'], cohort['meta'].sequence_id)
                      and np.array_equal(z['families'], [f for f, s in e.RUNS])
                      and np.array_equal(z['seeds'], [s if s is not None else -1 for f, s in e.RUNS]), 'Fresh predictions do not match endpoints/run order')
        sub = folder / name; sub.mkdir(exist_ok=True)
        sc = e.object_hash(dict(parent=contract, realization=name))
        if e.read_completion(sub, 'complete.json', sc) is None:
            point, boot, counts, strata = realization_statistics(cohort['y'], cohort['meta'].episode_id.to_numpy(),
                p, spec['bootstrap_seed'], n_resamples=N_BOOTSTRAP)
            e.write_array(sub / 'bootstrap.npz', point=point, family_scores=boot)
            e.write_array(sub / 'episode_confusions.npz', counts=counts, class_strata=strata,
                          episodes=np.array(sorted(cohort['meta'].episode_id.unique()), dtype=str))
            e.write_completion(sub, 'complete.json', sc, dict(draws=N_BOOTSTRAP, seed=spec['bootstrap_seed']),
                               ['bootstrap.npz', 'episode_confusions.npz'])
        with np.load(sub / 'bootstrap.npz', allow_pickle=False) as z: point = z['point']; boot = z['family_scores']
        with np.load(sub / 'episode_confusions.npz', allow_pickle=False) as z: cms = z['counts'].sum(0)[0]
        points.append(point); samples.append(boot); run_scores = e.macro_f1(cms)
        for fi, family in enumerate(e.FAMILIES):
            ii = [i for i, (f, s) in enumerate(e.RUNS) if f == family]
            by_realization.append(dict(realization=name, family=family, macro_f1=float(point[fi]),
                head_seed_sd=float(run_scores[ii].std(ddof=1)) if len(ii) > 1 else None))
            cm = cms[ii]; den = cm.sum(-1)+cm.sum(-2); tp = np.diagonal(cm, axis1=-2, axis2=-1)
            cf = np.divide(2.*tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(0)
            for k, label in enumerate(e.LABELS): perclass.append(dict(realization=name, family=family, label=label, mean_seed_f1=float(cf[k])))
            # Descriptive diagnosis, no stratum-dependent fitting, filtering or selection.
            for stratum in ('current_changed', 'history_only_changed', 'no_observable_change'):
                for k, label in enumerate(e.LABELS):
                    mask = (cohort['effects'].audit_effect_stratum.to_numpy() == stratum) & (cohort['y'] == k)
                    n = int(mask.sum())
                    diagnostics.append(dict(realization=name, family=family, label=label, effect_stratum=stratum, endpoints=n,
                        mean_seed_accuracy=float((p[ii][:, mask] == k).mean()) if n else None,
                        mean_seed_predicted_NORMAL_fraction=float((p[ii][:, mask] == 0).mean()) if n else None))
    summary, comparisons = summarize_realizations(np.stack(points), np.stack(samples, axis=1), protocol['frozen_selection']['selected_pool'])
    for row in summary:
        row['original_clean_macro_f1'] = float(previous['clean_scores'][row['family']])
        row['fresh_minus_original_descriptive'] = row['fresh_mean_macro_f1']-row['original_clean_macro_f1']
    outputs = {'model_summary.csv': summary, 'paired_comparisons_fresh.csv': comparisons,
        'by_realization.csv': by_realization, 'per_class_f1.csv': perclass,
        'projection_change_diagnostics.csv': diagnostics}
    for filename, rows in outputs.items(): e.write_csv(folder / filename, rows)
    e.write_array(folder / 'bootstrap_distribution.npz', family_scores=np.stack(samples, axis=1),
        realizations=np.array([s['name'] for s in protocol['realizations']]), families=np.array(e.FAMILIES))
    files = [p.relative_to(folder).as_posix() for p in sorted(folder.rglob('*')) if p.is_file() and p != folder / 'complete.json']
    e.write_completion(folder, 'complete.json', contract, dict(realizations=3, draws=N_BOOTSTRAP), files)


def verify_completion_tree(folder):
    for path in sorted(folder.rglob('*complete.json')):
        record = e.read_json(path)
        e.read_completion(path.parent, path.name, record['contract'])


def package_review(out, manifest):
    archive = out.parent / (out.name + '_review.zip')
    wanted = manifest['review_files'] + ['run_manifest.json']
    if archive.exists():
        try:
            import hashlib
            with zipfile.ZipFile(archive) as z:
                valid = len(z.namelist()) == len(wanted) and set(z.namelist()) == set(wanted) and z.testzip() is None
                valid = valid and all(hashlib.sha256(z.read(n)).hexdigest() == e.digest(out / n) for n in wanted)
            if valid: return archive
        except (OSError, zipfile.BadZipFile, KeyError): pass
    temp = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in wanted: z.write(e.safe_artifact(out, name), name)
    with zipfile.ZipFile(temp) as z: e.require(z.testzip() is None, 'Review archive failed CRC')
    e.replace_atomic(temp, archive)
    return archive


def finish_run(out, identity):
    path = out / 'run_manifest.json'
    if path.exists():
        manifest = e.read_json(path)
        e.require(manifest.get('status') == 'FRESH_REALIZATIONS_COMPLETE' and manifest.get('protocol_identity') == identity,
                  'Wrong completed Step 48 manifest')
        body = {k: v for k, v in manifest.items() if k != 'identity'}
        e.require(manifest.get('identity') == e.object_hash(body), 'Step 48 manifest identity changed')
        for name, sha in manifest['output_sha256'].items():
            payload = e.safe_artifact(out, name)
            e.require(payload.is_file() and e.digest(payload) == sha, 'Completed output changed/missing: ' + name)
    else:
        verify_completion_tree(out)
        files = [p.relative_to(out).as_posix() for p in sorted(out.rglob('*'))
                 if p.is_file() and p.name != '.run.lock' and not p.name.endswith('.tmp')]
        review = [n for n in files if (not n.startswith(('cache/', 'generation/'))
            or n.endswith(('complete.json', 'generation_audit.json', 'episode_split_manifest.csv',
                           'evaluation_endpoints_L10.csv', 'evaluation_context_observable_effects_L10.csv')))]
        manifest = dict(status='FRESH_REALIZATIONS_COMPLETE', protocol_identity=identity, completed_at_utc=e.utc(),
            realizations=3, trained_heads=50, deterministic_controls=3, training=False, test_used_for_selection=False,
            output_sha256={n: e.digest(out / n) for n in files}, review_files=review,
            review_archive_omits='Large generation and inference caches stay on PC; ALL retained files are authenticated here.')
        manifest['identity'] = e.object_hash(manifest)
        e.write_json(path, manifest)
    return package_review(out, manifest)


def execute(root, out, protocol, context, data, previous, makers, splitter, code_paths, generate_only=False):
    protocol = dict(protocol); protocol['identity'] = e.object_hash(protocol)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'protocol.json').exists():
        e.require(not any(p.name != '.run.lock' for p in out.iterdir()), 'Output has foreign files without a protocol; use an empty --output-root')
    e.save_matching_json(out / 'protocol.json', protocol)
    identity = protocol['identity']
    if (out / 'run_manifest.json').exists(): return finish_run(out, identity)
    e.save_matching_json(out / 'input_audit.json', context['observed'])
    e.save_matching_json(out / 'validation_selection.json', previous['selection'])
    source = out / 'source'; source.mkdir(exist_ok=True)
    paths = [Path(__file__), Path(e.__file__), context['helper_path'], *context['code_paths'].values(), *code_paths.values()]
    for p in paths:
        target = source / p.name
        if target.exists(): e.require(e.digest(target) == e.digest(p), 'Copied source changed: ' + p.name)
        else: shutil.copyfile(p, target)
    if generate_only:
        generate_all(root, out, protocol, makers, splitter)
        context['helper'].recheck_inputs(context['observed'])
        return None
    runtime = e.evaluation_runtime(context)
    e.require(runtime == protocol['expected_inference_runtime'],
              'Inference runtime differs from completed Step 47; use the same environment/device/settings')
    e.save_matching_json(out / 'inference_runtime.json', runtime)
    modules = e.import_models(context)
    for family, seed in e.RUNS[:50]:
        checkpoint, _ = e.paths_for(family, seed, context)
        e.validate_checkpoint(e.read_checkpoint(checkpoint), family, seed, context)
    tokenizer, encoder = e.load_encoder(root, context)
    canary = replay_canaries(root, out, identity, context, data, modules, tokenizer, encoder)
    e.save_matching_json(out / 'canary_audit.json', canary)
    generate_all(root, out, protocol, makers, splitter)
    for spec in protocol['realizations']:
        evaluate_realization(root, out, spec, protocol, context, tokenizer, encoder, modules)
    del encoder; gc.collect()
    context['helper'].recheck_inputs(context['observed'])
    e.require(e.digest(Path(__file__)) == protocol['script_sha256'], 'Step 48 source changed during run')
    finish_statistics(out, protocol, previous)
    return finish_run(out, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=BASE)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--step47-code', type=Path)
    parser.add_argument('--step47-root', type=Path)
    parser.add_argument('--maker-code-dir', type=Path, help='Folder with current 1--6 and BOTH shared modules; default project root')
    parser.add_argument('--maker-data-dir', type=Path, help='Original maker learning/raw CSVs; default project-root/dataset')
    parser.add_argument('--step15-code', type=Path)
    parser.add_argument('--tst-root', type=Path)
    parser.add_argument('--pooling-root', type=Path)
    parser.add_argument('--lstm-root', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--generate-only', action='store_true')
    args = parser.parse_args(argv); root = args.project_root.resolve()
    global e
    e, engine_path = load_engine(root, args.step47_code)
    print('Step 48: current shared-nominal generators; frozen original-validation settings; no retraining.', flush=True)
    context = e.audit_sources(root, args.step15_code, args.tst_root, args.pooling_root, args.lstm_root)
    context['helper'].track_file(engine_path, context['observed'])
    previous = audit_step47(root, (args.step47_root or root / e.OUT_REL).resolve(), context, engine_path)
    makers, splitter, code_paths = load_generators(root, (args.maker_code_dir or root).resolve(),
        context['build'], context['observed'], context['helper'])
    sources = audit_original_makers(root, (args.maker_data_dir or root / 'dataset').resolve(), context, splitter)
    data = e.load_cohorts(root, context)
    protocol = make_protocol(context, previous, code_paths, sources)
    context['helper'].recheck_inputs(context['observed'])
    if args.preflight_only:
        print('PREFLIGHT PASS. No generation/model loading/inference. Original numerical/generator replay remains for execution.', flush=True)
        return 0
    out = (args.output_root or root / OUT_REL).resolve()
    e.require(not out.exists() or out.is_dir(), 'Output root must be a directory')
    protected = [root / e.DATA_REL, root / e.ENC_REL, Path(previous['folder']),
                 Path(sources[0]['learning']).parent, *[Path(h['checkpoint']).parent for h in context['heads'].values()]]
    validate_output_location(out, protected)
    out.mkdir(parents=True, exist_ok=True)
    with e.RunLock(out / '.run.lock'):
        archive = execute(root, out, protocol, context, data, previous, makers, splitter, code_paths, args.generate_only)
    if archive is None:
        print('GENERATION ONLY COMPLETE. No model results. Rerun without --generate-only for frozen-model evaluation.', flush=True)
    else: print(f'COMPLETE. Review archive: {archive}', flush=True)
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print(f'\nSTOPPED: {type(exc).__name__}: {exc}\nResolve the cause and rerun the same command; completed stages are verified and reused.', file=sys.stderr, flush=True)
        raise
