#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 49 supplement: frozen A models under shared-nominal parameter stress.

Run: python "49.generator_b_transfer_evaluation.py"
Use the current makers 1--6, shared_nominal.py, shared_context_v4.py, and Steps
11/13/14/15/47/48 beside this script. Step48's completed run is not required.
Existing completed Step16 v1/v3 and Step47 outputs/checkpoints are reused.

Four fixed regimes use the ORIGINAL raw makers and unmodified common mapper.
Benign-condition odds and nominal receiver dispersion increase; actual attack
severity/event parameters decrease. No observable tokens are overwritten.
Reference uses the original generator settings with new shared paired backgrounds;
it is not the original A test set. All quiet/unobserved scenario labels remain.

New output: outputs/shared_nominal_v2/49_parameter_stress/run_parameter_stress_v1
Old Generator-B outputs are protected. No neural retraining, threshold retuning,
score-based regime selection, or external-validity claim. Whole common nominal
episodes (all six class replicas, all models, all four regimes) are bootstrapped.

--preflight-only checks source provenance and original Rule16 replay.
--generate-only builds and validates all regimes without loading neural models.
Rerun the same command to reuse verified completed stages. Do not edit parameters
inside an existing run. Raw primitive CSVs, projection audits, paired endpoints,
predictions and statistics are included in the automatically produced review ZIP.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT_REL = 'outputs/shared_nominal_v2/49_parameter_stress/run_parameter_stress_v1'
VERSION = 'shared_nominal_A_parameter_stress_v1'
GENERATOR_VERSION = 'original_maker_parameter_interventions_v1'
SEED = 2026101401
BLOCKS = 128
LENGTH = 64
DRAWS = 20000
BOOTSTRAP_SEED = 2026101402
CONDITIONS = ('P_reference', 'P_mild', 'P_moderate', 'P_severe')
PARAMETERS = {
 'P_reference': dict(benign_odds_multiplier=1., receiver_std_multiplier=1., severity_scale=1., event_retention=1.),
 'P_mild': dict(benign_odds_multiplier=2., receiver_std_multiplier=1.5, severity_scale=.8, event_retention=.8),
 'P_moderate': dict(benign_odds_multiplier=3., receiver_std_multiplier=2., severity_scale=.6, event_retention=.6),
 'P_severe': dict(benign_odds_multiplier=4., receiver_std_multiplier=3., severity_scale=.4, event_retention=.4)}
_SEEDS = tuple(int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(SEED).spawn(8))
MAKER_SEEDS, BACKGROUND_SEED, PERTURBATION_SEED = _SEEDS[:6], _SEEDS[6], _SEEDS[7]

FIELDS = ('observation_stage','path_type','direction','source_role','destination_role',
 'protocol_context','operational_phase','message_type','service_context','command_intent',
 'payload_or_parameter_context','sequence_freshness_status','control_security_state',
 'signal_quality_state','tracking_stability','navigation_consistency','timing_consistency',
 'decoder_status','command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
META = ('scenario_id','episode_id','episode_type','episode_step','window_id','timestamp','label')
CONTEXT_FIELDS = FIELDS[:7]
LABELS = ('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING','UNAUTHORIZED_TELECOMMAND_INJECTION',
 'TELECOMMAND_REPLAY','GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
FAMILIES = ('TST','Mean_MLP','Max_MLP','MeanMax_MLP','LSTM','ProbabilityMean','MajorityVote','EWMA','Rule_L10')
RUNS = tuple((f, s) for f in FAMILIES[:5] for s in range(20260610, 20260620)) + tuple((f, None) for f in FAMILIES[5:])
# Bound to the audited shared-nominal source snapshot used by the reviewed Step48 run.
GENERATOR_SOURCE_SHA256 = {'1': 'f43a6d6b43dc644125e88f44406a4994a218ca02c9c27b17e76c8c0c4f41620f', '2': '991990fc1398042967de14c08c419f6a95e73e349dfef855a7df071c6324ad49', '3': '41bb6e6785bfe0b61bc771dc94f0d7a7756fdfab03c9c018029c7f60afd8e2fd', '4': '1b8e2696cca74acf8300c4e042ec9a3315a48751d4b3ecf6adf96d652fd9bd69', '5': '0f8f8b5dc873db22507d8dc938b1f2ba9ed211b4f73be22ff7e298a3080359bc', '6': '17b69eb554d8417f145b7bbbdd07b030410fc8b3a1e20b9688e6bbe09015f1af', 'shared_nominal': 'adfe6e2ea4268287efa593fb63e16ca84bc2dddc1bf2d22cd95ad360ca17d71d', 'shared_context_v4': 'b228affd6dbcd22c6a68143e5640b0dafbe5528bd77a52d420f1fcf066eea05c'}
makers = None
e = m48 = context_module = nominal = None


def require(ok, message):
    if not ok: raise ValueError(message)


def resolve_numbered(root, number, explicit=None):
    import hashlib
    candidates = [Path(explicit)] if explicit else [p for p in root.glob('*.py') if re.match(r'^'+str(number)+r'[. _-]',p.name)]
    unique = {hashlib.sha256(p.read_bytes()).hexdigest(): p.resolve() for p in candidates if p.is_file()}
    require(len(unique) == 1, f'Expected one current Step {number} script in {root}, found {len(unique)} versions')
    return next(iter(unique.values()))


def import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies(root, args):
    global e, m48, context_module, nominal
    path48 = resolve_numbered(root, 48, args.step48_code)
    m48 = import_script(path48, 'step49_current48')
    require(getattr(m48, 'VERSION', None) == 'shared_nominal_fresh_realizations_v2', 'Use the revised Step 48')
    e, path47 = m48.load_engine(root, args.step47_code); m48.e = e
    require(e.FIELDS == FIELDS and e.META == META and e.LABELS == LABELS and e.RUNS == RUNS[:53], 'Current model schema/run order differs')
    shared = (args.shared_code_dir or root).resolve()
    paths = {'48': path48, '47': path47}
    for name in ('shared_context_v4', 'shared_nominal'):
        path = shared / (name+'.py'); require(path.is_file(), 'Missing shared dependency: '+str(path))
        paths[name] = path.resolve(); import_script(path, name)
    context_module = sys.modules['shared_context_v4']; nominal = sys.modules['shared_nominal']
    require(tuple(nominal.OBSERVABLE_FIELDS) == FIELDS and tuple(nominal.METADATA_FIELDS) == META
        and nominal.CODE_VERSION == '2.0-shared-nominal-substrate'
        and nominal.shared_context_module is context_module, 'Use the current shared nominal/context files')
    return paths


def generator_protocol():
    return dict(version=GENERATOR_VERSION, root_seed=SEED, rows_per_class=BLOCKS*LENGTH,
        maker_seeds=list(MAKER_SEEDS), background_seed=BACKGROUND_SEED, perturbation_seed=PERTURBATION_SEED,
        window_seconds=1, conditions=PARAMETERS, classes=list(LABELS),
        observation_mapper='Unmodified shared_nominal.normalize_raw_to_observable for actual AND nominal raw states.',
        benign='Multiply non-NOMINAL versus NOMINAL episode-condition odds; retain conditional benign-type weights. '
            'Insert extra benign events only into originally NOMINAL episodes using existing event/phase/command-mode functions.',
        receiver_dispersion='Multiply std in existing physical NORMAL_RAW_RANGES; means and clipping bounds unchanged.',
        physical_attack='RF/GNSS: stochastic rounding of the original ordinal severity rank times severity_scale, '
            'plus independent event retention; original raw effects and normalization remain unchanged.',
        command_attack='Injection: retain each originally sampled attempt window with event_retention probability. '
            'Replay: scale existing episode_recurrence_strength. Ground: scale attempt_strength and drift_increment. '
            'No scaling of audit-only severity names or direct edits of command/check/outcome tokens.',
        pairing='One shared nominal timeline across six classes and all regimes. Bootstrap units are actual '
            'nominal episodes, each containing all six class replicas and all four regimes; not 64-row allocation chunks.',
        truth='Original scenario/episode labels including quiet, unobserved and recovery windows; no row filtering.',
        scope='Retrospective within-generator parameter sensitivity; not external validation or a new Generator B.',
        severity='Joint parameter stress; no monotonic F1 assumption or identification of individual factor effects.',
        raw_audit='Per-class compressed raw CSVs retain primitives and paired nominal primitives for mapper replay.')


def _receiver_fields():
    return tuple(k for k in nominal.NORMAL_RAW_RANGES if k.startswith(('ttc_','isl_','iot_','uav_','gnss_'))
                 or k in ('position_jump_m','velocity_jump_mps','pseudorange_residual_m','time_residual_ns','clock_drift_ppm'))


def stress_nominal_timeline(timeline, multiplier):
    """Change the existing condition-selection odds without changing context or episode boundaries."""
    if multiplier == 1:
        return timeline
    rng = np.random.default_rng(np.random.SeedSequence([PERTURBATION_SEED, 0x42454E]))
    result = [dict(row) for row in timeline]
    starts = [i for i, row in enumerate(timeline) if row['episode_step'] == 0]
    for lo, hi in zip(starts, starts[1:]+[len(timeline)]):
        first = timeline[lo]
        profile = nominal.NORMAL_CONDITION_PROFILES[first['episode_type']]
        weights = np.asarray(profile['probs'], dtype=float); weights /= weights.sum()
        names = np.asarray(profile['conditions']); non_nominal = names != 'NOMINAL'
        p = float(weights[non_nominal].sum())
        q = multiplier*p/(1-p+multiplier*p)
        # Draw for EVERY episode, so higher multipliers use nested insertion gates.
        u = rng.random(); choice_u = rng.random(); event_seed = int(rng.integers(0, 2**32))
        if first['normal_condition'] != 'NOMINAL' or p <= 0 or p >= 1 or u >= (q-p)/(1-p):
            continue
        cdf = np.cumsum(weights[non_nominal]/p)
        condition = str(names[non_nominal][min(int(np.searchsorted(cdf, choice_u)), len(cdf)-1)])
        token = nominal._RANDOM_SOURCE.set(np.random.default_rng(event_seed))
        try:
            plan = nominal.sample_benign_event_plan(condition, first['episode_length'])
            for i in range(lo, hi):
                row = result[i]; step = row['episode_step']
                phase = nominal.get_benign_condition_phase(step=step, duration=row['episode_length'],
                    condition=condition, event_start=plan['benign_event_start'], event_duration=plan['benign_event_duration'])
                severity = nominal.phase_to_benign_severity(condition, phase)
                active = int(plan['benign_event_start'] <= step <= plan['benign_event_end'])
                row.update(plan, normal_condition=condition, condition_phase=phase, condition_severity=severity,
                    benign_event_active=active,
                    benign_event_local_step=step-plan['benign_event_start'] if active else -1)
                row['episode_command_mode'] = (nominal.choose_normal_command_mode(condition if severity != 'NONE' else 'NOMINAL')
                    if nominal.is_command_context(row) else 'NO_COMMAND_CONTEXT')
        finally:
            nominal._RANDOM_SOURCE.reset(token)
    return result


@contextmanager
def parameter_context(condition, maker, rows):
    """Scoped in-memory parameter interventions; all functions/settings restored even on failure."""
    p = PARAMETERS[condition]; original_timeline = nominal._build_nominal_timeline
    original_ranges = nominal.NORMAL_RAW_RANGES
    raw_name = getattr(maker, 'RAW_GENERATOR', None) if maker.MAKER_NUMBER != 1 else None
    original_raw = getattr(maker, raw_name) if raw_name else None
    original_attempt = getattr(maker, 'sample_attempt_count', None)
    original_rng = getattr(maker, 'RNG', None)
    number = maker.MAKER_NUMBER
    draws = np.random.default_rng(np.random.SeedSequence([PERTURBATION_SEED, number, 0x415454])).random((rows, 2))
    attempt_rng = np.random.default_rng(np.random.SeedSequence([PERTURBATION_SEED, number, 0x494E4A]))
    def timeline_builder(n):
        return stress_nominal_timeline(original_timeline(n), p['benign_odds_multiplier'])
    def raw_generator(window_id, timestamp, episode, **options):
        intervention = dict(episode)
        if number in (2,3):
            key, levels = ('jamming_severity', ('NONE','MILD','MODERATE','SEVERE','OUTAGE')) if number == 2 else (
                'spoofing_severity', ('NONE','SUBTLE','MODERATE','SEVERE'))
            rank = levels.index(episode[key]); scaled = rank*p['severity_scale']
            reduced = int(np.floor(scaled)) + int(draws[window_id, 0] < scaled-np.floor(scaled))
            intervention[key] = levels[reduced] if draws[window_id, 1] < p['event_retention'] else 'NONE'
        elif number == 5:
            intervention['episode_recurrence_strength'] *= p['event_retention']
        elif number == 6:
            intervention['attempt_strength'] *= p['event_retention']
            intervention['drift_increment'] *= p['severity_scale']
        return original_raw(window_id, timestamp, intervention, **options)
    def injection_attempt(episode):
        count = original_attempt(episode)
        keep = attempt_rng.random() < p['event_retention']
        return count if keep else 0
    try:
        if condition != CONDITIONS[0]:
            nominal._build_nominal_timeline = timeline_builder
            nominal.NORMAL_RAW_RANGES = {k:dict(v, std=v['std']*p['receiver_std_multiplier'])
                if k in _receiver_fields() else dict(v) for k,v in original_ranges.items()}
            if number != 1:
                setattr(maker, raw_name, raw_generator)
            if number == 4:
                maker.sample_attempt_count = injection_attempt
        yield
    finally:
        nominal._build_nominal_timeline = original_timeline
        nominal.NORMAL_RAW_RANGES = original_ranges
        if raw_name:
            setattr(maker, raw_name, original_raw)
        if original_attempt is not None:
            maker.sample_attempt_count = original_attempt
        if original_rng is not None:
            maker.RNG = original_rng


def generate_b(condition, blocks=BLOCKS, attacks_enabled=True, raw_folder=None):
    """Legacy function name retained; blocks controls 64-row allocation units, NOT bootstrap clusters."""
    require(condition in CONDITIONS and type(blocks) is int and blocks > 0, 'Invalid parameter-stress request')
    require(type(attacks_enabled) is bool and makers is not None, 'Load the current six makers first')
    n = blocks*LENGTH; frames=[]; raw_parts=[]; truth=[]; baseline0=None; ids0=None
    fields = _receiver_fields()
    for number, maker in enumerate(makers, 1):
        builder = getattr(maker, m48.BUILD_FUNCTIONS[number-1])
        with parameter_context(condition, maker, n):
            full = builder(num_rows=n, window_seconds=1, seed=MAKER_SEEDS[number-1],
                start_time=datetime(2026, 4, 1), return_raw=True,
                background_seed=BACKGROUND_SEED, attacks_enabled=attacks_enabled)
        # Validates command causality, target gating, stateful receiver checks,
        # raw-to-token reconstruction and paired nominal reconstruction.
        maker.validate_raw_audit_dataset(full, expected_rows=n, expected_window_seconds=1)
        if condition == CONDITIONS[0]:
            canary = builder(num_rows=min(n,128), window_seconds=1, seed=MAKER_SEEDS[number-1],
                start_time=datetime(2026,4,1), return_raw=False,
                background_seed=BACKGROUND_SEED, attacks_enabled=attacks_enabled)
            pd.testing.assert_frame_equal(full[list(META+FIELDS)].iloc[:len(canary)].reset_index(drop=True), canary, check_exact=True)
        if raw_folder is not None:
            path=Path(raw_folder)/f'raw_class_{number}.csv.gz';temp=path.with_name(path.name+'.tmp')
            full.to_csv(temp,index=False,encoding='utf-8',compression={'method':'gzip','mtime':0})
            temp.replace(path)
        baseline=full[['baseline_observable_'+f for f in FIELDS]].to_numpy(dtype=str)
        if baseline0 is None:
            baseline0=baseline.copy(); ids0=pd.factorize(full.episode_id, sort=False)[0]
        else:
            require(np.array_equal(baseline,baseline0), 'Class-specific nominal projection')
            require(np.array_equal(pd.factorize(full.episode_id,sort=False)[0],ids0), 'Class-specific episode boundaries')
        frame=full[list(META+FIELDS)].copy()
        frame['episode_id']='P_'+frame.episode_id.astype(str)
        frame['scenario_id']='P_'+frame.scenario_id.astype(str)
        for episode, positions in frame.groupby('episode_id',sort=False).indices.items():
            ix=np.asarray(positions); truth.append(dict(episode_id=episode,label=maker.CLASS_LABEL,
                cause_id=number-1,block_id=int(ids0[ix[0]]),windows=len(ix)))
        attempts=next((full[k].to_numpy(dtype=np.int64) for k in (
            'raw_attempted_injected_command_count','raw_attempted_replayed_command_count','raw_attempted_compromise_command_count')
            if k in full),full.raw_attack_active.to_numpy(dtype=np.int64))
        raw_parts.append(dict(block_id=ids0.copy(), baseline_observables=baseline,
            attack_active=full.raw_attack_active.to_numpy(dtype=np.int64),
            persistent_effect=full.raw_persistent_effect_present.to_numpy(dtype=np.int64),
            attacks_enabled=full.raw_attacks_enabled.to_numpy(dtype=bool), attack_attempt_count=attempts,
            nominal_benign=full.baseline_benign_impairment_flag.to_numpy(dtype=np.int64),
            baseline_receiver=full[['baseline_'+k for k in fields]].to_numpy(dtype=float)))
        frames.append(frame)
        print(f'  {condition}: maker {number}/6 validated ({n:,} rows)',flush=True)
    frame=pd.concat(frames,ignore_index=True);raw={k:np.concatenate([v[k] for v in raw_parts]) for k in raw_parts[0]}
    truth=pd.DataFrame(truth);validate_b(frame,raw,truth)
    return frame,raw,truth


def validate_b(frame, raw, truth):
    require(list(frame)==list(META+FIELDS) and not frame.isna().any().any(), 'Parameter-stress canonical schema differs')
    require(len(frame)>0 and len(frame)%6==0, 'Incomplete six-class cohort')
    n=len(frame)//6
    require(all(len(v)==len(frame) for v in raw.values()), 'Unaligned primitive audit arrays')
    require(np.array_equal(frame.label,np.repeat(np.asarray(LABELS),n)), 'Scenario labels changed')
    for field in FIELDS:
        require(set(frame[field]) <= set(nominal.OBSERVABLE_DOMAINS[field]) | {'UNK'}, 'Unsupported observable: '+field)
    ctx=frame[list(CONTEXT_FIELDS)].to_numpy().reshape(6,n,7)
    require(np.array_equal(ctx,np.broadcast_to(ctx[:1],ctx.shape)), 'Class-specific context')
    for key in ('baseline_observables','block_id','nominal_benign','baseline_receiver'):
        values=raw[key].reshape(6,n,*raw[key].shape[1:])
        require(np.array_equal(values,np.broadcast_to(values[:1],values.shape)), 'Class-specific shared background: '+key)
    require(np.array_equal(frame.loc[frame.label.eq('NORMAL'),list(FIELDS)].to_numpy(dtype=str),raw['baseline_observables'][:n]),
            'NORMAL differs from nominal counterfactual')
    if not raw['attacks_enabled'].any():
        require(np.array_equal(frame[list(FIELDS)].to_numpy(dtype=str),raw['baseline_observables']), 'Disabled attack leaves a projection change')
    require(not truth.episode_id.duplicated().any() and truth.episode_id.tolist()==frame.episode_id.drop_duplicates().tolist(),
            'Episode inventory differs')
    require(np.isin(raw['attack_active'],[0,1]).all() and np.isin(raw['persistent_effect'],[0,1]).all(), 'Invalid effect flags')


def window_effects(frame, raw):
    changes=frame[list(FIELDS)].to_numpy(dtype=str)!=raw['baseline_observables']
    audit=frame[list(META)].copy()
    audit['audit_observable_effect_present']=changes.any(1).astype(np.int64)
    audit['audit_observable_difference_count']=changes.sum(1).astype(np.int64)
    audit['audit_persistent_effect_present']=raw['persistent_effect']
    audit['audit_attack_active']=raw['attack_active']
    return audit


def generate_all(out, protocol):
    baseline=None;truth0=None;meta0=None;reports=[]
    for name in CONDITIONS:
        folder=out/'generation'/name;folder.mkdir(parents=True,exist_ok=True)
        contract=e.object_hash(dict(protocol=protocol['identity'],stage='generation',condition=name))
        done=e.read_completion(folder,'complete.json',contract)
        if done is None:
            print(f'Generate {name}: {BLOCKS*LENGTH:,} rows/class, unmodified common mapper',flush=True)
            frame,raw,truth=generate_b(name,raw_folder=folder)
            audit=window_effects(frame,raw);ix,y,meta,effects=m48.build_endpoints(frame,audit)
            blocks=pd.DataFrame(dict(sequence_id=meta.sequence_id,episode_id=meta.episode_id,block_id=raw['block_id'][ix[:,-1]]))
            e.write_csv(folder/'evaluation_records.csv',frame)
            e.write_csv(folder/'evaluation_window_observable_effects.csv',audit)
            e.write_csv(folder/'evaluation_endpoints_L10.csv',meta)
            e.write_csv(folder/'evaluation_context_observable_effects_L10.csv',pd.concat([meta,effects],axis=1))
            e.write_csv(folder/'endpoint_blocks.csv',blocks);e.write_csv(folder/'latent_episode_truth.csv',truth)
            e.write_array(folder/'sequences_L10.npz',indices=ix,y=y);e.write_array(folder/'measurement_audit.npz',**raw)
        else:
            cohort=m48.load_fresh(folder);frame=cohort['frame'];ix=cohort['indices'];y=cohort['y'];meta=cohort['meta'];effects=cohort['effects']
            with np.load(folder/'measurement_audit.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            truth=pd.read_csv(folder/'latent_episode_truth.csv',keep_default_na=False)
            validate_b(frame,raw,truth)
            m48.same_frame(window_effects(frame,raw),pd.read_csv(folder/'evaluation_window_observable_effects.csv',keep_default_na=False),
                           'Saved parameter-stress effects differ')
            blocks=pd.read_csv(folder/'endpoint_blocks.csv')
        require(np.array_equal(blocks.sequence_id,meta.sequence_id) and np.array_equal(blocks.episode_id,meta.episode_id)
            and np.array_equal(blocks.block_id,raw['block_id'][ix[:,-1]]), 'Endpoint block mapping differs')
        require(np.all(raw['block_id'][ix]==raw['block_id'][ix[:,-1]][:,None]), 'Sequence crosses paired episode boundary')
        if baseline is None:
            baseline=frame.copy();truth0=truth.copy();meta0=meta.copy()
        else:
            pd.testing.assert_frame_equal(truth,truth0,check_exact=True)
            pd.testing.assert_frame_equal(meta,meta0)
            m48.same_frame(frame[list(META+CONTEXT_FIELDS)],baseline[list(META+CONTEXT_FIELDS)],'Regime metadata/context differ')
        hashes=m48.row_fingerprints(frame);_,codes=np.unique(hashes[ix],axis=0,return_inverse=True)
        collision=pd.DataFrame(dict(code=codes,label=y)).groupby('code').label.transform('nunique').to_numpy()>1
        report=dict(condition=name,records=len(frame),episodes=int(frame.episode_id.nunique()),
            paired_episode_blocks=int(blocks.block_id.nunique()),endpoints=len(y),
            current_schema_categories_valid=True,mapper_reconstruction_exact=True,shared_counterfactual_background_exact=True,
            unmodified_reference_prefix_replayed=name==CONDITIONS[0],
            label_semantics='Original maker scenario/episode labels; no filtering or relabeling',
            changed_fraction_vs_reference={f:float((frame[f]!=baseline[f]).mean()) for f in FIELDS},
            token_counts={f:{str(k):int(v) for k,v in frame[f].value_counts().items()} for f in FIELDS},
            attack_active_windows_by_label={label:int(raw['attack_active'][frame.label.eq(label)].sum()) for label in LABELS},
            attack_attempts_by_label={label:int(raw['attack_attempt_count'][frame.label.eq(label)].sum()) for label in LABELS},
            nominal_benign_window_fraction=float(raw['nominal_benign'].mean()),
            receiver_fields=list(_receiver_fields()),baseline_receiver_sd=raw['baseline_receiver'].std(0).tolist(),
            cross_label_identical_L10_fraction=float(collision.mean()),
            collision_scope='64-bit vector fingerprints; descriptive only; no rows removed',
            effect_strata=effects.audit_effect_stratum.value_counts().to_dict())
        if done is None:
            e.write_json(folder/'generation_audit.json',report)
            files=['evaluation_records.csv','evaluation_window_observable_effects.csv','evaluation_endpoints_L10.csv',
                'evaluation_context_observable_effects_L10.csv','endpoint_blocks.csv','latent_episode_truth.csv',
                'sequences_L10.npz','measurement_audit.npz','generation_audit.json']+[f'raw_class_{i}.csv.gz' for i in range(1,7)]
            e.write_completion(folder,'complete.json',contract,report,files)
        else:
            require(done==report,'Recomputed generation audit differs')
        reports.append(report)
    e.save_matching_json(out/'generation_summary.json',dict(generator=generator_protocol(),conditions=reports,
        model_performance_used_for_generation=False))


















def rule_source_contract(record):
    """Accept the two recorded Step16 schemas; never substitute a different run."""
    old=record.get('dataset_contract');new=record.get('source_contract')
    if old is not None and new is not None:
        require(all(old.get(k)==new.get(k) for k in e.PROVENANCE_KEYS),
                'Conflicting Step16 dataset_contract/source_contract identities')
    source=new if new is not None else old
    require(isinstance(source,dict) and all(source.get(k) for k in e.PROVENANCE_KEYS),
            'Step16 manifest lacks a complete dataset/source contract')
    return source


def resolve_rule_run(root, explicit, source):
    if explicit is not None:
        p=Path(explicit).expanduser()
        return (p if p.is_absolute() else root/p).resolve()
    base=root/e.PREFIX/'16_observable_rule_baseline'
    candidates=[];rejected=[]
    for path in sorted(base.glob('run_*/run_manifest.json')):
        try:
            record=e.read_json(path)
            require(record.get('status')=='COMPLETE' and record.get('calibrate_only') is False,
                    'not a completed test run')
            contract=rule_source_contract(record)
            require(all(contract[k]==source[k] for k in e.PROVENANCE_KEYS),'different dataset/encoder/sequence lineage')
            candidates.append(path.parent)
        except (ValueError,KeyError,TypeError) as exc:
            rejected.append(path.parent.name+': '+str(exc))
    require(len(candidates)==1, f'Found {len(candidates)} completed current Step16 runs in {base}. '
            'Use --rule-run to identify the intended run; no score-based selection. '+
            ('Rejected: '+'; '.join(rejected[:5]) if rejected else ''))
    return candidates[0]


def saved_rule_engine(spec):
    """Restore only hash-bound inference definitions; no calibration is executed."""
    import ast
    from types import SimpleNamespace
    require(spec.get('version')=='shared_nominal_v2_semantic_rules_v1'
            and tuple(spec.get('available_fields',()))==FIELDS
            and tuple(spec.get('canonical_labels',()))==LABELS,'Legacy Rule16 schema differs')
    tokens=spec.get('membership_sets')
    require(isinstance(tokens,dict) and tokens,'Legacy rule membership sets missing')
    env=dict(np=np,pd=pd,require=require,FIELDS=FIELDS,LABELS=LABELS,ATTACKS=LABELS[1:],
             LENGTH=10,TOKENS=tokens,__builtins__=__builtins__)
    nodes=[]
    for name in ('window_evidence','aggregate_evidence','classify'):
        text=spec.get('implementation',{}).get(name)
        require(isinstance(text,str),'Legacy rule inference definition missing: '+name)
        tree=ast.parse(text)
        require(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
                and tree.body[0].name==name and not tree.body[0].decorator_list
                and not tree.body[0].args.defaults and not any(tree.body[0].args.kw_defaults),
                'Unexpected saved Rule16 inference definition: '+name)
        nodes.append(tree.body[0])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<verified Step16 rule_spec>','exec'),env)
    return SimpleNamespace(**{k:env[k] for k in ('window_evidence','aggregate_evidence','classify')})


def audit_rule_v1(root,folder,manifest,context,data,explicit_code=None):
    """Read the completed shared-nominal v1 record without recalibration/upgrading rules."""
    from datetime import datetime
    helper=context['helper'];observed=context['observed'];source=context['source_contract']
    legacy=rule_source_contract(manifest)
    require(manifest.get('calibrate_only') is False and manifest.get('test_opened_after_freeze') is True
            and all(legacy[k]==source[k] for k in e.PROVENANCE_KEYS),'Legacy Rule16 source/freeze differs')
    records=manifest['outputs'];hashes={r['file']:r['sha256'] for r in records}
    require(len(hashes)==len(records),'Duplicate Step16 output records')
    needed={'frozen_rule_config.json','protocol.json','rule_spec.json','results_summary.csv',
            'val_predictions_Rule_L10.csv','test_predictions_Rule_L10.csv'}
    require(needed<=set(hashes),'Incomplete Step16 outputs')
    helper.verify_selected(folder,hashes,list(hashes),observed)
    config=e.read_json(folder/'frozen_rule_config.json');protocol=e.read_json(folder/'protocol.json');spec=e.read_json(folder/'rule_spec.json')
    require(config.get('protocol_version')==protocol.get('version')==spec.get('version')=='shared_nominal_v2_semantic_rules_v1',
            'Unsupported legacy Step16 protocol')
    require(protocol.get('dataset_contract')==legacy and protocol.get('metadata_feature_use') is False
            and tuple(protocol.get('available_observable_fields',()))==FIELDS,'Legacy rule input schema differs')
    require(e.object_hash(protocol)==config['protocol_sha256']==manifest['protocol_sha256']
            and e.object_hash(spec)==config['rule_spec_sha256']==protocol['rule_spec_sha256']
            and config['script_sha256']==manifest['script_sha256']
            and hashes['frozen_rule_config.json']==manifest['frozen_config_sha256'],
            'Legacy Rule16 protocol/specification/config hash changed')
    require(config.get('label_to_id')==dict(zip(LABELS,range(6)))
            and config.get('test_inputs_read_before_freeze') is False,'Legacy Rule16 labels/freeze differ')
    expected={e.DATA_REL+'/build_manifest.json':source['observability_audit']['step7_build_manifest_sha256'],
              e.DATA_REL+'/splits/label_mapping.json':source['input_sha256']['splits/label_mapping.json'],
              e.SEQ_REL+'/sequence_build_report_index.json':source['step10_artifacts']['sequence_build_report_index.json']}
    for split in ('train','val','test'):
        count=context['build']['counts'][split]
        require(legacy.get('expected_counts',{}).get(split)==[count['rows'],count['episodes'],source['common_endpoint_report']['splits'][split]['common_endpoint_count']],
                'Legacy Rule16 population differs: '+split)
        expected[e.DATA_REL+'/splits/'+split+'_observable.csv']=source['input_sha256']['splits/'+split+'_observable.csv']
    for split in ('val','test'):
        name=split+'_window_metadata.csv';expected[e.EMB_REL+'/'+name]=source['step9_artifacts'][name]
        for suffix in ('sequence_metadata_L10.csv','sequences_L10.npz'):
            name=split+'_tst_'+suffix;expected[e.SEQ_REL+'/'+name]=source['step10_artifacts'][name]
    normalized=lambda value:re.sub('/+','/',str(value).replace('\\','/'))
    inputs=manifest['inputs'];matched=[]
    for relative,digest in expected.items():
        matches=[r for r in inputs if normalized(r.get('path','')).endswith('/'+relative)]
        require(len(matches)==1 and matches[0].get('sha256')==digest,'Legacy Rule16 recorded input differs: '+relative)
        helper.track_file(root/relative,observed,digest);matched.append(matches[0])
        if not Path(relative).name.startswith('test_'):
            require(config.get('calibration_input_sha256',{}).get(relative)==digest,
                    'Legacy Rule16 calibration input differs: '+relative)
    require(len(matched)==len(inputs),'Unexpected or duplicate legacy Rule16 input inventory')
    test_reads=[r for r in inputs if normalized(r['path']).split('/')[-1].startswith('test_')]
    require(bool(test_reads) and all(datetime.fromisoformat(r['read_at_utc'])>datetime.fromisoformat(config['frozen_at_utc']) for r in test_reads),
            'Rule16 recorded test access precedes saved threshold freeze')
    if explicit_code is not None:
        code=Path(explicit_code);code=code if code.is_absolute() else root/code
        helper.track_file(code,observed,manifest['script_sha256'])
    thresholds=config['models']['Rule_L10']['thresholds']
    require(len(thresholds)==5 and all(type(t) in (float,int) and np.isfinite(t) and t in spec['threshold_grid'] for t in thresholds),
            'Invalid frozen legacy rule thresholds')
    rule=saved_rule_engine(spec);summary=pd.read_csv(folder/'results_summary.csv');replay={}
    for split in ('val','test'):
        d=data[split];saved=pd.read_csv(folder/f'{split}_predictions_Rule_L10.csv',keep_default_na=False)
        require(len(saved)==len(d['meta']) and saved.sequence_id.is_unique,'Incomplete/duplicate original rule endpoints')
        saved=saved.sort_values('sequence_id').reset_index(drop=True)
        require(set(d['meta'])<=set(saved),'Legacy Rule16 endpoint metadata missing')
        m48.same_frame(saved[list(d['meta'])],d['meta'],'Legacy Rule16 endpoint population differs: '+split)
        if 'model' in saved:require(saved.model.eq('Rule_L10').all(),'Legacy saved model differs')
        scores=rule.aggregate_evidence(d['frame'][list(FIELDS)],d['indices'],10);pred,_=rule.classify(scores,thresholds)
        truth_cols=[k for k in ('y_true','true_label_id') if k in saved];pred_cols=[k for k in ('y_pred','pred_label_id') if k in saved]
        require(truth_cols and pred_cols,'Legacy Rule16 prediction IDs missing')
        target=saved[['evidence_'+label for label in LABELS[1:]]].to_numpy(float)
        require(all(np.array_equal(saved[k],d['y']) for k in truth_cols)
                and all(np.array_equal(saved[k],pred) for k in pred_cols)
                and np.allclose(scores,target,rtol=0,atol=1e-12),'Original Rule_L10 numerical replay failed: '+split)
        met=e.metrics(d['y'],pred);row=summary.loc[(summary.split==split)&(summary.model=='Rule_L10')]
        require(len(row)==1 and int(row.iloc[0].n_sequences)==len(pred)
                and np.isclose(float(row.iloc[0].macro_f1),met['macro_f1'],rtol=1e-10,atol=1e-12),'Legacy Rule16 summary differs')
        if split=='val':require(np.isclose(config['models']['Rule_L10']['validation_macro_f1'],met['macro_f1'],rtol=1e-10,atol=1e-12),'Legacy rule calibration score differs')
        replay[split]=dict(rows=len(pred),macro_f1=met['macro_f1'],max_abs_score_error=float(np.abs(scores-target).max()))
    print('Rule16: dataset_contract_v1 accepted; original val/test predictions reproduced; thresholds unchanged.',flush=True)
    return rule,dict(folder=str(folder),code=str(folder/'rule_spec.json'),code_sha256=hashes['rule_spec.json'],
        code_kind='hash-bound original inference functions in rule_spec.json',producer_script_sha256=manifest['script_sha256'],
        run_id=manifest.get('run_id',folder.name),contract_adapter='dataset_contract_v1',
        config_sha256=manifest['frozen_config_sha256'],thresholds=thresholds,original_replay=replay,
        original_test_macro_f1=replay['test']['macro_f1'])


def audit_rule(root, folder, context, data, explicit_code=None):
    helper=context['helper']; observed=context['observed']
    manifest=helper.load_json(folder/'run_manifest.json',observed)
    require(manifest.get('status')=='COMPLETE', 'Finish current Step16 (not calibration-only) before Step49')
    if 'dataset_contract' in manifest and 'source_contract' not in manifest:
        return audit_rule_v1(root,folder,manifest,context,data,explicit_code)
    path=e.resolve_code(root,16,manifest['script_sha256'],explicit_code)
    rule=e.import_script(path,'step49_current_rule16');helper.track_file(path,observed)
    require(rule.PROTOCOL_VERSION=='shared_nominal_semantic_rules_v3' and rule.FIELDS==FIELDS and rule.LABELS==LABELS,
            'Current shared-nominal Rule16 is required')
    records=manifest['outputs']; hashes={r['file']:r['sha256'] for r in records}
    require(len(hashes)==len(records), 'Duplicate Step16 output records')
    required={'frozen_rule_config.json','protocol.json','rule_spec.json','results_summary.csv',
              'val_predictions_Rule_L10.csv','test_predictions_Rule_L10.csv'}
    require(required<=set(hashes), 'Incomplete Step16 outputs')
    helper.verify_selected(folder,hashes,list(hashes),observed)
    config=e.read_json(folder/'frozen_rule_config.json');protocol=e.read_json(folder/'protocol.json')
    audit=[];current=rule.load_input_contract(root,audit)
    require(current==manifest.get('source_contract')==config.get('source_contract'), 'Rule16 source differs from current A dataset')
    require(all(current[k]==context['source_contract'][k] for k in e.PROVENANCE_KEYS), 'Rule/neural A provenance differs')
    for item in [*audit,*manifest['inputs']]:
        p=Path(item['path']);require(p.is_file() and e.digest(p)==item['sha256'], 'Rule16 input changed: '+str(p))
        observed[str(p.resolve())]=item['sha256']
    require(config['protocol_version']==rule.PROTOCOL_VERSION==protocol['version']
        and config['script_sha256']==manifest['script_sha256']
        and config['rule_spec_sha256']==rule.object_digest(rule.rule_specification())==rule.object_digest(e.read_json(folder/'rule_spec.json'))
        and config['protocol_sha256']==manifest['protocol_sha256']==rule.object_digest(protocol)
        and manifest['frozen_config_sha256']==hashes['frozen_rule_config.json'], 'Rule16 settings/specification changed')
    require(config['label_to_id']==dict(zip(LABELS,range(6)))
        and config.get('test_inputs_read_before_freeze') is False and manifest.get('test_opened_after_freeze') is True,
        'Rule16 class mapping or validation-before-test protocol differs')
    from datetime import datetime
    test_reads=[r for r in manifest['inputs'] if Path(r['path']).name.startswith('test_')]
    require(bool(test_reads) and all(datetime.fromisoformat(r['read_at_utc'])>datetime.fromisoformat(config['frozen_at_utc']) for r in test_reads),
        'Rule16 recorded test access precedes saved thresholds')
    thresholds=config['models']['Rule_L10']['thresholds']
    require(len(thresholds)==5 and all(type(t) in (float,int) and t in rule.GRID for t in thresholds), 'Invalid saved rule thresholds')
    summary=pd.read_csv(folder/'results_summary.csv');replay={}
    for split in ('val','test'):
        d=data[split];saved=pd.read_csv(folder/f'{split}_predictions_Rule_L10.csv',keep_default_na=False)
        require(not saved.sequence_id.duplicated().any(), 'Repeated original rule endpoint')
        saved=saved.sort_values('sequence_id').reset_index(drop=True)
        m48.same_frame(saved[list(d['meta'])],d['meta'], 'Rule16 endpoint population differs: '+split)
        for key in e.PROVENANCE_KEYS: require(saved[key].eq(context['source_contract'][key]).all(), 'Rule prediction provenance differs: '+key)
        require(saved.step16_run_id.eq(manifest['run_id']).all() and saved.rule_config_sha256.eq(manifest['frozen_config_sha256']).all(), 'Rule prediction run differs')
        scores=rule.aggregate_evidence(d['frame'],d['indices'],10);pred,_=rule.classify(scores,thresholds)
        target=saved[['evidence_'+label for label in LABELS[1:]]].to_numpy(float)
        require(np.array_equal(saved.y_true,d['y']) and np.array_equal(pred,saved.y_pred)
                and np.allclose(scores,target,rtol=0,atol=1e-12), 'Original Rule_L10 numerical replay failed: '+split)
        met=e.metrics(d['y'],pred);row=summary.loc[(summary.split==split)&(summary.model=='Rule_L10')]
        require(len(row)==1 and int(row.iloc[0].n_sequences)==len(pred)
                and np.isclose(float(row.iloc[0].macro_f1),met['macro_f1'],rtol=1e-10,atol=1e-12), 'Rule16 summary differs from predictions')
        if split=='val': require(np.isclose(config['models']['Rule_L10']['validation_macro_f1'],met['macro_f1'],rtol=1e-10,atol=1e-12), 'Rule calibration score differs')
        replay[split]=dict(rows=len(pred),macro_f1=met['macro_f1'],max_abs_score_error=float(np.abs(scores-target).max()))
    return rule,dict(folder=str(folder),code=str(path),code_sha256=e.digest(path),run_id=manifest['run_id'],
        config_sha256=manifest['frozen_config_sha256'],thresholds=thresholds,original_replay=replay,
        original_test_macro_f1=replay['test']['macro_f1'])


def make_protocol(context, previous, rule_info, paths):
    return dict(version=VERSION,script_sha256=e.digest(Path(__file__)),generator=generator_protocol(),
        source_contract=context['source_contract'],checkpoint_contracts=context['heads'],input_audit=context['observed'],
        code_sha256={k:e.digest(p) for k,p in paths.items()},source_step47_identity=previous['protocol']['identity'],
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        expected_inference_runtime=previous['protocol']['runtime'],inference=previous['protocol']['inference'],
        frozen_selection=previous['selection'],rule=rule_info,fields=list(FIELDS),families=list(FAMILIES),runs=[list(r) for r in RUNS],
        training=False,stress_used_for_selection=False,
        primary='P_moderate TST minus ORIGINAL A-validation-selected pooling family',
        statistics=dict(draws=DRAWS,seed=BOOTSTRAP_SEED,resampling_unit='whole common nominal episode, containing all six class replicas and paired across all models and regimes',
            estimator='mean of per-trained-head six-class Macro-F1; no pooled confusion matrices or probability ensemble',
            intervals='95% linear percentile conditional on the fixed paired nominal realization and trained A models; no generator-seed uncertainty',
            head_seed_resampling=False,comparisons=32,primary_contrasts=1,
            multiplicity='supplementary Bonferroni across four regimes x eight comparators',
            stress_changes='paired within parameter regimes; descriptive intervals',original_A_difference='unpaired/descriptive only',equivalence_claim=False),
        scope='Retrospective within-generator parameter sensitivity; reference uses new paired backgrounds, not the original A test cohort. No external validity or factor-specific causal effects.')





def audit_token_budget(out, tokenizer, protocol, context):
    folder=out/'cache/token_budget';folder.mkdir(parents=True,exist_ok=True)
    contract=e.object_hash(dict(protocol=protocol['identity'],stage='all_B_token_lengths'))
    done=e.read_completion(folder,'complete.json',contract)
    if done is not None:return done
    reports={}
    for name in CONDITIONS:
        frame=e.read_records(out/'generation'/name/'evaluation_records.csv');lengths=[]
        for start in range(0,len(frame),512):
            ids=tokenizer(e.serialize_frame(frame.iloc[start:start+512]),truncation=False,padding=False,return_attention_mask=False)['input_ids']
            require(len(ids)==min(512,len(frame)-start) and all(ids),'Tokenizer dropped B records')
            require(all(tokenizer.unk_token_id not in row for row in ids),'Actual tokenizer [UNK] in B input; literal field UNK is distinct')
            n=[len(row) for row in ids];require(max(n)<=context['max_length'],'B exceeds SAVED encoder token budget; do not silently truncate')
            lengths.extend(n)
        reports[name]=dict(rows=len(frame),minimum=int(min(lengths)),maximum=int(max(lengths)),median=float(np.median(lengths)),
            max_length=context['max_length'],truncated_rows=0)
    e.write_json(folder/'audit.json',reports);e.write_completion(folder,'complete.json',contract,reports,['audit.json'])
    return reports


def evaluate_condition(root,out,name,protocol,context,tokenizer,encoder,modules,rule,rule_info):
    # Reuse current Step48 neural path (same Step9 batch/FP32, original head AMP,
    # strict checkpoints and cache validation); append the unchanged Rule_L10.
    m48.evaluate_realization(root,out,dict(name=name),protocol,context,tokenizer,encoder,modules)
    folder=out/'evaluation'/name
    contract=e.object_hash(dict(protocol=protocol['identity'],stage='combined_rule_and_neural',condition=name))
    if e.read_completion(folder,'transfer_complete.json',contract) is not None:return
    cohort=m48.load_fresh(out/'generation'/name)
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
        require(np.array_equal(z['y'],cohort['y']) and np.array_equal(z['sequence_id'],cohort['meta'].sequence_id)
            and z['predictions'].shape==(53,len(cohort['y']))
            and np.array_equal(z['families'],[f for f,s in RUNS[:53]])
            and np.array_equal(z['seeds'],[s if s is not None else -1 for f,s in RUNS[:53]]),'Neural parameter-stress run grid differs')
        neural=z['predictions']
    scores=rule.aggregate_evidence(cohort['frame'],cohort['indices'],10)
    thresholds=np.asarray(rule_info['thresholds'],dtype=float);pred,aux=rule.classify(scores,thresholds)
    predictions=np.concatenate([neural,pred[None].astype(np.int8)])
    e.write_array(folder/'combined_predictions.npz',predictions=predictions,y=cohort['y'],sequence_id=cohort['meta'].sequence_id.to_numpy(),
        families=np.array([f for f,s in RUNS]),seeds=np.array([s if s is not None else -1 for f,s in RUNS]))
    e.write_array(folder/'rule_evidence_scores.npz',scores=scores,thresholds=thresholds,binary_predictions=scores>=thresholds,**aux)
    e.write_csv(folder/'combined_metrics_by_run.csv',[dict(condition=name,family=f,seed=s,macro_f1=e.metrics(cohort['y'],p)['macro_f1'],
        accuracy=e.metrics(cohort['y'],p)['accuracy']) for (f,s),p in zip(RUNS,predictions)])
    e.write_completion(folder,'transfer_complete.json',contract,dict(condition=name,endpoints=len(cohort['y']),runs=54),
        ['combined_predictions.npz','rule_evidence_scores.npz','combined_metrics_by_run.csv'])


def family_scores(scores):
    require(scores.shape[-1]==54,'Expected all 54 runs')
    return np.stack([scores[...,[i for i,(f,_) in enumerate(RUNS) if f==family]].mean(-1) for family in FAMILIES],axis=-1)


def block_counts(y,blocks,predictions):
    y=np.asarray(y);p=np.asarray(predictions);codes,names=pd.factorize(blocks,sort=True)
    require(p.shape==(4,54,len(y)) and len(codes)==len(y),'Wrong parameter-stress prediction/block shape')
    require(np.issubdtype(y.dtype,np.integer) and np.issubdtype(p.dtype,np.integer)
        and np.isin(y,range(6)).all() and np.isin(p,range(6)).all() and (codes>=0).all(),'Invalid parameter-stress class/block IDs')
    for i in range(len(names)):
        counts=np.bincount(y[codes==i],minlength=6)
        require((counts>0).all() and np.all(counts==counts[0]),'Incomplete paired six-class block')
    counts=np.zeros((len(names),4,54,6,6),dtype=np.int64)
    for ci in range(4):
        for ri in range(54): counts[:,ci,ri]=np.bincount((codes*36+6*y+p[ci,ri]).astype(np.int64),minlength=len(names)*36).reshape(-1,6,6)
    return counts


def bootstrap(counts,draws=DRAWS,seed=BOOTSTRAP_SEED,batch=100,progress=True):
    require(counts.ndim==5 and counts.shape[1:]==(4,54,6,6) and len(counts)>1,'Invalid block confusion tensor')
    require(draws>0 and batch>0,'Invalid bootstrap size')
    sufficient=np.stack((np.diagonal(counts,axis1=-2,axis2=-1),counts.sum(-1),counts.sum(-2)),axis=-2).astype(float)
    flat=sufficient.reshape(len(counts),-1);rng=np.random.Generator(np.random.PCG64(seed));output=np.empty((draws,4,9))
    for lo in range(0,draws,batch):
        n=min(batch,draws-lo);w=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),size=n)
        z=(w@flat).reshape(n,4,54,3,6);den=z[...,1,:]+z[...,2,:]
        f=np.divide(2*z[...,0,:],den,out=np.zeros_like(den),where=den>0).mean(-1)
        output[lo:lo+n]=family_scores(f)
        if progress and (lo==0 or (lo+n)%1000==0 or lo+n==draws):print(f'Block bootstrap {lo+n:,}/{draws:,}',flush=True)
    return output


def interval(condition,comparator,point,samples,selected_pool):
    lo,hi=np.quantile(samples,[.025,.975],method='linear');blo,bhi=np.quantile(samples,[.05/64,1-.05/64],method='linear')
    return dict(condition=condition,reference='TST',comparator=comparator,
        primary=condition=='P_moderate' and comparator==selected_pool,difference=float(point),
        ci95_low=float(lo),ci95_high=float(hi),familywise_ci_low=float(blo),familywise_ci_high=float(bhi),bonferroni_comparisons=32,
        interpretation='reference_higher' if lo>0 else 'comparator_higher' if hi<0 else 'zero_included_not_equivalence',
        familywise_interpretation='reference_higher' if blo>0 else 'comparator_higher' if bhi<0 else 'zero_included_not_equivalence')


def finish_statistics(out,protocol,previous,rule_info):
    folder=out/'statistics';folder.mkdir(exist_ok=True);identity=protocol['identity']
    contract=e.object_hash(dict(protocol=identity,stage='statistics'))
    if e.read_completion(folder,'complete.json',contract) is not None:return
    predictions=[];cohorts=[];meta0=None;blocks0=None;y0=None
    for name in CONDITIONS:
        cohort=m48.load_fresh(out/'generation'/name);meta=cohort['meta'];cohorts.append(cohort)
        blocks=pd.read_csv(out/'generation'/name/'endpoint_blocks.csv')
        m48.same_frame(blocks[['sequence_id','episode_id']],meta[['sequence_id','episode_id']],'Unaligned block identities')
        with np.load(out/'evaluation'/name/'combined_predictions.npz',allow_pickle=False) as z:
            require(np.array_equal(z['y'],cohort['y']) and np.array_equal(z['sequence_id'],meta.sequence_id)
                and np.array_equal(z['families'],[f for f,s in RUNS]) and np.array_equal(z['seeds'],[s if s is not None else -1 for f,s in RUNS]),'Combined prediction metadata differs')
            p=z['predictions'];y=z['y']
        if meta0 is None:meta0=meta;blocks0=blocks;y0=y
        else:
            m48.same_frame(meta,meta0,'Unpaired parameter-stress endpoints');m48.same_frame(blocks,blocks0,'Unpaired parameter-stress blocks')
            require(np.array_equal(y,y0),'Unpaired parameter-stress truth')
        predictions.append(p)
    counts=block_counts(y0,blocks0.block_id.to_numpy(),np.stack(predictions))
    require(len(counts)==len(np.unique(blocks0.block_id)) and len(counts)>1,'Missing paired episode clusters')
    cms=counts.sum(0);run_scores=e.macro_f1(cms);point=family_scores(run_scores)
    boot=bootstrap(counts,draws=DRAWS,seed=BOOTSTRAP_SEED)
    summary=[];comparisons=[];degradation=[];perclass=[];diagnostics=[]
    for ci,name in enumerate(CONDITIONS):
        for fi,family in enumerate(FAMILIES):
            ii=[i for i,(f,s) in enumerate(RUNS) if f==family];lo,hi=np.quantile(boot[:,ci,fi],[.025,.975])
            original=previous['clean_scores'][family] if fi<8 else rule_info['original_test_macro_f1']
            summary.append(dict(condition=name,family=family,macro_f1=float(point[ci,fi]),ci95_low=float(lo),ci95_high=float(hi),
                head_seed_sd=float(run_scores[ci,ii].std(ddof=1)) if len(ii)>1 else None,
                original_A_macro_f1=float(original),stress_minus_original_descriptive=float(point[ci,fi]-original),
                normal_false_alarm_rate=float((1-cms[ci,ii,0,0]/cms[ci,ii,0,:].sum(-1)).mean())))
            if fi:comparisons.append(interval(name,family,point[ci,0]-point[ci,fi],boot[:,ci,0]-boot[:,ci,fi],protocol['frozen_selection']['selected_pool']))
            if ci:
                lo,hi=np.quantile(boot[:,ci,fi]-boot[:,0,fi],[.025,.975])
                degradation.append(dict(condition=name,family=family,stress_minus_reference=float(point[ci,fi]-point[0,fi]),descriptive_ci95_low=float(lo),descriptive_ci95_high=float(hi)))
            cm=cms[ci,ii];den=cm.sum(-1)+cm.sum(-2);tp=np.diagonal(cm,axis1=-2,axis2=-1)
            cf=np.divide(2.*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(0)
            for k,label in enumerate(LABELS):perclass.append(dict(condition=name,family=family,label=label,mean_seed_f1=float(cf[k])))
            for stratum in ('current_changed','history_only_changed','no_observable_change'):
                for k,label in enumerate(LABELS):
                    mask=(cohorts[ci]['effects'].audit_effect_stratum.to_numpy()==stratum)&(y0==k);n=int(mask.sum())
                    diagnostics.append(dict(condition=name,family=family,label=label,effect_stratum=stratum,endpoints=n,
                        mean_seed_accuracy=float((predictions[ci][ii][:,mask]==k).mean()) if n else None,
                        mean_seed_predicted_NORMAL_fraction=float((predictions[ci][ii][:,mask]==0).mean()) if n else None))
    for filename,rows in {'model_summary.csv':summary,'TST_paired_comparisons.csv':comparisons,
        'paired_stress_changes.csv':degradation,'per_class_f1.csv':perclass,'projection_change_diagnostics.csv':diagnostics}.items():e.write_csv(folder/filename,rows)
    e.write_array(folder/'block_confusions.npz',counts=counts,blocks=np.array(sorted(blocks0.block_id.unique())),
                  families=np.array([f for f,s in RUNS]),seeds=np.array([s if s is not None else -1 for f,s in RUNS]))
    e.write_array(folder/'bootstrap_distribution.npz',family_scores=boot,point=point,conditions=np.array(CONDITIONS),families=np.array(FAMILIES))
    files=[p.name for p in folder.iterdir() if p.is_file() and p.name!='complete.json']
    e.write_completion(folder,'complete.json',contract,dict(draws=DRAWS,seed=BOOTSTRAP_SEED,clusters=len(counts)),files)


def required_cache_stages(identity):
    stages=[('cache/token_budget','complete.json',e.object_hash(dict(protocol=identity,stage='all_B_token_lengths'))),
        ('cache/original_replay','complete.json',e.object_hash(dict(protocol=identity,stage='numerical_canaries')))]
    for family,seed in e.RUNS[:50]:
        relative='cache/original_replay/heads/'+e.run_key(family,seed)
        stages.append((relative,'complete.json',e.object_hash(dict(protocol=identity,stage=relative))))
    for name in CONDITIONS:
        parent=e.object_hash(dict(protocol=identity,stage='evaluate',name=name))
        stages.append((f'cache/{name}','embeddings_complete.json',e.object_hash(dict(parent=parent,stage='embeddings'))))
        for relative in [f'cache/{name}/heads/'+e.run_key(f,s) for f,s in e.RUNS[:50]]+[f'cache/{name}/window_probabilities']:
            stages.append((relative,'complete.json',e.object_hash(dict(protocol=identity,stage=relative))))
    return stages


def verify_required_stages(out,identity):
    for relative,filename,contract in required_cache_stages(identity):
        require(e.read_completion(out/relative,filename,contract) is not None,'Missing required cache stage: '+relative+'/'+filename)
    for name in CONDITIONS:
        stages=[('generation','complete.json',dict(protocol=identity,stage='generation',condition=name)),
                ('evaluation','complete.json',dict(protocol=identity,stage='evaluate',name=name)),
                ('evaluation','transfer_complete.json',dict(protocol=identity,stage='combined_rule_and_neural',condition=name))]
        for tree,filename,contract in stages:
            require(e.read_completion(out/tree/name,filename,e.object_hash(contract)) is not None,'Missing completed '+tree+'/'+name)
    require(e.read_completion(out/'statistics','complete.json',e.object_hash(dict(protocol=identity,stage='statistics'))) is not None,'Missing completed parameter-stress statistics')


def finish_archive(out,identity):
    manifest_path=out/'run_manifest.json'
    if manifest_path.exists():
        manifest=e.read_json(manifest_path)
        require(manifest.get('status')=='PARAMETER_STRESS_COMPLETE' and manifest.get('protocol_identity')==identity,
                'Wrong completed parameter-stress manifest')
        require(manifest.get('identity')==e.object_hash({k:v for k,v in manifest.items() if k!='identity'}),'Parameter-stress manifest changed')
        for name,sha in manifest['output_sha256'].items():
            path=e.safe_artifact(out,name);require(path.is_file() and e.digest(path)==sha,'Completed parameter-stress output changed/missing: '+name)
    else:
        verify_required_stages(out,identity);m48.verify_completion_tree(out)
        files=[p.relative_to(out).as_posix() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='.run.lock' and not p.name.endswith('.tmp')]
        review=[n for n in files if not n.startswith('cache/') or n.endswith(('complete.json','audit.json','evidence.json'))]
        manifest=dict(status='PARAMETER_STRESS_COMPLETE',protocol_identity=identity,completed_at_utc=e.utc(),
            conditions=4,trained_heads=50,deterministic_controls=4,training=False,stress_used_for_selection=False,
            output_sha256={n:e.digest(out/n) for n in files},review_files=review,
            review_archive_omits='Embedding/probability caches remain on PC; all retained files bound here. Raw maker primitives/records/predictions are included.')
        manifest['identity']=e.object_hash(manifest);e.write_json(manifest_path,manifest)
    return m48.package_review(out,manifest)


def audit_train_overlap(root,out,identity):
    folder=out/'audit/train_overlap';folder.mkdir(parents=True,exist_ok=True)
    contract=e.object_hash(dict(protocol=identity,stage='original_train_overlap'))
    if e.read_completion(folder,'complete.json',contract) is not None:return
    original=np.unique(m48.row_fingerprints(e.read_records(root/e.DATA_REL/'splits/train_observable.csv')))
    report={}
    for name in CONDITIONS:
        frame=e.read_records(out/'generation'/name/'evaluation_records.csv')
        report[name]=dict(rows=len(frame),matching_rows=int(np.isin(m48.row_fingerprints(frame),original).sum()))
    report['scope']='64-bit hashes of 22-field vectors only; descriptive finite-vocabulary overlap; no row removal or independence proof'
    e.write_json(folder/'audit.json',report);e.write_completion(folder,'complete.json',contract,report,['audit.json'])


def execute(root,out,protocol,context,data,previous,rule,rule_info,paths,generate_only=False):
    protocol=json.loads(json.dumps(protocol,allow_nan=False));protocol['identity']=e.object_hash(protocol)
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():require(not any(p.name!='.run.lock' for p in out.iterdir()),'Foreign output files without parameter-stress protocol; choose an empty --output-root')
    e.save_matching_json(out/'protocol.json',protocol);identity=protocol['identity']
    if (out/'run_manifest.json').exists():return finish_archive(out,identity)
    # Deleted/completely missing stages are checked explicitly before finalizing.
    m48.verify_completion_tree(out)
    e.save_matching_json(out/'input_audit.json',context['observed'])
    e.save_matching_json(out/'validation_selection.json',previous['selection'])
    e.save_matching_json(out/'rule_source.json',rule_info)
    source=out/'source';source.mkdir(exist_ok=True)
    for path in [Path(__file__),*paths.values(),context['helper_path'],*context['code_paths'].values(),Path(rule_info['code'])]:
        dest=source/path.name
        if dest.exists():require(e.digest(dest)==e.digest(path),'Saved source changed: '+path.name)
        else:shutil.copyfile(path,dest)
    if generate_only:
        generate_all(out,protocol);audit_train_overlap(root,out,identity)
        context['helper'].recheck_inputs(context['observed'])
        require(e.digest(Path(__file__))==protocol['script_sha256'],'Step49 changed during generation')
        return None
    runtime=e.evaluation_runtime(context)
    require(runtime==protocol['expected_inference_runtime'],'Use the same inference environment/device/settings as completed Step47')
    e.save_matching_json(out/'inference_runtime.json',runtime)
    modules=e.import_models(context)
    for family,seed in e.RUNS[:50]:e.validate_checkpoint(e.read_checkpoint(e.paths_for(family,seed,context)[0]),family,seed,context)
    tokenizer,encoder=e.load_encoder(root,context)
    canary=m48.replay_canaries(root,out,identity,context,data,modules,tokenizer,encoder)
    e.save_matching_json(out/'canary_audit.json',dict(neural=canary,rule_L10=rule_info['original_replay']))
    generate_all(out,protocol);audit_token_budget(out,tokenizer,protocol,context);audit_train_overlap(root,out,identity)
    for name in CONDITIONS:evaluate_condition(root,out,name,protocol,context,tokenizer,encoder,modules,rule,rule_info)
    del encoder;gc.collect()
    context['helper'].recheck_inputs(context['observed'])
    require(e.digest(Path(__file__))==protocol['script_sha256'],'Step49 source changed during execution')
    finish_statistics(out,protocol,previous,rule_info)
    return finish_archive(out,identity)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=BASE)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--step47-code',type=Path)
    parser.add_argument('--step48-code',type=Path)
    parser.add_argument('--step47-root',type=Path)
    parser.add_argument('--shared-code-dir',type=Path)
    parser.add_argument('--maker-code-dir',type=Path,help='Folder containing current makers 1--6 and both shared modules')
    parser.add_argument('--maker-data-dir',type=Path,help='Original maker learning/raw CSV folder; default project-root/dataset')
    parser.add_argument('--step15-code',type=Path)
    parser.add_argument('--step16-code',type=Path)
    parser.add_argument('--rule-run',type=Path,help='Completed current Step16 run; required if more than one matches')
    parser.add_argument('--tst-root',type=Path)
    parser.add_argument('--pooling-root',type=Path)
    parser.add_argument('--lstm-root',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only',action='store_true',help='Read-only original source/Rule16 replay; no neural model loading or parameter generation')
    mode.add_argument('--generate-only',action='store_true',help='Generate four parameter regimes; no neural model loading, evaluation or formal result claim')
    args=parser.parse_args(argv);root=args.project_root.resolve();paths=load_dependencies(root,args)
    print('Step49: '+VERSION+' | frozen A models -> shared-nominal parameter stress; no retraining or regime selection.',flush=True)
    context=e.audit_sources(root,args.step15_code,args.tst_root,args.pooling_root,args.lstm_root)
    for path in paths.values():context['helper'].track_file(path,context['observed'])
    global makers, nominal, context_module
    makers, splitter, maker_paths = m48.load_generators(root,
        (args.maker_code_dir or args.shared_code_dir or root).resolve(), context['build'], context['observed'], context['helper'])
    nominal=sys.modules['shared_nominal'];context_module=sys.modules['shared_context_v4'];paths.update(maker_paths)
    for key, expected in GENERATOR_SOURCE_SHA256.items():
        require(e.digest(paths[key])==expected, 'Maker/shared source differs from the audited A generator: '+key)
    original_sources=m48.audit_original_makers(root,(args.maker_data_dir or root/'dataset').resolve(),context,splitter)
    require(len(set(_SEEDS))==8 and not set(MAKER_SEEDS).intersection(s['seed'] for s in original_sources),
        'Prespecified parameter-stress seeds overlap original seeds; stop without choosing a replacement based on scores')
    previous=m48.audit_step47(root,(args.step47_root or root/e.OUT_REL).resolve(),context,paths['47'])
    data=e.load_cohorts(root,context)
    folder=resolve_rule_run(root,args.rule_run,context['source_contract'])
    rule,rule_info=audit_rule(root,folder,context,data,args.step16_code)
    protocol=make_protocol(context,previous,rule_info,paths)
    protocol['original_maker_sources']=original_sources
    context['helper'].recheck_inputs(context['observed'])
    if args.preflight_only:
        print('PREFLIGHT PASS. Original Rule16 replay passed; no parameter generation or neural backend loaded. Neural replay is checked at execution.',flush=True)
        return 0
    out=(args.output_root or root/OUT_REL).resolve()
    protected=[root/'outputs/shared_nominal_v2/49_generator_b_transfer',root/'dataset',root/e.ENC_REL,root/e.EMB_REL,root/e.SEQ_REL,Path(previous['folder']),folder,
               *[Path(v) for v in context['roots'].values()]]
    m48.validate_output_location(out,protected)
    out.mkdir(parents=True,exist_ok=True)
    with e.RunLock(out/'.run.lock'):archive=execute(root,out,protocol,context,data,previous,rule,rule_info,paths,args.generate_only)
    if archive is None:print('GENERATION ONLY COMPLETE. No parameter-stress model results. Rerun without --generate-only for evaluation.',flush=True)
    else:print(f'COMPLETE. Review archive: {archive}',flush=True)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print(f'\nSTOPPED: {type(exc).__name__}: {exc}\nResolve the cause and rerun the same command; completed stages are verified and reused.',file=sys.stderr,flush=True)
        raise
