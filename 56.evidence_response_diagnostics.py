#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 56: frozen-prediction response and sparse-observation diagnostics.

Run beside the existing numbered scripts:
    python 56.evidence_response_diagnostics.py

CPU only (NumPy/Pandas), two BLAS threads. No PyTorch, training, inference,
model selection, threshold calibration, new data, or changes to Step 47.
Default inputs: completed Step47 run_missing8_uniform_v3 and completed Step16.
--step47 accepts the original run folder OR its review ZIP.
--rule-run accepts the original Step16 folder OR its archive.
--rule-review alternatively accepts the completed NOS3 evaluator review which
already reproduced every original clean Rule_L10 decision. Only BENCHMARK clean
Rule_L10 rows are used; no NOS3 observations enter this analysis.

This is a retrospective, execution-locked supplement, NOT a preregistration.
The first paired-nominal observable change is NOT attack onset. Sparse evidence
is NOT a short attack. Response means two consecutive correct-class predictions,
confirmed at the SECOND timestamp, with both predictions at/after t0.
Latency eligibility requires predictions available at t0 and through t0+10s.
Undetected episodes contribute 10s to RESTRICTED waiting time, not a fabricated
actual latency. Zero-change, warm-up and short-follow-up exclusions are reported.
All clean endpoints, including excluded episodes, remain in sparse diagnostics.
Normal alarms require two consecutive predictions of the SAME non-normal class.

Primary family: TST minus the Step47 CLEAN-VALIDATION-selected pooling family,
for restricted waiting, detection-by-10s, and normal episode alarm probability.
20,000 paired whole-episode draws, stratified by class and latency eligibility;
95% descriptive and three-outcome Bonferroni intervals. Six-class outputs remain.
Head seeds are averaged after per-head computation, never ensembled or resampled.
Sparse-count contrasts are descriptive; no confirmatory subgroup claims.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone

# Set before importing NumPy; this is a small CPU-only analysis.
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '2'
import numpy as np
import pandas as pd

VERSION = '1.0-frozen-evidence-response-diagnostics'
DATASET = '478f782ceb970d7a2de73319ccc4e929be06cfea684a054aa72d932ed65ae83b'
PREFIX = 'outputs/shared_nominal_v2'
STEP47 = PREFIX + '/47_missing8_aggregation/run_missing8_uniform_v3'
RULE_ROOT = PREFIX + '/16_observable_rule_baseline'
OUTPUT = PREFIX + '/56_evidence_response_diagnostics'
LABELS = ('NORMAL','RF_LINK_JAMMING','GNSS_SPOOFING',
          'UNAUTHORIZED_TELECOMMAND_INJECTION','TELECOMMAND_REPLAY',
          'GROUND_SEGMENT_TTC_CONTROL_PATH_COMPROMISE')
FIELDS = ('observation_stage','path_type','direction','source_role','destination_role',
 'protocol_context','operational_phase','message_type','service_context','command_intent',
 'payload_or_parameter_context','sequence_freshness_status','control_security_state',
 'signal_quality_state','tracking_stability','navigation_consistency','timing_consistency',
 'decoder_status','command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
SEEDS = tuple(range(20260610,20260620))
FAMILIES = ('TST','Mean_MLP','Max_MLP','MeanMax_MLP','LSTM',
            'ProbabilityMean','MajorityVote','EWMA','Rule_L10')
RUNS = tuple((f,s) for f in FAMILIES[:5] for s in SEEDS) + tuple((f,-1) for f in FAMILIES[5:])
BINS = ('0','1','2','3-5','6-10')
HORIZON = 10
DRAWS = 20000
BOOT_SEED = 2026100756


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return sha_bytes(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.2*(attempt+1))


def json_records(frame):
    # Pandas promotes None in optional numeric columns (e.g. SD of one
    # deterministic rule) to NaN. Preserve the explicit JSON null instead.
    return frame.astype(object).where(pd.notna(frame),None).to_dict(orient='records')


def safe_name(name):
    p = PurePosixPath(name)
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name
            and not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,
            'Unsafe/noncanonical artifact name: '+str(name))
    return p


class Bundle:
    """Read-only folder/ZIP. Never imports or executes code from an input bundle."""
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        require(self.path.exists(), 'Input not found: '+str(self.path))
        self.zip = None
        self.prefix = ''
        self.observed = {}
        self.expected = {}
        if self.path.is_file():
            require(zipfile.is_zipfile(self.path), 'Expected a result folder or ZIP: '+str(self.path))
            self.zip = zipfile.ZipFile(self.path)
            names = [i.filename for i in self.zip.infolist() if not i.is_dir()]
            require(len(names)==len(set(names)), 'Duplicate ZIP member names')
            for name in names:
                safe_name(name)
            if not any(n in names for n in ('run_manifest.json','result_manifest.json')):
                candidates = [str(PurePosixPath(n).parent)+'/' for n in names
                              if n.endswith('/run_manifest.json') or n.endswith('/result_manifest.json')]
                require(len(set(candidates))==1, 'Archive must contain exactly one result root')
                self.prefix = candidates[0]
            self.names = set(n[len(self.prefix):] for n in names if n.startswith(self.prefix))
        else:
            self.names = None

    def read(self, name):
        safe_name(name)
        if self.zip is not None:
            require(name in self.names,'Missing archive member: '+name)
            require(self.zip.getinfo(self.prefix+name).file_size <= 512*1024**2,
                    'Unexpected oversized review member: '+name)
            data = self.zip.read(self.prefix+name)
        else:
            path = (self.path/name).resolve()
            require(path.is_relative_to(self.path) and path.is_file(), 'Missing/unsafe input: '+str(path))
            data = path.read_bytes()
        actual = sha_bytes(data)
        require(name not in self.expected or actual==self.expected[name],
                'Artifact changed after verification: '+name)
        self.observed[name] = actual
        return data

    def json(self, name):
        return json.loads(self.read(name).decode('utf-8-sig'))

    def csv(self, name):
        return pd.read_csv(io.BytesIO(self.read(name)),encoding='utf-8-sig',keep_default_na=False)

    def npz(self, name):
        with np.load(io.BytesIO(self.read(name)),allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def verify(self, hashes):
        require(isinstance(hashes,dict) and bool(hashes), 'Missing artifact hashes')
        for name, expected in hashes.items():
            require(isinstance(expected,str) and len(expected)==64, 'Invalid hash: '+name)
            self.expected[name] = expected
            require(sha_bytes(self.read(name))==expected, 'Artifact hash mismatch: '+name)

    def recheck(self):
        previous = dict(self.observed)
        for name,expected in previous.items():
            require(sha_bytes(self.read(name))==expected,'Input changed during analysis: '+name)

    def close(self):
        if self.zip is not None:
            self.zip.close()


def integers(values, name):
    a = pd.to_numeric(np.asarray(values),errors='raise')
    require(np.isfinite(a).all() and np.equal(a,np.floor(a)).all(), 'Non-integer '+name)
    return np.asarray(a,dtype=np.int64)


def align_rule(frame, meta, y):
    require(len(frame)==len(meta) and 'sequence_id' in frame, 'Incomplete Rule_L10 endpoints')
    f = frame.copy()
    f['sequence_id'] = integers(f.sequence_id,'Rule sequence_id')
    require(f.sequence_id.is_unique,'Duplicate Rule sequence_id')
    f = f.sort_values('sequence_id').reset_index(drop=True)
    for k in ('sequence_id','episode_id','sequence_end_row'):
        require(k in f and np.array_equal(f[k].to_numpy(),meta[k].to_numpy()),'Rule alignment differs: '+k)
    yc = next((k for k in ('true_label_id','y_true','label_id') if k in f),None)
    pc = next((k for k in ('pred_label_id','y_pred','predicted_label_id') if k in f),None)
    require(yc is not None and pc is not None, 'Rule CSV lacks supported label columns')
    truth = integers(f[yc],'Rule true labels')
    pred = integers(f[pc],'Rule predictions')
    require(np.array_equal(truth,y) and np.isin(pred,np.arange(6)).all(),'Rule labels/predictions differ')
    for col,values in [('true_label',truth),('pred_label',pred),('prediction',pred)]:
        if col in f:
            require(np.array_equal(f[col].to_numpy(),np.asarray(LABELS)[values]),'Rule class order differs: '+col)
    if 'model' in f:
        require(f.model.eq('Rule_L10').all(),'Rule CSV contains another model')
    if 'condition' in f:
        require(f.condition.eq('clean').all(),'Rule CSV contains non-clean predictions')
    return pred.astype(np.int8)


def load_step47(path):
    b = Bundle(path)
    manifest = b.json('run_manifest.json')
    require(manifest.get('status')=='MISSING8_AGGREGATION_COMPLETE','Step47 is not complete')
    b.verify(manifest.get('output_sha256'))
    p = b.json('protocol.json')
    content = {k:v for k,v in p.items() if k!='identity'}
    require(object_hash(content)==p['identity']==manifest['protocol_identity'],'Step47 protocol identity mismatch')
    require(p['source_contract']['dataset_id']==DATASET and tuple(p['labels'])==LABELS
            and tuple(p['fields'])==FIELDS and tuple(p['head_seeds'])==SEEDS,'Wrong dataset/schema/head seeds')
    require(p['training'] is False and p['test_used_for_selection'] is False,'Unexpected Step47 training/selection')
    z = b.npz('predictions.npz')
    require(str(z['dataset_id'])==DATASET and str(z['step47_protocol_identity'])==p['identity'], 'Prediction lineage differs')
    require(tuple(z['conditions'])==('clean','missing8_20261001','missing8_20261002','missing8_20261003'), 'Wrong Step47 condition order')
    require(list(zip(z['families'].tolist(),z['seeds'].tolist()))==list(RUNS[:-1]),'Wrong head/run order')
    y = z['y']; predictions = z['predictions']
    require(predictions.dtype.kind in 'iu' and y.dtype.kind in 'iu' and predictions.shape==(4,53,38078)
            and y.shape==(38078,) and np.isin(predictions,range(6)).all() and np.isin(y,range(6)).all(), 'Invalid Step47 predictions')
    w = b.csv('window_mask_alignment.csv')
    m = b.csv('evaluation_endpoints_L10.csv')
    require(len(w)==44846 and len(m)==38078 and w.episode_id.nunique()==752,'Unexpected test population')
    for col in ('embedding_row_index','episode_step','window_id'):
        w[col] = integers(w[col],col)
    for col in ('sequence_id','sequence_start_row','sequence_end_row','sequence_end_step','label_id'):
        m[col] = integers(m[col],col)
    require(np.array_equal(w.embedding_row_index,np.arange(len(w))) and np.array_equal(m.sequence_id,np.arange(len(m))), 'Rows/endpoints are not canonical')
    require(not w.duplicated(['episode_id','episode_step']).any() and w.groupby('episode_id').label.nunique().eq(1).all(),'Duplicate/mixed-label episodes')
    require(np.array_equal(m.label_id,y) and np.array_equal(m.label,np.asarray(LABELS)[y]),'Endpoint labels disagree')
    times = pd.to_datetime(w.timestamp,utc=True,errors='raise').to_numpy(dtype='datetime64[ns]')
    expected = []
    for _,ix in w.groupby('episode_id',sort=False).indices.items():
        ix = np.asarray(ix)
        require(np.array_equal(w.episode_step.to_numpy()[ix],np.arange(len(ix)))
                and (np.diff(ix)==1).all() and (np.diff(times[ix])==np.timedelta64(1,'s')).all(),
                'Windows must be contiguous, one-second, zero-based episodes')
        if len(ix)>=10:
            expected.extend(ix[9:])
    require(np.array_equal(m.sequence_end_row,np.array(expected))
            and np.array_equal(m.sequence_start_row,m.sequence_end_row-9),'Missing/reordered/noncausal L10 endpoints')
    end = m.sequence_end_row.to_numpy()
    require(np.array_equal(m.episode_id,w.episode_id.to_numpy()[end])
            and np.array_equal(m.sequence_end_step,w.episode_step.to_numpy()[end])
            and np.array_equal(m.label,w.label.to_numpy()[end]),'Endpoint/window alignment differs')
    changes = b.npz('paired_nominal_field_changes.npz')
    require(tuple(changes['fields'])==FIELDS and bool(changes['diagnostic_only'])
            and changes['changes'].shape==(len(w),22) and changes['changes'].dtype==bool,'Wrong diagnostic field-change matrix')
    flags = changes['changes'].any(1)
    require(not flags[w.label.eq('NORMAL').to_numpy()].any(),'Normal has paired-nominal changes')
    ix = end[:,None]-np.arange(9,-1,-1)
    count = flags[ix].sum(1)
    ef = b.csv('clean_context_observable_effects_L10.csv')
    require(np.array_equal(ef.sequence_id,m.sequence_id)
            and np.array_equal(ef.audit_context_effect_window_count,count)
            and np.array_equal(ef.audit_current_observable_effect,flags[end].astype(int)), 'Observable-change audits disagree')
    selection = b.json('validation_selection.json')
    scores = selection['candidate_pool_mean_validation_f1']
    pools = ('Mean_MLP','Max_MLP','MeanMax_MLP')
    require(set(scores)==set(pools) and all(np.isfinite(v) for v in scores.values()),'Invalid validation selection')
    chosen = pools[int(np.argmax([scores[f] for f in pools]))]
    require(selection['selected_pool']==chosen and selection['all_ten_head_seeds_retained'] is True
            and selection['selection_data']=='clean validation only; no missing-mask or test metrics', 'Pooling was not selected on clean validation')
    vm = b.csv('validation_metrics.csv')
    for family in pools:
        s = vm[vm.family.eq(family)]
        require(len(s)==10 and abs(s.macro_f1.mean()-scores[family])<1e-12,'Saved validation selection mismatch')
    return dict(bundle=b,protocol=p,manifest=manifest,windows=w,meta=m,y=y,
                predictions=predictions[0].astype(np.int8),flags=flags,evidence_count=count,selected_pool=chosen)


def rule_contract(manifest, source):
    c = manifest.get('source_contract',manifest.get('dataset_contract',{}))
    require(manifest.get('status')=='COMPLETE' and manifest.get('calibrate_only') is False,'Rule run is not a completed test evaluation')
    for key in ('dataset_id','step8_run_id','step9_run_id','step10_run_id'):
        require(c.get(key)==source[key],'Rule source contract differs: '+key)
    require(manifest.get('test_opened_after_freeze') is True,'Rule calibration/test freeze record missing')


def verify_rule_freeze(manifest, config, protocol):
    # Step16 protocol_sha256 is a canonical JSON OBJECT digest. Its outputs
    # inventory separately binds the byte hash of the pretty-printed JSON file.
    require(object_hash(protocol)==manifest['protocol_sha256']==config['protocol_sha256'],
            'Rule canonical protocol identity differs')
    require(config.get('label_to_id')==dict(zip(LABELS,range(6))),'Rule frozen class order differs')


def original_rule(path, data):
    b = Bundle(path)
    m = b.json('run_manifest.json')
    rule_contract(m,data['protocol']['source_contract'])
    hashes = {r['file']:r['sha256'] for r in m['outputs']}
    names = ('test_predictions_Rule_L10.csv','frozen_rule_config.json','protocol.json')
    require(all(n in hashes for n in names),'Rule run lacks required saved artifacts')
    b.verify({n:hashes[n] for n in names})
    require(hashes['frozen_rule_config.json']==m['frozen_config_sha256'],'Rule frozen config hash disagrees')
    verify_rule_freeze(m,b.json('frozen_rule_config.json'),b.json('protocol.json'))
    inputs = {r['path'].replace('\\','/').split('/')[-1]:r['sha256'] for r in m['inputs']}
    source = data['protocol']['source_contract']
    require(inputs.get('test_observable.csv')==source['input_sha256']['splits/test_observable.csv']
            and inputs.get('test_tst_sequences_L10.npz')==source['step10_artifacts']['test_tst_sequences_L10.npz'],
            'Rule evaluation used different test observations/sequences')
    pred = align_rule(b.csv(names[0]),data['meta'],data['y'])
    return pred,b,dict(route='original_Step16',input=str(b.path),config_sha256=m['frozen_config_sha256'],
                       prediction_sha256=hashes[names[0]],recalibrated=False)


def reviewed_rule(path, data):
    b = Bundle(path)
    hashes = b.json('result_manifest.json')
    names = ('summary.json','rule_provenance.json','rule_original_run_manifest.json',
             'frozen_rule_config.json','rule_original_protocol.json','benchmark_clean_Rule_L10_predictions.csv')
    require(all(n in hashes for n in names),'Rule review lacks complete clean benchmark evidence')
    b.verify({n:hashes[n] for n in names})
    summary = b.json('summary.json'); proof = b.json('rule_provenance.json')
    require(summary.get('status')=='EXTENDED_COMPLETE' and not summary.get('rule_errors')
            and summary.get('benchmark',{}).get('status')=='COMPLETE','Rule review is not fully complete')
    original = b.json('rule_original_run_manifest.json')
    rule_contract(original,data['protocol']['source_contract'])
    require(proof.get('status')=='PASS' and proof.get('clean_control_endpoints')==len(data['y'])
            and proof.get('recalibrated') is False,'No full clean Rule reproduction evidence')
    require(hashes['frozen_rule_config.json']==original['frozen_config_sha256']==proof['frozen_config_sha256'],
            'Reviewed Rule frozen configuration differs')
    verify_rule_freeze(original,b.json('frozen_rule_config.json'),b.json('rule_original_protocol.json'))
    pred = align_rule(b.csv(names[-1]),data['meta'],data['y'])
    return pred,b,dict(route='verified_full_clean_Rule_reproduction',input=str(b.path),
                      config_sha256=proof['frozen_config_sha256'],prediction_sha256=hashes[names[-1]],
                      original_decision_agreement='Full population PASS in saved rule_provenance.json',recalibrated=False)


def choose_rule(root, explicit, review, data):
    if review is not None:
        return reviewed_rule(review,data)
    if explicit is not None:
        return original_rule(explicit,data)
    base = root/RULE_ROOT
    candidates = []
    errors = []
    for p in sorted(base.glob('run*/run_manifest.json')):
        try:
            m = json.loads(p.read_text(encoding='utf-8-sig'))
            rule_contract(m,data['protocol']['source_contract'])
            candidates.append(p.parent)
        except (ValueError,KeyError) as ex:
            errors.append(str(p)+': '+str(ex))
    require(len(candidates)==1,'Expected exactly one matching completed Step16 run; found '+str(len(candidates))+
            '. Use --rule-run with its result folder, or --rule-review with the completed evaluator ZIP. '+
            ('Rejected: '+' | '.join(errors[:2]) if errors else ''))
    return original_rule(candidates[0],data)


def eligibility(label, t0, first_prediction, last_step):
    if label==0:
        return 'normal_control'
    if t0 is None:
        return 'no_observed_change'
    if t0<first_prediction:
        return 'change_before_first_prediction'
    if last_step-t0<HORIZON:
        return 'less_than_10s_followup'
    return 'eligible'


def confirmed_alarms(pred):
    p = np.asarray(pred)
    require(p.ndim==2 and p.shape[1]>=2 and np.isin(p,range(6)).all(),'Invalid alarm prediction sequence')
    return (p[:,1:]==p[:,:-1]) & (p[:,1:]!=0)


def response(pred, steps, t0, label):
    p = np.asarray(pred); steps = np.asarray(steps)
    require(p.ndim==2 and p.shape[1]==len(steps) and len(steps)>=2
            and np.isin(p,range(6)).all() and np.all(np.diff(steps)==1),'Nonconsecutive/invalid response sequence')
    require(1<=label<=5 and steps[0]<=t0 and steps[-1]>=t0+HORIZON,'Insufficient common response follow-up')
    correct = p==label
    pairs = correct[:,1:] & correct[:,:-1]
    valid = steps[:-1]>=t0
    within = valid & (steps[1:]<=t0+HORIZON)
    hits = pairs & within[None,:]
    detected = hits.any(1)
    delays = np.where(detected,steps[1:][hits.argmax(1)]-t0,-1)
    any_hits = pairs & valid[None,:]
    any_delay = np.where(any_hits.any(1),steps[1:][any_hits.argmax(1)]-t0,-1)
    pre = steps<t0
    return dict(delay_s=delays,detected_by_horizon=detected,
        capped_wait_s=np.where(detected,delays,HORIZON),
        first_confirmation_any_delay_s=any_delay,
        pre_t0_prediction_count=int(pre.sum()),
        pre_t0_correct_any=correct[:,pre].any(1),
        pre_t0_correct_pair=(pairs & (steps[1:]<t0)[None,:]).any(1))


def evidence_bins(count):
    a = np.asarray(count)
    require(np.isin(a,np.arange(11)).all(),'Invalid observable-change window count')
    return np.select([a==0,a==1,a==2,a<=5],[0,1,2,3],default=4).astype(np.int8)


def bootstrap_weights(labels, eligible, rng, n):
    labels = np.asarray(labels); eligible = np.asarray(eligible,bool)
    require(labels.shape==eligible.shape and labels.ndim==1,'Invalid bootstrap strata')
    strata = labels*2+eligible.astype(int)
    weights = np.zeros((n,len(labels)),dtype=np.float64)
    for value in np.unique(strata):
        ix = np.flatnonzero(strata==value)
        weights[:,ix] = rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=n)
    return weights


def build_inventory(data):
    w = data['windows']; m = data['meta']; flags = data['flags']
    endpoint_groups = m.groupby('episode_id',sort=False).indices
    inventory = []
    for ep, positions in w.groupby('episode_id',sort=False).indices.items():
        g = w.iloc[positions]; ei = np.asarray(endpoint_groups[ep])
        labels = g.label.map(dict(zip(LABELS,range(6)))).to_numpy()
        require(np.isfinite(labels).all() and len(np.unique(labels))==1,'Unknown/mixed episode label')
        label = int(labels[0]); changed = g.episode_step.to_numpy()[flags[positions]]
        t0 = int(changed[0]) if len(changed) else None
        first = int(m.iloc[ei].sequence_end_step.min()); last = int(g.episode_step.max())
        reason = eligibility(label,t0,first,last)
        inventory.append(dict(episode_id=ep,label_id=label,label=LABELS[label],windows=len(g),
            common_endpoints=len(ei),first_prediction_step=first,last_step=last,
            first_observable_change_step=t0,observable_change_windows=int(len(changed)),
            followup_after_first_change_s=last-t0 if t0 is not None else None,
            latency_eligible=reason=='eligible',latency_status=reason))
    return pd.DataFrame(inventory)


def fixed_protocol(data, rule_info, script_sha):
    inventory = build_inventory(data)
    return dict(code_version=VERSION,script_sha256=script_sha,dataset_id=DATASET,
        step47_protocol_identity=data['protocol']['identity'],step47_input_sha256=dict(data['bundle'].observed),
        rule=rule_info,labels=list(LABELS),runs=[list(r) for r in RUNS],selected_pool=data['selected_pool'],
        condition='clean only; no substitution of missingness or NOS3 predictions',
        horizon_seconds=HORIZON,consecutive_correct_predictions=2,
        confirmation_time='Second prediction, both at/after first paired-nominal observable change; no backdating',
        latency_population='Common predictions present at t0 and through t0+10 inclusive; input-only eligibility',
        latency_population_counts=inventory.latency_status.value_counts().astype(int).to_dict(),
        waiting_time='min(D,10); D is first two-correct confirmation delay. No event by 10 contributes 10, not an actual detection time.',
        pre_evidence_predictions='Any correct and confirmed correct before t0 reported; not called false alarms on latent attack episodes',
        normal_alarm='Two consecutive SAME non-NORMAL class predictions. Episode event and endpoint rates both reported.',
        sparse_bins=list(BINS),sparse_metric='True-class recall on ALL clean attack endpoints; classwise and pooled. No-change is not no-information.',
        sparse_effects='Original full observable projection change flags, valid for CLEAN ONLY; never model features.',
        primary_comparison='TST minus clean-validation-selected pooling; three outcome family',
        primary_outcomes=['restricted_wait_s','detected_by_10s','normal_episode_confirmed_alarm'],
        multiplicity='95% descriptive plus Bonferroni 98.333333% marginal intervals for three primary outcomes',
        sparse_contrasts='Descriptive differences only; no subgroup significance tests',
        bootstrap=dict(draws=DRAWS,seed=BOOT_SEED,unit='whole episode',
            strata='class x input-defined latency eligibility; preserves eligible class counts',
            paired='same draw for all fixed heads/models/outcomes',head_resampling=False,
            estimand='per-head metric then mean of head seeds, never probability ensemble',
            scope='conditional on fitted heads, encoder, dataset, evidence definition, eligibility, and fixed alert rule'),
        chronology='Retrospective supplement designed after Steps45/47 and NOS3 review; execution rules frozen before new outcomes.',
        training=False,new_inference=False,threshold_tuning=False,new_live_commands=False,
        forbidden_interpretations=['attack-onset delay','short-attack detection','causal attention explanation',
            'temporal-order superiority','statistical equivalence from an interval including zero','deployment validation'],
        expected_population_note='Reviewed input suggests 379 latency episodes. Eligibility is recomputed, never selected using predictions.',
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,blas_threads=2))


def analyze(data, out):
    inv = build_inventory(data)
    meta = data['meta']; y = data['y']; pred = data['predictions']
    require(pred.shape==(len(RUNS),len(y)),'Incomplete model grid')
    groups = meta.groupby('episode_id',sort=False).indices
    n_ep = len(inv); n_runs = len(RUNS)
    cap = np.full((n_ep,n_runs),np.nan)
    det = np.full_like(cap,np.nan); alarms = np.full_like(cap,np.nan)
    raw_fp = np.full_like(cap,np.nan); pair_fp = np.full_like(cap,np.nan)
    support = np.zeros((n_ep,len(BINS)),np.int64)
    correct = np.zeros((n_ep,n_runs,len(BINS)),np.int64)
    bins = evidence_bins(data['evidence_count'])
    latency_rows=[]; normal_rows=[]
    for ep_index, row in inv.iterrows():
        ix = np.asarray(groups[row.episode_id]); steps=meta.iloc[ix].sequence_end_step.to_numpy()
        p = pred[:,ix]; label=int(row.label_id)
        if label:
            for bi in range(len(BINS)):
                take=bins[ix]==bi; support[ep_index,bi]=int(take.sum())
                correct[ep_index,:,bi]=(p[:,take]==label).sum(1)
        else:
            alerts=confirmed_alarms(p)
            alarms[ep_index]=alerts.any(1)
            raw_fp[ep_index]=(p!=0).mean(1)
            pair_fp[ep_index]=alerts.mean(1)
            for ri,(family,seed) in enumerate(RUNS):
                normal_rows.append(dict(episode_id=row.episode_id,family=family,seed=seed,
                    endpoints=len(ix),adjacent_pairs=alerts.shape[1],
                    nonnormal_endpoints=int((p[ri]!=0).sum()),confirmed_alarm_pairs=int(alerts[ri].sum()),
                    confirmed_episode_alarm=bool(alerts[ri].any())))
        if row.latency_eligible:
            r=response(p,steps,int(row.first_observable_change_step),label)
            cap[ep_index]=r['capped_wait_s']; det[ep_index]=r['detected_by_horizon']
            for ri,(family,seed) in enumerate(RUNS):
                latency_rows.append(dict(episode_id=row.episode_id,label_id=label,label=LABELS[label],family=family,seed=seed,
                    first_observable_change_step=int(row.first_observable_change_step),
                    detected_by_10s=bool(r['detected_by_horizon'][ri]),
                    confirmation_delay_s=int(r['delay_s'][ri]) if r['delay_s'][ri]>=0 else None,
                    restricted_wait_s=int(r['capped_wait_s'][ri]),
                    first_confirmation_any_delay_s=int(r['first_confirmation_any_delay_s'][ri]) if r['first_confirmation_any_delay_s'][ri]>=0 else None,
                    no_confirmation_through_episode_end=bool(r['first_confirmation_any_delay_s'][ri]<0),
                    pre_t0_prediction_count=r['pre_t0_prediction_count'],
                    pre_t0_correct_any=bool(r['pre_t0_correct_any'][ri]),
                    pre_t0_correct_pair=bool(r['pre_t0_correct_pair'][ri])))
    latency=pd.DataFrame(latency_rows); normal=pd.DataFrame(normal_rows)
    require(len(latency)==int(inv.latency_eligible.sum())*n_runs and len(normal)==int(inv.label_id.eq(0).sum())*n_runs,'Episode/run grid incomplete')
    inv.to_csv(out/'episode_inventory.csv',index=False,encoding='utf-8-sig')
    latency.to_csv(out/'latency_by_episode_seed.csv',index=False,encoding='utf-8-sig')
    normal.to_csv(out/'normal_alarms_by_episode_seed.csv',index=False,encoding='utf-8-sig')
    exclusions=inv.groupby(['label','latency_status'],sort=False).size().rename('episodes').reset_index()
    exclusions.to_csv(out/'eligibility_counts.csv',index=False,encoding='utf-8-sig')
    summary=[]; byseed=[]; sparse=[]; primary_vectors=[]
    for family in FAMILIES:
        ri=[i for i,(f,s) in enumerate(RUNS) if f==family]
        # Linear response/recall metrics: averaging fixed heads within each
        # episode is exactly equal to computing the metric per head then mean.
        primary_vectors.append(np.stack([np.nan_to_num(cap[:,ri],nan=0).mean(1),
            np.nan_to_num(det[:,ri],nan=0).mean(1),np.nan_to_num(alarms[:,ri],nan=0).mean(1)],axis=1))
        for label in [None,1,2,3,4,5]:
            block=latency[latency.family.eq(family)]
            if label is not None: block=block[block.label_id.eq(label)]
            rows=[]
            for seed,g in block.groupby('seed',sort=False):
                detected=g[g.detected_by_10s]
                before=g[g.pre_t0_prediction_count.gt(0)]
                r=dict(family=family,seed=int(seed),label='ALL_ATTACKS' if label is None else LABELS[label],
                    episodes=len(g),detected_by_10s=float(g.detected_by_10s.mean()),
                    restricted_wait_s=float(g.restricted_wait_s.mean()),
                    detected_only_mean_s=float(detected.confirmation_delay_s.mean()) if len(detected) else None,
                    detected_only_median_s=float(detected.confirmation_delay_s.median()) if len(detected) else None,
                    pre_t0_available_episodes=len(before),pre_t0_correct_any=float(before.pre_t0_correct_any.mean()) if len(before) else None,
                    pre_t0_correct_pair=float(before.pre_t0_correct_pair.mean()) if len(before) else None,
                    no_confirmation_through_episode_end=float(g.no_confirmation_through_episode_end.mean()))
                byseed.append(r); rows.append(r)
            if rows:
                summary.append(dict(family=family,label=rows[0]['label'],episodes=rows[0]['episodes'],head_seeds=len(rows),
                    restricted_wait_s=float(np.mean([r['restricted_wait_s'] for r in rows])),
                    restricted_wait_head_sd=float(np.std([r['restricted_wait_s'] for r in rows],ddof=1)) if len(rows)>1 else None,
                    detected_by_10s=float(np.mean([r['detected_by_10s'] for r in rows])),
                    not_detected_by_10s=float(1-np.mean([r['detected_by_10s'] for r in rows])),
                    role='restricted wait and detection probability, NOT attack-onset latency'))
        for bi,name in enumerate(BINS):
            for label in [None,1,2,3,4,5]:
                take=inv.label_id.ne(0).to_numpy() if label is None else inv.label_id.eq(label).to_numpy()
                den=int(support[take,bi].sum()); counts=correct[take][:,ri,bi].sum(0)
                vals=counts/den if den else None
                sparse.append(dict(family=family,evidence_windows=name,label='ALL_ATTACKS_POOLED' if label is None else LABELS[label],
                    endpoints=den,episodes=int((support[take,bi]>0).sum()),head_seeds=len(ri),
                    recall=float(vals.mean()) if den else None,
                    head_seed_sd=float(vals.std(ddof=1)) if den and len(ri)>1 else None,
                    role='descriptive pooled/per-class recall; sparse observable change is not short attack'))
    normal_summary=[]
    for family,g in normal.groupby('family',sort=False):
        seed_values=[]
        for seed,s in g.groupby('seed',sort=False):
            seed_values.append(dict(episode_alarm=float(s.confirmed_episode_alarm.mean()),
                endpoint_fpr=float(s.nonnormal_endpoints.sum()/s.endpoints.sum()),
                pair_rate=float(s.confirmed_alarm_pairs.sum()/s.adjacent_pairs.sum())))
        normal_summary.append(dict(family=family,normal_episodes=g.episode_id.nunique(),head_seeds=len(seed_values),
            normal_episode_confirmed_alarm=float(np.mean([r['episode_alarm'] for r in seed_values])),
            normal_endpoint_false_alarm=float(np.mean([r['endpoint_fpr'] for r in seed_values])),
            normal_confirmed_pair_rate=float(np.mean([r['pair_rate'] for r in seed_values])),
            exposure='Complete observed normal episodes; unequal durations, shared by all models; overlapping pairs not independent'))
    for name,rows in [('latency_summary.csv',summary),('latency_by_seed.csv',byseed),
                      ('normal_alarm_summary.csv',normal_summary),('sparse_evidence_recall.csv',sparse)]:
        pd.DataFrame(rows).to_csv(out/name,index=False,encoding='utf-8-sig')
    vectors=np.stack(primary_vectors,axis=1)
    np.savez_compressed(out/'episode_sufficient_statistics.npz',
        episode_id=inv.episode_id.to_numpy(dtype=str),labels=inv.label_id.to_numpy(),eligible=inv.latency_eligible.to_numpy(bool),
        family_primary_values=vectors,families=np.array(FAMILIES),
        sparse_support=support,sparse_correct=correct,runs=np.array([f+':'+str(s) for f,s in RUNS]))
    # Preserve all clean decisions required to independently recompute this analysis.
    np.savez_compressed(out/'clean_predictions.npz',predictions=pred,y=y,families=np.array([f for f,s in RUNS]),seeds=np.array([s for f,s in RUNS]))
    meta.to_csv(out/'evaluation_endpoints_L10.csv',index=False,encoding='utf-8-sig')
    windows=data['windows'].copy();windows['paired_nominal_changed']=data['flags'].astype(int)
    windows.to_csv(out/'window_observable_changes.csv',index=False,encoding='utf-8-sig')
    details=meta[['sequence_id','episode_id','sequence_end_step','sequence_end_row','label_id']].copy()
    details['observable_change_windows']=data['evidence_count']
    details['evidence_bin']=np.asarray(BINS)[bins]
    details.to_csv(out/'endpoint_evidence_counts.csv',index=False,encoding='utf-8-sig')
    return inv,vectors,pd.DataFrame(summary),pd.DataFrame(normal_summary),pd.DataFrame(sparse)


def paired_statistics(inv, vectors, selected_pool, out):
    labels=inv.label_id.to_numpy(); eligible=inv.latency_eligible.to_numpy(bool)
    normal=labels==0
    require(eligible.any() and normal.any(),'Empty response or normal control population')
    ti=FAMILIES.index('TST'); pi=FAMILIES.index(selected_pool)
    numerators=vectors[:,ti]-vectors[:,pi]
    denominator=np.stack([eligible,eligible,normal],axis=1).astype(float)
    point=numerators.sum(0)/denominator.sum(0)
    rng=np.random.Generator(np.random.PCG64(BOOT_SEED))
    samples=np.empty((DRAWS,3))
    for start in range(0,DRAWS,100):
        n=min(100,DRAWS-start)
        w=bootstrap_weights(labels,eligible,rng,n)
        samples[start:start+n]=(w@numerators)/(w@denominator)
        if start==0 or (start+n)%2000==0:
            print(f'  Paired episode bootstrap {start+n:,}/{DRAWS:,}',flush=True)
    names=('restricted_wait_s','detected_by_10s','normal_episode_confirmed_alarm')
    rows=[]
    for i,name in enumerate(names):
        lo,hi=np.quantile(samples[:,i],[.025,.975]); blo,bhi=np.quantile(samples[:,i],[.05/6,1-.05/6])
        smaller=name!='detected_by_10s'
        direction='zero_included_not_equivalence'
        if bhi<0: direction='TST_favorable' if smaller else 'pooling_favorable'
        elif blo>0: direction='pooling_favorable' if smaller else 'TST_favorable'
        rows.append(dict(reference='TST',comparator=selected_pool,metric=name,
            delta_TST_minus_pool=float(point[i]),ci95_low=float(lo),ci95_high=float(hi),
            bonferroni_low=float(blo),bonferroni_high=float(bhi),family_size=3,
            lower_is_better=smaller,adjusted_interval_direction=direction,
            interpretation='This outcome only; no automatic overall advantage, noninferiority, attention-causality or ordering claim'))
    pd.DataFrame(rows).to_csv(out/'primary_paired_comparisons.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'primary_bootstrap.npz',deltas=samples,metrics=np.array(names),point=point)
    return rows


def package(out, status):
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ('run_manifest.json','.run.lock') and not p.name.endswith('.tmp'))
    hashes={p.relative_to(out).as_posix():sha_file(p) for p in files}
    manifest=dict(status=status,code_version=VERSION,completed_at_utc=now(),output_sha256=hashes)
    write_json(out/'run_manifest.json',manifest)
    destination=out.with_suffix('.zip'); temporary=out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in [*hashes,'run_manifest.json']:
            z.write(out/name,name)
    with zipfile.ZipFile(temporary) as z:
        require(z.testzip() is None,'Result ZIP integrity check failed')
        for name,digest in hashes.items():
            require(sha_bytes(z.read(name))==digest,'Result ZIP hash check failed: '+name)
    temporary.replace(destination)
    return destination


def verify_completed(out):
    b=Bundle(out)
    m=b.json('run_manifest.json')
    require(m.get('status')=='EVIDENCE_RESPONSE_COMPLETE','Previous run is not complete')
    b.verify(m['output_sha256']); b.close()
    archive=out.with_suffix('.zip')
    if archive.exists():
        z=Bundle(archive)
        try:
            require(z.json('run_manifest.json')==m,'Existing ZIP manifest differs')
            z.verify(m['output_sha256'])
        finally:z.close()
    else:
        archive=package(out,'EVIDENCE_RESPONSE_COMPLETE')
    return archive


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--step47',type=Path,help='Completed Step47 run folder or review ZIP')
    rules=parser.add_mutually_exclusive_group()
    rules.add_argument('--rule-run',type=Path,help='Original completed Step16 folder or ZIP')
    rules.add_argument('--rule-review',type=Path,help='Completed NOS3 evaluator ZIP/folder with verified benchmark clean Rule predictions')
    parser.add_argument('--output-root',type=Path,help='Separate output parent; original results are never modified')
    parser.add_argument('--preflight-only',action='store_true',help='Verify input hashes, alignment and eligibility; no outcome analysis')
    args=parser.parse_args(argv)
    root=args.project_root.expanduser().resolve()
    resolve=lambda p:(p if p.is_absolute() else root/p).resolve()
    out_root=resolve(args.output_root) if args.output_root else root/OUTPUT
    out=None; lock=None; bundles=[]
    started=now()
    print('Step 56: '+VERSION,flush=True)
    print('CPU only | Frozen clean predictions | No training/inference | No source result changes',flush=True)
    try:
        print('[1/4] Verify completed Step47 artifacts, clean predictions and causal observation alignment.',flush=True)
        data=load_step47(resolve(args.step47) if args.step47 else root/STEP47)
        bundles.append(data['bundle'])
        print('[2/4] Verify existing Rule_L10 and observation-only eligibility.',flush=True)
        rp,rb,rule_info=choose_rule(root,resolve(args.rule_run) if args.rule_run else None,
            resolve(args.rule_review) if args.rule_review else None,data)
        bundles.append(rb)
        data['predictions']=np.concatenate([data['predictions'],rp[None,:]],axis=0)
        rule_info['input_sha256']=dict(rb.observed)
        protocol=fixed_protocol(data,rule_info,sha_file(__file__))
        print('Eligibility: '+json.dumps(protocol['latency_population_counts'],sort_keys=True),flush=True)
        print('Latency endpoints begin at t0; confirmation uses the second correct prediction; horizon=10s.',flush=True)
        if args.preflight_only:
            print('PREFLIGHT_PASS. No response/sparse/false-alarm outcomes calculated; no output files written.',flush=True)
            return 0
        identity=object_hash(protocol)
        out=out_root/('run_response_v1_'+identity[:16])
        for b in bundles:
            require(out!=b.path and not out.is_relative_to(b.path),'Output must not be inside an input result folder')
        out.mkdir(parents=True,exist_ok=True)
        lock=out/'.run.lock'
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            lock=None
            raise ValueError('Another analysis may be running. If the previous process ended, remove only .run.lock in '+str(out))
        with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
        protocol_path=out/'protocol.json'
        if protocol_path.exists():
            require(json.loads(protocol_path.read_text(encoding='utf-8'))==protocol,'Output protocol differs')
        else:write_json(protocol_path,protocol)
        if (out/'run_manifest.json').exists():
            archive=verify_completed(out)
            print('Completed analysis verified; no recalculation.',flush=True)
        else:
            write_json(out/'input_audit.json',dict(step47=str(data['bundle'].path),rule=str(rb.path),
                step47_sha256=data['bundle'].observed,rule_sha256=rb.observed))
            shutil.copyfile(__file__,out/Path(__file__).name)
            print('[3/4] Calculate fixed response, sparse-observation recall and normal alarms.',flush=True)
            inv,vectors,latency,normal,sparse=analyze(data,out)
            print('[4/4] Paired episode statistics and verified review ZIP.',flush=True)
            comparisons=paired_statistics(inv,vectors,data['selected_pool'],out)
            for b in bundles:b.recheck()
            require(sha_file(__file__)==protocol['script_sha256'],'Code changed during analysis')
            summary=dict(status='EVIDENCE_RESPONSE_COMPLETE',code_version=VERSION,started_at_utc=started,
                finished_at_utc=now(),dataset_id=DATASET,protocol_identity=identity,
                clean_endpoints=len(data['y']),episodes=len(inv),latency_episodes=int(inv.latency_eligible.sum()),
                normal_episodes=int(inv.label_id.eq(0).sum()),trained_head_count=50,deterministic_controls=4,
                selection_source='Existing Step47 clean validation only',selected_pool=data['selected_pool'],
                eligibility_counts=protocol['latency_population_counts'],comparisons=comparisons,
                main_latency=json_records(latency[latency.label.eq('ALL_ATTACKS')]),
                normal_alarms=json_records(normal),new_training=False,new_inference=False,
                limitations=protocol['forbidden_interpretations']+[
                    'Retrospective follow-up analysis on an already inspected test set.',
                    '379 is not an independent new campaign; eligibility excludes early changes and short follow-up.',
                    'Pre-evidence correct predictions can precede first observable change; latency alone is not evidence responsiveness.',
                    'Normal episode event rates depend on observed episode duration; durations are identical across compared models.',
                    'An interval including zero does not establish equal false-alarm rates or equivalence.',
                    'Sparse-bin results are descriptive and may differ in class/episode composition.'])
            write_json(out/'summary.json',summary)
            archive=package(out,'EVIDENCE_RESPONSE_COMPLETE')
        print('STATUS=EVIDENCE_RESPONSE_COMPLETE',flush=True)
        print('ZIP_SHA256='+sha_file(archive),flush=True)
        print('UPLOAD_THIS='+str(archive),flush=True)
        return 0
    except Exception as exc:
        print('STOPPED: '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        if not args.preflight_only:
            failure=out_root/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
            try:
                failure.mkdir(parents=True,exist_ok=False)
                write_json(failure/'failure.json',dict(status='FAILED',code_version=VERSION,error_type=type(exc).__name__,
                    error=str(exc),started_at_utc=started,failed_at_utc=now(),project_root=str(root),partial_run=str(out) if out else None))
                (failure/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
                archive=package(failure,'FAILED')
                print('FAILURE_REVIEW_ZIP='+str(archive),flush=True)
            except OSError as package_error:
                print('Failure packaging also failed: '+str(package_error),file=sys.stderr)
        return 1
    finally:
        for b in bundles:b.close()
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__=='__main__':
    sys.exit(main())
