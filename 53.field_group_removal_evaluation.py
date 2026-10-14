#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 53: current shared_nominal_v2 test-time field-group sensitivity.

Run AFTER the current revised Step 47 completes:
    python "53.field_group_removal_evaluation.py"
Keep current Steps 11, 13, 14, 15 and 47 beside this file; Step51 is NOT required.
--preflight-only (alias --audit-only) checks sources, hashes, cohorts and the
completed Step47 handoff on CPU. No model loading, training or output changes.
Rerun the identical command to resume; completed stages are hash-verified.

Four removals: context(7), message/control(6), receiver(4), operational trace(5).
Replace every value in the selected group by literal UNK on EVERY test record.
Keep all 22 names/slots, records, episodes and causal stride-1 L10 endpoints,
including quiet contexts. Reuse the Step8 trained encoder and all ten EXISTING
Step11/12 TST heads. Authenticate Step47 clean predictions and replay each head
on full clean validation/test cohorts before any removal predictions.

Four contrasts: clean minus removal; mean per-head six-class macro-F1. Paired
class-stratified WHOLE-EPISODE bootstrap, 20,000 draws, fixed fitted heads;
Bonferroni family 4 yields 98.75% marginal intervals. Negative/zero drops remain.
This is trained-model sensitivity, distinct from Step51 train-time field
subsets and Step47 uniform Missing-8. Unequal group sizes do not establish
causal necessity, per-field importance, representation superiority or
cross-window aggregation superiority. Zero-containing CIs are not equivalence.
A revised Fig.5 caption must identify the current corpus/encoder, literal UNK,
all ten heads and this sensitivity estimand; legacy Fig.5 is not comparable.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
from pathlib import Path
import re
import shutil
import sys
import zipfile
import zlib
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
OUT_REL='outputs/shared_nominal_v2/53_field_group_removal/run_group_removal_v2'
VERSION='shared_nominal_test_time_field_group_removal_v2'
ENGINE_VERSION='shared_nominal_uniform_missing8_aggregation_v2'
FIELDS=('observation_stage','path_type','direction','source_role','destination_role',
    'protocol_context','operational_phase','message_type','service_context','command_intent',
    'payload_or_parameter_context','sequence_freshness_status','control_security_state',
    'signal_quality_state','tracking_stability','navigation_consistency','timing_consistency',
    'decoder_status','command_execution_result','retry_timeout_state','session_state','state_transition_consistency')
GROUPS={'remove_context':list(FIELDS[:7]),'remove_message':list(FIELDS[7:13]),
        'remove_receiver':list(FIELDS[13:17]),'remove_trace':list(FIELDS[17:])}
CONDITIONS=('clean',*GROUPS)
N_BOOTSTRAP=20000
BOOTSTRAP_SEED=2026100153
BOOTSTRAP_BATCH=100
e=None  # Current Step47 is imported only after parsing CLI; --help needs no models.


def require(ok,message):
    if not bool(ok):raise ValueError(message)


def load_engine(root,explicit=None):
    paths=[Path(explicit)] if explicit else [p for p in root.glob('*.py') if re.match(r'^47[. _-]',p.name)]
    unique={hashlib.sha256(p.read_bytes()).hexdigest():p.resolve() for p in paths if p.is_file()}
    require(len(unique)==1,'Place one current Step47 script beside Step53, or use --step47-code.')
    path=next(iter(unique.values()))
    spec=importlib.util.spec_from_file_location('step53_current47',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    require(module.PROTOCOL_VERSION==ENGINE_VERSION and tuple(module.FIELDS)==FIELDS,
            'Use current shared_nominal_v2 Step47 and the exact 22-field schema')
    require(len(module.SEEDS)==10 and len(set(module.SEEDS))==10 and len(module.LABELS)==6
            and len(set(module.LABELS))==6 and 'NORMAL' in module.LABELS,'Invalid current head seeds/class order')
    return module,path


def same_frame(actual,expected,message):
    require(list(actual)==list(expected) and actual.reset_index(drop=True).astype(str).equals(
        expected.reset_index(drop=True).astype(str)),message)


def audit_step47(root,folder,context,engine_path,data):
    """Inline completed47 audit; no dependency on Step48 or obsolete Step51."""
    helper=context['helper'];observed=context['observed'];folder=folder.resolve()
    manifest=helper.load_json(folder/'run_manifest.json',observed)
    require(manifest.get('status')=='MISSING8_AGGREGATION_COMPLETE',
            'Finish the current revised Step47 before Step53.')
    files=manifest.get('output_sha256',{})
    require(set(e.FINAL_FILES)<=set(files),'Step47 completion misses required evidence')
    helper.verify_selected(folder,files,list(files),observed)
    protocol=e.read_json(folder/'protocol.json')
    require(protocol.get('identity')==e.object_hash({k:v for k,v in protocol.items() if k!='identity'})
            ==manifest.get('protocol_identity'),'Step47 protocol identity differs')
    require(protocol.get('version')==ENGINE_VERSION and protocol.get('script_sha256')==e.digest(engine_path),
            'Current Step47 source does not match its completed run')
    require(protocol.get('source_contract')==context['source_contract']
            and protocol.get('checkpoint_contracts')==context['heads']
            and protocol.get('source_hashes')==context['code_hashes']
            and protocol.get('audit_script_sha256')==e.digest(context['helper_path']),
            'Step47 used different data/models/source contracts')
    require(protocol.get('runs')==[list(r) for r in e.RUNS]
            and protocol.get('fields')==list(FIELDS) and protocol.get('labels')==list(e.LABELS)
            and protocol.get('head_seeds')==list(e.SEEDS) and protocol.get('conditions')==list(e.CONDITIONS),
            'Step47 labels/seeds/conditions/order differ')
    require(e.read_json(folder/'input_audit.json')==protocol['input_audit'],'Step47 input audit differs')
    for name,sha in protocol['input_audit'].items():
        path=Path(name)
        require(path.is_file() and e.digest(path)==sha,'Step47 input changed: '+name)
        observed[str(path.resolve())]=sha
    helper.track_file(engine_path,observed)
    require(any(name.startswith('source/') and re.match(r'^47[. _-]',Path(name).name)
                and sha==e.digest(engine_path) for name,sha in files.items()),
            'Step47 archived source is missing or differs')
    for split in ('val','test'):
        cohort=data[split];record=protocol['populations'][split]
        expected=dict(windows=len(cohort['frame']),episodes=int(cohort['frame'].episode_id.nunique()),
            l10=len(cohort['y']),eligible_episodes=int(cohort['meta'].episode_id.nunique()),
            endpoint_identity=e.object_hash(cohort['meta'].to_dict(orient='records')))
        require(record==expected,'Step47 endpoint population differs: '+split)
        e.validate_indices(cohort['frame'],cohort['indices'],cohort['y'])
    same_frame(helper.read_csv(folder/'evaluation_endpoints_L10.csv'),data['test']['meta'],
               'Step47 endpoint metadata/order differs')
    same_frame(helper.read_csv(folder/'clean_context_observable_effects_L10.csv'),
               pd.concat([data['test']['meta'],data['test']['effects']],axis=1),
               'Step47 clean-context diagnostic alignment differs')
    selection=e.read_json(folder/'validation_selection.json')
    validation=helper.read_csv(folder/'validation_metrics.csv')
    pool=e.choose_pool(validation.to_dict('records'))
    rows=selection['ewma_validation_scores'];scores={float(r['alpha']):float(r['macro_f1']) for r in rows}
    require(len(rows)==len(e.ALPHAS) and set(scores)==set(e.ALPHAS),'Step47 EWMA candidate grid differs')
    require(selection['selected_pool']==pool and selection['ewma_alpha']==e.choose_alpha(scores)
            and selection.get('all_ten_head_seeds_retained') is True,'Step47 validation selection differs')
    summary=helper.read_csv(folder/'model_summary.csv')
    require(set(summary.family)==set(e.FAMILIES) and not summary.family.duplicated().any(),
            'Step47 summary family inventory differs')
    with np.load(folder/'predictions.npz',allow_pickle=False) as saved:
        pred=saved['predictions'];y=saved['y']
        require(pred.shape==(len(e.CONDITIONS),len(e.RUNS),len(data['test']['y']))
                and pred.dtype.kind in 'iu' and np.isin(pred,np.arange(len(e.LABELS))).all(),
                'Invalid completed Step47 prediction grid')
        require(np.array_equal(y,data['test']['y']) and np.array_equal(saved['conditions'],e.CONDITIONS)
                and np.array_equal(saved['families'],[f for f,s in e.RUNS])
                and np.array_equal(saved['seeds'],[s if s is not None else -1 for f,s in e.RUNS]),
                'Step47 prediction truth/condition/head order differs')
        require(saved['dataset_id'].shape==() and saved['dataset_id'].item()==context['source_contract']['dataset_id']
                and saved['step47_protocol_identity'].shape==()
                and saved['step47_protocol_identity'].item()==protocol['identity'],
                'Step47 prediction source identity differs')
        slots=[e.RUNS.index(('TST',seed)) for seed in e.SEEDS]
        clean=pred[0,slots].copy()
    for si,seed in enumerate(e.SEEDS):
        ref=data['test']['references'][e.run_key('TST',seed)]
        require(np.array_equal(clean[si],ref.argmax(1)),'Step47 clean TST decisions differ from authenticated upstream heads')
    score=np.mean([e.metrics(data['test']['y'],p)['macro_f1'] for p in clean])
    require(np.isclose(score,float(summary.set_index('family').loc['TST','clean_macro_f1']),rtol=0,atol=1e-12),
            'Step47 clean TST summary differs')
    helper.recheck_inputs(observed)
    return dict(folder=str(folder),protocol=protocol,selection=selection,clean=clean,
                manifest_sha256=e.digest(folder/'run_manifest.json'))


def apply_removal(frame,condition):
    require(condition in GROUPS,'Unknown removal condition')
    require(frame.columns.is_unique and set(FIELDS)<=set(frame.columns),'Observable fields missing or duplicated')
    fields=GROUPS[condition];changed=int(frame[fields].ne('UNK').to_numpy().sum())
    masked=frame.copy(deep=True);masked.loc[:,fields]='UNK'
    require(masked[fields].eq('UNK').all().all(),'Incomplete group removal')
    require(masked.drop(columns=fields).equals(frame.drop(columns=fields)),
            'Removal changed metadata or retained fields')
    require(masked.columns.equals(frame.columns) and masked.index.equals(frame.index),'Removal changed rows or slots')
    return masked,dict(condition=condition,rows=len(frame),removed_fields=fields,n_removed_fields=len(fields),
        masked_cells=len(frame)*len(fields),newly_changed_cells=changed,
        already_UNK_cells=len(frame)*len(fields)-changed,replacement='UNK',
        all_records_masked=True,uses_label=False,retained_fields_and_metadata_unchanged=True)


def validate_predictions(predictions,n):
    pred=np.asarray(predictions)
    require(n>0 and pred.shape==(len(CONDITIONS),len(e.SEEDS),n) and pred.dtype.kind in 'iu'
            and np.isin(pred,np.arange(len(e.LABELS))).all(),'Invalid prediction grid')


def compute_statistics(meta,y,predictions,n_draws=None,progress=True):
    y=np.asarray(y);validate_predictions(predictions,len(y))
    require(y.dtype.kind in 'iu' and set(y)==set(range(len(e.LABELS))) and len(meta)==len(y)
            and np.array_equal(meta.label.map(dict(zip(e.LABELS,range(len(e.LABELS))))),y),
            'Statistics endpoint labels/classes differ')
    counts,strata=e.episode_counts(y,meta.episode_id.to_numpy(),predictions)
    point=e.macro_f1(counts.sum(0)).mean(1)
    samples=e.bootstrap_scores(counts,strata,N_BOOTSTRAP if n_draws is None else n_draws,
        BOOTSTRAP_SEED,BOOTSTRAP_BATCH,progress).mean(-1)
    return point,samples,counts,strata


def make_statistics(out,data,predictions):
    y=data['test']['y'];meta=data['test']['meta'];n=len(y)
    points,draws,counts,strata=compute_statistics(meta,y,predictions)
    total=counts.sum(0);f1=e.macro_f1(total)
    accuracy=np.trace(total,axis1=-2,axis2=-1)/n
    normal=list(e.LABELS).index('NORMAL');normal_total=total[:,:,normal].sum(-1)
    normal_fpr=(normal_total-total[:,:,normal,normal])/normal_total
    rows=[];seed_rows=[];class_rows=[];conf_rows=[]
    for ci,condition in enumerate(CONDITIONS):
        lo,hi=np.quantile(draws[:,ci],[.025,.975],method='linear')
        rows.append(dict(condition=condition,n_removed_fields=len(GROUPS.get(condition,[])),head_seeds=len(e.SEEDS),
            n_endpoints=n,n_episodes=len(counts),macro_f1_mean=points[ci],macro_f1_head_seed_sd_ddof1=f1[ci].std(ddof=1),
            ci95_low=lo,ci95_high=hi,accuracy_mean=accuracy[ci].mean(),normal_fpr_mean=normal_fpr[ci].mean(),
            delta_macro_f1_clean_minus_condition=points[0]-points[ci]))
        for si,seed in enumerate(e.SEEDS):
            seed_rows.append(dict(condition=condition,seed=seed,macro_f1=f1[ci,si],
                accuracy=accuracy[ci,si],normal_fpr=normal_fpr[ci,si]))
        cms=total[ci];tp=np.diagonal(cms,axis1=-2,axis2=-1);truth=cms.sum(-1);pred=cms.sum(-2)
        precision=np.divide(tp,pred,out=np.zeros_like(tp,dtype=float),where=pred>0)
        recall=np.divide(tp,truth,out=np.zeros_like(tp,dtype=float),where=truth>0)
        cf=np.divide(2*tp,truth+pred,out=np.zeros_like(tp,dtype=float),where=truth+pred>0)
        norm=(cms/truth[...,None]).mean(0)
        for li,label in enumerate(e.LABELS):
            class_rows.append(dict(condition=condition,label=label,support=int(truth[0,li]),precision_mean=precision[:,li].mean(),
                recall_mean=recall[:,li].mean(),f1_mean=cf[:,li].mean(),f1_head_seed_sd_ddof1=cf[:,li].std(ddof=1)))
            for pi,pred_label in enumerate(e.LABELS):
                conf_rows.append(dict(condition=condition,true_label=label,predicted_label=pred_label,
                    head_seed_mean_row_normalized=norm[li,pi]))
    comparisons=[]
    for ci,condition in enumerate(CONDITIONS[1:],1):
        lo,hi,adjlo,adjhi=np.quantile(draws[:,0]-draws[:,ci],[.025,.975,.00625,.99375],method='linear')
        comparisons.append(dict(contrast='clean_minus_'+condition,delta_macro_f1=points[0]-points[ci],
            delta_percentage_points=100*(points[0]-points[ci]),ci95_low=lo,ci95_high=hi,
            bonferroni_low=adjlo,bonferroni_high=adjhi,family_size=4,marginal_confidence=.9875,
            direction='clean_higher' if adjlo>0 else 'removed_higher' if adjhi<0 else 'zero_included_not_equivalence'))
    inventory=meta.assign(y=y).groupby('episode_id',sort=True).agg(label_id=('y','first'),n_endpoints=('y','size')).reset_index()
    e.write_csv(out/'model_summary.csv',rows);e.write_csv(out/'metrics_by_seed.csv',seed_rows)
    e.write_csv(out/'paired_comparisons.csv',comparisons);e.write_csv(out/'per_class_metrics.csv',class_rows)
    e.write_csv(out/'row_normalized_confusion.csv',conf_rows);e.write_csv(out/'episode_inventory.csv',inventory)
    e.write_array(out/'episode_confusions.npz',counts=counts,episodes=inventory.episode_id.to_numpy(dtype=str),class_strata=strata)
    e.write_array(out/'bootstrap_distribution.npz',macro_f1_seed_mean=draws,conditions=np.array(CONDITIONS))
    e.write_json(out/'statistics_audit.json',dict(draws=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,batch_size=BOOTSTRAP_BATCH,
        unit='class-stratified paired whole episode; complete overlapping L10 endpoints retained',
        estimand='per-head endpoint six-class macro-F1 then equal mean of ten heads; not ensemble or pooled-seed F1',
        CI_scope='conditional on frozen encoder, ten fitted heads, this shared_nominal_v2 realization and four declared groups',
        multiplicity='four clean-minus-group contrasts; 98.75% marginal Bonferroni linear-percentile intervals',
        head_seed_resampling=False,group_resampling=False,quiet_endpoint_filtering=False,
        equivalence_claim=False,causal_necessity_claim=False,per_field_importance_claim=False,groups_equal_size=False))
    (out/'analysis_summary.txt').write_text(
        'Step53: four test-time field-group removals on current shared_nominal_v2. No training or selection.\n'
        'All 22 field names/slots, test records, episodes and eligible L10 endpoints retained, including quiet contexts.\n'
        'Literal UNK replaces all values in a group; this is not tokenizer [UNK] and not column deletion.\n'
        'Clean reference: authenticated current Step47 TST predictions, after exact clean decision/probability replay.\n'
        'Four clean-minus-removal mean per-head six-class macro-F1 contrasts; negative drops retained.\n'
        '20,000 class-stratified paired whole-episode draws; 98.75% marginal Bonferroni intervals (family 4).\n'
        'Conditional on encoder, fitted heads, corpus and masks; no head/seed/group resampling.\n'
        'Distinct from Step51 train-time field subsets and Step47 uniform Missing-8. Unequal group sizes.\n'
        'No causal necessity, per-field importance, raw/canonical or cross-window aggregation superiority claim.\n'
        'A CI including zero does not demonstrate equivalence. Clean context effect strata remain diagnostics only;\n'
        'they do not identify attack evidence surviving removal. Update Fig.5 caption/protocol with the current\n'
        'corpus, Step8 encoder, ten TST heads and literal UNK; legacy Fig.5 values are not directly comparable.\n',encoding='utf-8')


def make_protocol(context,data,previous,engine_path):
    body=dict(version=VERSION,script_sha256=e.digest(Path(__file__)),engine_sha256=e.digest(engine_path),
        source_hashes=context['code_hashes'],audit_script_sha256=e.digest(context['helper_path']),
        source_contract=context['source_contract'],checkpoint_contracts=context['heads'],input_audit=dict(context['observed']),
        original_step47_identity=previous['protocol']['identity'],original_step47_manifest_sha256=previous['manifest_sha256'],
        original_step47_folder=previous['folder'],expected_inference_runtime=previous['protocol']['runtime'],
        inference=previous['protocol']['inference'],fields=list(FIELDS),labels=list(e.LABELS),head_seeds=list(e.SEEDS),
        conditions=list(CONDITIONS),groups=GROUPS,populations=previous['protocol']['populations'],
        clean_source='authenticated completed Step47, verified against all ten original Step11/12 TST heads',
        clean_replay='full validation and test probabilities/decisions before any removal; batch-aligned Step9 encoder canary',
        mask_rule='literal UNK on every selected-group field value on EVERY test row; retain all 22 names/slots and metadata',
        training=False,new_model_selection=False,test_used_for_selection=False,quiet_endpoint_filtering=False,
        legacy_Fig5_equivalent_protocol=False,retrospective_revision_extension=True,
        scope='trained-model test-time group sensitivity; distinct from Step51 train-time subsets and Step47 uniform Missing-8',
        interpretation='unequal sizes; no causal necessity/per-field importance or representation/aggregation superiority; zero CI is not equivalence',
        diagnostics='original clean-context projection-change strata only; never post-removal evidence or selection',
        bootstrap=dict(draws=N_BOOTSTRAP,seed=BOOTSTRAP_SEED,batch=BOOTSTRAP_BATCH,
            unit='class-stratified whole episode, same draw for every condition and head',
            estimand='mean of per-head six-class endpoint macro-F1',contrasts='clean minus each of four removals',
            family_size=4,marginal_confidence=.9875,percentile_method='linear',head_seed_resampling=False,
            conditional_on='fixed encoder, ten fitted heads, current corpus realization and declared groups'))
    return dict(identity=e.object_hash(body),**body)


def validate_output_location(out,protected):
    out=out.resolve()
    for directory in protected:
        path=Path(directory).resolve()
        require(out!=path and out not in path.parents and path not in out.parents,
                'Output root overlaps protected inputs: '+str(path))


def import_tst(context):
    path=context['code_paths']['TST']
    require(e.digest(path)==context['code_hashes']['TST'],'TST source changed before import')
    return {'TST':e.import_script(path,'step53_original_tst')}


def infer(root,out,identity,context,data,previous):
    modules=import_tst(context)
    for seed in e.SEEDS:
        path=Path(e.paths_for('TST',seed,context)[0])
        context['helper'].track_file(path,context['observed'],context['observed'][str(path.resolve())])
        e.validate_checkpoint(e.read_checkpoint(path),'TST',seed,context)
    tokenizer,encoder=e.load_encoder(root,context)
    folder=out/'cache/encoder_canary';folder.mkdir(parents=True,exist_ok=True)
    contract=e.object_hash(dict(protocol=identity,stage='encoder_canary'))
    canary=e.read_completion(folder,'complete.json',contract)
    n=e.canary_count(len(data['val']['frame']),context['encoder_batch_size'])
    if canary is None:
        fresh,info=e.encode_records(data['val']['frame'].iloc[:n],tokenizer,encoder,context)
        ref=data['val']['embeddings'][:n]
        require(fresh.shape==ref.shape and fresh.dtype==np.float32 and np.isfinite(fresh).all()
                and np.allclose(fresh,ref,rtol=1e-5,atol=2e-5),'Step9 encoder canary differs; stop before removals')
        canary=dict(status='PASS',rows=n,max_abs_error=float(np.max(np.abs(fresh-ref))),encoding=info)
        e.write_json(folder/'evidence.json',canary)
        e.write_completion(folder,'complete.json',contract,canary,['evidence.json'])
    require(canary.get('status')=='PASS' and canary.get('rows')==n,'Invalid encoder canary evidence')
    e.save_matching_json(out/'encoder_canary.json',canary)
    # All twenty replay calls finish before creating any removal embeddings/predictions.
    clean_evidence=[]
    for split in ('val','test'):
        for si,seed in enumerate(e.SEEDS):
            print(f'Clean {split} replay: TST seed={seed}',flush=True)
            p=e.head_probabilities(root,out,context,modules,'TST',seed,data[split],data[split]['embeddings'],split,'clean',identity)
            require(p.shape==(len(data[split]['y']),len(e.LABELS)),'Clean replay endpoint shape differs')
            if split=='test':require(np.array_equal(p.argmax(1),previous['clean'][si]),'Step47 clean replay differs')
            clean_evidence.append(dict(split=split,seed=seed,**e.verify_clean_replay(p,data[split],'TST',seed)))
    e.save_matching_json(out/'clean_replay.json',dict(status='PASS',heads=len(e.SEEDS),splits=['val','test'],records=clean_evidence))
    predictions=np.empty((len(CONDITIONS),len(e.SEEDS),len(data['test']['y'])),dtype=np.int8)
    predictions[0]=previous['clean'];mask_audits=[]
    for ci,condition in enumerate(CONDITIONS[1:],1):
        print('Removal: '+condition,flush=True)
        masked,mask_audit=apply_removal(data['test']['frame'],condition)
        folder=out/'cache'/condition;folder.mkdir(parents=True,exist_ok=True)
        contract=e.object_hash(dict(protocol=identity,stage=condition))
        done=e.read_completion(folder,'embeddings_complete.json',contract)
        if done is None:
            embeddings,encoding=e.encode_records(masked,tokenizer,encoder,context,
                lambda n,t:print(f'  {condition}: {n:,}/{t:,} records',flush=True))
            require(embeddings.shape==(len(masked),768) and embeddings.dtype==np.float32
                    and np.isfinite(embeddings).all(),'Invalid removal embeddings')
            done=dict(**mask_audit,encoding=encoding)
            e.write_array(folder/'embeddings.npy',value=embeddings);e.write_json(folder/'mask_audit.json',done)
            e.write_completion(folder,'embeddings_complete.json',contract,done,['embeddings.npy','mask_audit.json'])
        else:embeddings=np.load(folder/'embeddings.npy',mmap_mode='r',allow_pickle=False)
        require(embeddings.shape==(len(masked),768) and embeddings.dtype==np.float32
                and np.isfinite(embeddings).all(),'Invalid cached removal embedding shape/precision/values')
        require(all(done.get(k)==v for k,v in mask_audit.items())
                and e.read_json(folder/'mask_audit.json')==done,'Cached removal mask audit differs')
        mask_audits.append(done)
        for si,seed in enumerate(e.SEEDS):
            p=e.head_probabilities(root,out,context,modules,'TST',seed,data['test'],embeddings,'test',condition,identity)
            require(p.shape==(len(data['test']['y']),len(e.LABELS)),'Removal probability endpoint shape differs')
            predictions[ci,si]=p.argmax(1)
        del embeddings,masked
    validate_predictions(predictions,len(data['test']['y']))
    e.write_json(out/'mask_audits.json',mask_audits)
    return predictions


FINAL_FILES=['protocol.json','input_audit.json','inference_runtime.json','encoder_canary.json','clean_replay.json',
    'mask_audits.json','predictions.npz','evaluation_endpoints_L10.csv','clean_context_observable_effects_L10.csv',
    'test_window_alignment.csv','model_summary.csv','metrics_by_seed.csv','paired_comparisons.csv','per_class_metrics.csv',
    'row_normalized_confusion.csv','episode_inventory.csv','episode_confusions.npz','bootstrap_distribution.npz',
    'statistics_audit.json','analysis_summary.txt','inference_sources_verified.json',
    'reference47/run_manifest.json','reference47/model_summary.csv','reference47/validation_selection.json']


def verify_caches(out,identity,data,context,complete=False):
    """Authenticate omitted caches as well as their archived completion records."""
    expected=[]
    expected.append(('cache/encoder_canary','complete.json',
        e.object_hash(dict(protocol=identity,stage='encoder_canary')),'encoder',None,None))
    for split in ('val','test'):
        for seed in e.SEEDS:
            relative=f'cache/{split}/clean/'+e.run_key('TST',seed)
            expected.append((relative,'complete.json',e.object_hash(dict(protocol=identity,stage=relative)),
                             'probabilities',split,seed))
    for condition in CONDITIONS[1:]:
        expected.append(('cache/'+condition,'embeddings_complete.json',
            e.object_hash(dict(protocol=identity,stage=condition)),'embeddings',condition,None))
        for seed in e.SEEDS:
            relative=f'cache/test/{condition}/'+e.run_key('TST',seed)
            expected.append((relative,'complete.json',e.object_hash(dict(protocol=identity,stage=relative)),
                             'probabilities','test',seed))
    inventory=[]
    for relative,name,contract,kind,split,seed in expected:
        folder=out/relative;record=e.read_completion(folder,name,contract)
        require(not complete or record is not None,'Missing completed cache: '+relative)
        if record is None:continue
        inventory.append(relative+'/'+name)
        required={'encoder':{'evidence.json'},'embeddings':{'embeddings.npy','mask_audit.json'},
                  'probabilities':{'probabilities.npy'}}[kind]
        require(set(e.read_json(folder/name)['artifact_sha256'])==required,
                'Cache artifact binding inventory differs: '+relative)
        if kind=='encoder':
            require(record.get('status')=='PASS' and record.get('rows')==e.canary_count(
                len(data['val']['frame']),context['encoder_batch_size']),
                'Invalid cached encoder replay evidence')
            require(e.read_json(folder/'evidence.json')==record,'Cached encoder evidence differs')
        elif kind=='embeddings':
            embeddings=np.load(folder/'embeddings.npy',mmap_mode='r',allow_pickle=False)
            require(embeddings.shape==(len(data['test']['frame']),768) and embeddings.dtype==np.float32
                    and np.isfinite(embeddings).all(),'Invalid cached removal embedding shape/precision/values')
            _,expected_mask=apply_removal(data['test']['frame'],split)
            require(all(record.get(k)==v for k,v in expected_mask.items())
                    and e.read_json(folder/'mask_audit.json')==record,'Cached removal mask audit differs')
        else:
            probs=e.check_probs(np.load(folder/'probabilities.npy',mmap_mode='r',allow_pickle=False))
            require(probs.dtype==np.float32 or probs.dtype==np.float64,'Invalid cached probability dtype')
            require(probs.shape==(len(data[split]['y']),len(e.LABELS)),'Cached probability endpoint shape differs')
            if '/clean/' in relative:e.verify_clean_replay(probs,data[split],'TST',seed)
    if complete:
        require(e.read_completion(out,'inference_sources_verified.json',identity) is not None,
                'Missing inference provenance completion')
    return inventory


def package_review(out,manifest):
    archive=out.with_name(out.name+'_review.zip');temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as saved:
        for name in [*manifest['output_sha256'],'run_manifest.json']:
            saved.write(e.safe_artifact(out,name),name)
    with zipfile.ZipFile(temporary) as saved:require(saved.testzip() is None,'Review archive CRC failed')
    e.replace_atomic(temporary,archive)
    return archive


def verify_final(out,identity,data,context):
    manifest=e.read_json(out/'run_manifest.json')
    require(manifest.get('status')=='GROUP_REMOVAL_COMPLETE' and manifest.get('protocol_identity')==identity,
            'Incomplete or mismatched completed Step53 run')
    hashes=manifest.get('output_sha256',{})
    require(set(FINAL_FILES)<=set(hashes),'Incomplete final artifact inventory')
    for name,sha in hashes.items():
        path=e.safe_artifact(out,name)
        require(path.is_file() and e.digest(path)==sha,'Completed review artifact changed/missing: '+name)
    cache_records=verify_caches(out,identity,data,context,complete=True)
    require(set(cache_records)<=set(hashes),'Completed cache records missing from final inventory')
    archive=out.with_name(out.name+'_review.zip');valid=False
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as saved:
                names=saved.namelist()
                valid=(len(names)==len(set(names)) and set(names)==set(hashes)|{'run_manifest.json'}
                    and saved.read('run_manifest.json')==(out/'run_manifest.json').read_bytes()
                    and saved.testzip() is None)
                # CRC + names alone would accept a different, internally valid ZIP payload.
                if valid:
                    for name,sha in hashes.items():
                        h=hashlib.sha256()
                        with saved.open(name) as stream:
                            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                        if h.hexdigest()!=sha:valid=False;break
        except (OSError,KeyError,zipfile.BadZipFile,RuntimeError,zlib.error):valid=False
    if not valid:
        print('Rebuilding review archive from verified outputs; no inference/statistics rerun.',flush=True)
        return package_review(out,manifest)
    return archive


def finish_run(out,protocol):
    names=list(FINAL_FILES)
    names+=['source/'+p.name for p in sorted((out/'source').iterdir()) if p.is_file()]
    names+=[p.relative_to(out).as_posix() for p in sorted((out/'cache').rglob('*complete.json'))]
    manifest=dict(status='GROUP_REMOVAL_COMPLETE',protocol_identity=protocol['identity'],protocol_version=VERSION,
        completed_at_utc=e.utc(),dataset_id=protocol['source_contract']['dataset_id'],training=False,
        test_used_for_selection=False,head_seeds=len(e.SEEDS),new_removal_conditions=len(GROUPS),
        clean_predictions_reused=True,output_sha256={name:e.digest(e.safe_artifact(out,name)) for name in names},
        review_archive_omits='Model weights and raw probability/embedding caches; all decisions and sufficient statistics included.')
    e.write_json(out/'run_manifest.json',manifest)
    return package_review(out,manifest)


def copy_matching(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():require(e.digest(target)==e.digest(source),'Saved source/reference changed: '+str(target))
    else:shutil.copyfile(source,target)


def execute(root,out,context,data,previous,engine_path):
    context['helper'].track_file(Path(__file__),context['observed'])
    context['helper'].recheck_inputs(context['observed'])
    protocol=make_protocol(context,data,previous,engine_path);identity=protocol['identity']
    out.mkdir(parents=True,exist_ok=True)
    e.initialize_protocol(out,protocol)
    if (out/'run_manifest.json').exists():return verify_final(out,identity,data,context)
    verify_caches(out,identity,data,context)
    runtime=e.evaluation_runtime(context)
    require(runtime==protocol['expected_inference_runtime'],
            'Runtime differs from completed Step47; use the same environment/device/settings')
    e.save_matching_json(out/'input_audit.json',context['observed'])
    e.save_matching_json(out/'inference_runtime.json',runtime)
    for path in (Path(__file__),engine_path,context['helper_path'],context['code_paths']['TST']):
        copy_matching(path,out/'source'/path.name)
    for name in ('run_manifest.json','model_summary.csv','validation_selection.json'):
        copy_matching(Path(previous['folder'])/name,out/'reference47'/name)
    predictions=infer(root,out,identity,context,data,previous)
    e.write_array(out/'predictions.npz',predictions=predictions,y=data['test']['y'],conditions=np.array(CONDITIONS),
        seeds=np.array(e.SEEDS),labels=np.array(e.LABELS),dataset_id=np.array(context['source_contract']['dataset_id']),
        step53_protocol_identity=np.array(identity),step47_protocol_identity=np.array(previous['protocol']['identity']))
    e.write_csv(out/'evaluation_endpoints_L10.csv',data['test']['meta'])
    e.write_csv(out/'clean_context_observable_effects_L10.csv',pd.concat([data['test']['meta'],data['test']['effects']],axis=1))
    meta=data['test']['frame'][list(e.META)].copy();meta.insert(0,'embedding_row_index',np.arange(len(meta)))
    e.write_csv(out/'test_window_alignment.csv',meta)
    context['helper'].recheck_inputs(context['observed'])
    require(e.digest(Path(__file__))==protocol['script_sha256'] and e.digest(engine_path)==protocol['engine_sha256'],
            'Evaluation code changed while running')
    e.write_completion(out,'inference_sources_verified.json',identity,dict(status='PASS',heads=len(e.SEEDS),conditions=len(CONDITIONS)),
        ['predictions.npz','evaluation_endpoints_L10.csv','clean_context_observable_effects_L10.csv','mask_audits.json','clean_replay.json'])
    make_statistics(out,data,predictions)
    context['helper'].recheck_inputs(context['observed'])
    return finish_run(out,protocol)


def main(argv=None):
    global e
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=BASE)
    parser.add_argument('--step47-code',type=Path,help='Current completed Step47 source')
    parser.add_argument('--step47-run',type=Path,help='Completed current Step47 output folder')
    parser.add_argument('--audit-script',type=Path,help='Current revised Step15 validator; no Step15 output dependency')
    parser.add_argument('--tst-root',type=Path,help='Current Step12 multiseed output root')
    parser.add_argument('--pooling-root',type=Path,help='Current Step13 output root used by Step47')
    parser.add_argument('--lstm-root',type=Path,help='Current Step14 output root used by Step47')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight-only','--audit-only',dest='preflight_only',action='store_true',
        help='CPU source/cohort/completed47 audit only; no neural loading or output changes')
    args=parser.parse_args(argv);root=args.project_root.expanduser().resolve()
    def resolved(value):return (root/value.expanduser()).resolve() if value is not None else None
    e,engine_path=load_engine(root,resolved(args.step47_code))
    previous_folder=resolved(args.step47_run) or root/e.OUT_REL
    out=resolved(args.output) or root/OUT_REL
    context=e.audit_sources(root,resolved(args.audit_script),resolved(args.tst_root),resolved(args.pooling_root),resolved(args.lstm_root))
    protected=[root/e.DATA_REL,root/e.ENC_REL,root/e.EMB_REL,root/e.SEQ_REL,previous_folder]
    protected.extend(context['roots'].values())
    protected.extend(Path(h['checkpoint']).parent for h in context['heads'].values())
    protected.extend(Path(p).parent for h in context['heads'].values() for p in h['predictions'].values())
    validate_output_location(out,protected)
    data=e.load_cohorts(root,context)
    previous=audit_step47(root,previous_folder,context,engine_path,data)
    print(f"Step53 PREFLIGHT PASS: {len(data['test']['frame']):,} test records; {len(data['test']['y']):,} L10 endpoints; all ten TST heads.",flush=True)
    if args.preflight_only:
        print('No neural loading/inference. Tensor restore and numerical clean replay remain for the full run.',flush=True)
        return 0
    out.mkdir(parents=True,exist_ok=True)
    with e.RunLock(out/'.run.lock'):archive=execute(root,out,context,data,previous,engine_path)
    print('GROUP_REMOVAL_COMPLETE. Upload this compact review ZIP:\n'+str(archive),flush=True)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print(f'\nSTOPPED: {type(exc).__name__}: {exc}\nOriginal inputs were not modified. Resolve the cause and rerun the same command; completed stages are verified and reused.',file=sys.stderr,flush=True)
        raise
