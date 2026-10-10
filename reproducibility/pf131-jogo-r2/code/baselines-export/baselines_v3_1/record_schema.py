"""Strict result schema and self-consistency checks. No manual table editing."""
from __future__ import annotations
import math
import numpy as np
from evaluation import target_met
from lj_oracle import LJOracle
from protocol import digest,make_seed,attempt_seed,KNOWN_MINIMA

SCHEMA='jogo-baselines-v3.0'
CENSORED={'time_limit','restart_limit','single_attempt_complete'}
INVALID={'numerical_error','audit_failed','interrupted'}
NORMAL={'target_reached','evaluation_limit'}

def seal(record):
    record.pop('record_sha256',None)
    record['record_sha256']=digest(record)
    return record

def validate_record(r,*,verify_energy=True):
    errors=[]
    def require(test,msg):
        if not test:errors.append(msg)
    try:
        require(r['schema']==SCHEMA,'schema mismatch')
        unsigned=dict(r);unsigned.pop('record_sha256',None)
        require(r.get('record_sha256')==digest(unsigned),'record checksum mismatch')
        cfg=r['identity'];require(r['identity_sha256']==digest(cfg),'identity checksum mismatch')
        require(type(r['n_atoms']) is int and type(r['run_id']) is int,'invalid instance/run types')
        n=int(r['n_atoms']);run=int(r['run_id']);method=r['method']
        require(n>=2 and run>=1,'invalid instance/run range')
        require(type(r['success']) is bool and type(r['budget_exhausted']) is bool,'flags must be Boolean')
        require(n==cfg['n_atoms'] and run==cfg['run_id'] and method==cfg['method'],'identity mismatch')
        count=r['n_func_evals'];limit=r['max_evals']
        require(type(count) is int and type(limit) is int and 0<=count<=limit,'invalid evaluation count')
        require(limit==cfg['budget'],'budget mismatch')
        require(r['method_params']==cfg['method_params'],'effective parameters mismatch')
        counts=r['counters']
        require(counts['objective_global']+counts['objective_local']==count,'origin counter sum mismatch')
        require(all(type(counts[k]) is int and counts[k]>=0 for k in ('objective_global','objective_local','gradient_calls','audit_objective_evaluations','nonfinite_energy_evaluations')),'invalid subcounter')
        reason=r['stop_reason'];require(reason in NORMAL|CENSORED|INVALID,'unknown stop reason')
        require(bool(r['budget_exhausted'])==(count==limit),'budget flag mismatch')
        require(reason!='evaluation_limit' or count==limit,'evaluation limit before budget spent')
        require(r['target_energy']==cfg['target_energy'] and r['target_tolerance']==cfg['target_tolerance'],'target mismatch')
        if n in KNOWN_MINIMA:require(r['target_energy']==KNOWN_MINIMA[n],'unexpected reference target')
        require(r['base_seed']==cfg['base_seed'],'base seed mismatch')
        require(r['seed']==make_seed(cfg['base_seed'],n,run,1000),'run seed mismatch')
        require(r['wall_time']>=0 and math.isfinite(r['wall_time']),'invalid elapsed time')
        attempts=r['attempts'];require(r['n_attempts']==len(attempts),'attempt count mismatch')
        require(r['n_restarts']==max(0,len(attempts)-1),'restart count mismatch')
        last=0
        for k,a in enumerate(attempts,1):
            require(a['attempt']==k and a['seed']==attempt_seed(r['seed'],n,k),'attempt seed/index mismatch')
            require(a['eval_start']==last and a['eval_end']>=last,'attempt counter discontinuity')
            last=a['eval_end']
        require(last==count,'attempt total mismatch')
        e=r['energy'];good=e is not None and math.isfinite(e)
        expected=bool(good and r.get('audit',{}).get('passed',False) and target_met(e,r['target_energy'],r['target_tolerance']))
        require(bool(r['success'])==expected,'success flag inconsistent with audited energy')
        require(reason!='target_reached' or expected,'target stop without valid success')
        require(reason!='target_reached' or r['first_target_eval'] is not None,'missing first target crossing')
        if r['first_target_eval'] is not None:require(1<=r['first_target_eval']<=count,'invalid target crossing counter')
        if good:
            x=np.asarray(r['best_x'],dtype=float)
            require(x.shape==(3*n,) and np.isfinite(x).all(),'invalid final coordinates')
            require(r['audit']['passed'] is True or reason in INVALID,'failed final audit on non-error result')
            require(r['audit']['energy']==e,'audited energy mismatch')
            require(r['audit']['search_energy']==r['search_energy'],'search energy mismatch')
            require(1<=r['best_eval']<=count,'invalid incumbent evaluation index')
            require(counts['audit_objective_evaluations']==1,'expected one reporting audit')
            require(r['target_energy'] is None or e>=r['target_energy']-r['target_tolerance'] or reason in INVALID,'energy significantly below reference')
            require(abs(float(r['gap'])-(e-r['target_energy']))<=1e-10 if r['target_energy'] is not None else r['gap'] is None,'gap mismatch')
            if verify_energy and x.shape==(3*n,) and np.isfinite(x).all():
                value=LJOracle(n,cfg['potential']).reference(x)
                require(abs(value-e)<=1e-8+abs(value)*1e-12,'coordinate/energy mismatch')
                require(np.max(np.abs(x))<=cfg['bounds']+1e-12,'coordinates outside box')
        else:
            require(reason in INVALID,'missing finite energy on non-error result')
        if reason in NORMAL|CENSORED:require(not r['errors'],'non-error stop with errors')
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        errors.append(f'malformed result: {exc}')
    return errors
