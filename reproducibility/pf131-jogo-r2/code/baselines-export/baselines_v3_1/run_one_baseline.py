#!/usr/bin/env python3
"""One method/instance/run, shared budget across fully logged attempts.

Resume skips only an identical, audited completed record. Errors/interruption
are not silently retried. --force archives previous records before recomputing.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid
import numpy as np
from baselines import METHODS,VERSION,create_optimizer
from lj_oracle import LJOracle
from local_refinement import POLICY as LOCAL_REFINEMENT_POLICY
from evaluation import EvaluationCounter,RunStop,target_met,NumericalFailure
from protocol import (HERE,KNOWN_MINIMA,COUNTING_CONVENTION,make_seed,attempt_seed,method_params,
                      load_overrides,load_budget,sha_file,code_digest,digest,environment)
from record_schema import SCHEMA,seal,validate_record,INVALID


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.part')
    with temp.open('w') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

@contextlib.contextmanager
def result_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError(f'Job already locked: {path}') from exc
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version',action='version',version=VERSION)
    ap.add_argument('--method',required=True,choices=METHODS)
    ap.add_argument('--atoms',type=int,required=True)
    ap.add_argument('--run',type=int,required=True)
    ap.add_argument('--eval-config',default=str(HERE/'baseline_eval_limits_v3.json'))
    ap.add_argument('--max-evals',type=int)
    ap.add_argument('--base-seed',type=int,default=2024)
    ap.add_argument('--outdir',default='results_v3')
    ap.add_argument('--backend',choices=['cpu','cupy'],default='cpu')
    ap.add_argument('--potential',choices=['pf131','standard'],default='pf131')
    ap.add_argument('--bounds',type=float,default=5.)
    ap.add_argument('--initialization',choices=['box','sphere'],default='box')
    ap.add_argument('--center',action='store_true',help='Center proposals for non-SMC only; changes the protocol')
    ap.add_argument('--target-tol',type=float,default=1e-4)
    ap.add_argument('--restart-seconds',type=float,default=0.,help='0: no arbitrary time restart')
    ap.add_argument('--job-seconds',type=float,default=36000.,help='Cooperative whole-job deadline; 0: disabled')
    ap.add_argument('--max-restarts','--max-attempts',dest='max_attempts',type=int,default=0,
                    help='Maximum attempts INCLUDING initial; 0: unlimited')
    ap.add_argument('--no-restarts',action='store_true')
    ap.add_argument('--chunk-size',type=int,default=32)
    ap.add_argument('--params',help='JSON: {METHOD: {accepted_algorithm_parameter: value}}')
    ap.add_argument('--verbose',action='store_true')
    ap.add_argument('--force',action='store_true',help='Archive previous result rather than overwrite silently')
    return ap


def run(args):
    if args.atoms<2 or args.run<1 or args.max_attempts<0 or args.base_seed<0:
        raise ValueError('Invalid atoms/run/base seed/attempt limit')
    if any(not math.isfinite(v) or v<0 for v in (args.restart_seconds,args.job_seconds,args.target_tol)):
        raise ValueError('Invalid timeout/tolerance')
    budget,cfg=load_budget(args.eval_config,args.atoms,args.max_evals)
    params=method_params(args.method,args.atoms,load_overrides(args.params,args.method))
    oracle=LJOracle(args.atoms,args.potential,args.backend)
    target=KNOWN_MINIMA.get(args.atoms)
    max_attempts=1 if args.no_restarts else args.max_attempts
    identity=dict(version=VERSION,method=args.method,n_atoms=args.atoms,run_id=args.run,
                  base_seed=args.base_seed,budget=budget,budget_config_sha256=sha_file(args.eval_config),
                  target_energy=target,target_tolerance=args.target_tol,potential=args.potential,
                  oracle=oracle.metadata(),backend=args.backend,bounds=args.bounds,
                  initialization=args.initialization,remove_com=args.center,chunk_size=args.chunk_size,
                  method_params=params,restart_seconds=args.restart_seconds,job_seconds=args.job_seconds,
                  max_attempts=max_attempts,code_sha256=code_digest(),environment=environment(oracle),
                  seed_protocol='md5-base-N-run-1000-first8; first attempt uses root directly; later attempts MD5(root,N,a,1000)',
                  restart_protocol='distinct-seed restart on algorithm completion or explicit attempt deadline; shared budget/incumbent (also available to BH-OJ jumps)',
                  counting=COUNTING_CONVENTION)
    if args.method in ('BH-OJ','GA'):
        identity['local_refinement_policy'] = LOCAL_REFINEMENT_POLICY
    fingerprint=digest(identity)
    root=Path(args.outdir).resolve();folder=root/f'LJ{args.atoms}'/args.method
    folder.mkdir(parents=True,exist_ok=True)
    final=folder/f'run{args.run:02d}.json';events=folder/f'run{args.run:02d}.events.jsonl'
    # Validate method parameters and platform before looking at an existing result.
    dry_counter=EvaluationCounter(oracle,budget,bounds=args.bounds)
    create_optimizer(args.method,args.atoms,counter=dry_counter,bounds=args.bounds,
                     initialization=args.initialization,remove_com=args.center,seed=0,**params)
    with result_lock(final.with_suffix('.lock')):
        if final.exists():
            try:old=json.loads(final.read_text());errs=validate_record(old)
            except Exception as exc:old={};errs=[str(exc)]
            if not args.force:
                if errs:raise RuntimeError(f'Existing result invalid: {errs}. Use --force to archive and rerun.')
                if old.get('identity_sha256')!=fingerprint:
                    raise RuntimeError('Existing result has DIFFERENT code/config/budget/environment; new outdir or --force required')
                if old.get('stop_reason') in INVALID:
                    raise RuntimeError('Existing run failed/interrupted. Inspect it; --force archives before retry.')
                print(f'[skip-identical-audited] {final}')
                return 0
            backup=root/'_replaced'/f'{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'/f'LJ{args.atoms}'/args.method
            backup.mkdir(parents=True)
            os.replace(final,backup/final.name)
            if events.exists():os.replace(events,backup/events.name)
        elif events.exists():
            # Preserve evidence from a killed process; no mid-run-state resume is claimed.
            os.replace(events,events.with_name(events.name+'.interrupted-'+uuid.uuid4().hex[:8]))
        with events.open('w',buffering=1) as ef:
            def log(e):
                ef.write(json.dumps(e,allow_nan=False)+'\n')
                if args.verbose and e['event'] in ('attempt_end','incumbent'):print(json.dumps(e),flush=True)
            counter=EvaluationCounter(oracle,budget,bounds=args.bounds,target=target,tolerance=args.target_tol,
                                      job_seconds=args.job_seconds,chunk_size=args.chunk_size,event_callback=log)
            root_seed=make_seed(args.base_seed,args.atoms,args.run,1000)
            attempts=[];errors=[];reason=None;iterations=0
            old_handlers={}
            def interrupt(signum,frame):counter.interrupted=True
            for sig in (signal.SIGINT,signal.SIGTERM):
                old_handlers[sig]=signal.signal(sig,interrupt)
            try:
                while reason is None:
                    try:counter.guard()
                    except RunStop as exc:reason=exc.reason;break
                    if max_attempts and len(attempts)>=max_attempts:
                        reason='single_attempt_complete' if args.no_restarts else 'restart_limit';break
                    ai=len(attempts)+1;seed=attempt_seed(root_seed,args.atoms,ai)
                    counter.begin_attempt(ai,args.restart_seconds)
                    begin=counter.count;t0=time.monotonic()
                    a=dict(attempt=ai,seed=seed,eval_start=begin)
                    log(dict(event='attempt_start',**a))
                    opt=create_optimizer(args.method,args.atoms,counter=counter,bounds=args.bounds,
                                         initialization=args.initialization,remove_com=args.center,
                                         seed=seed,verbose=args.verbose,**params)
                    try:
                        res=opt.optimize(target_energy=target,target_tol=args.target_tol)
                        ar=res['stop_reason']
                    except Exception as exc:
                        ar='numerical_error'
                        errors.append(dict(attempt=ai,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
                    a.update(eval_end=counter.count,elapsed=time.monotonic()-t0,stop_reason=ar,
                             iterations=opt.iteration,diagnostics=opt.diagnostics)
                    attempts.append(a);iterations+=opt.iteration;log(dict(event='attempt_end',**a))
                    counter.attempt_deadline=math.inf
                    if ar in ('target_reached','evaluation_limit','time_limit','interrupted','numerical_error'):
                        reason=ar
                    elif counter.count==begin:
                        reason='numerical_error';errors.append(dict(attempt=ai,type='NoProgress',message='Attempt performed no evaluations'))
            finally:
                for sig,h in old_handlers.items():signal.signal(sig,h)
            search_seconds=time.monotonic()-counter.started
            t0=time.monotonic();audit=counter.audit();audit_seconds=time.monotonic()-t0
            e=audit.get('energy')
            if not audit['passed'] and reason not in ('numerical_error','interrupted'):
                reason='audit_failed';errors.append(dict(type='FinalAudit',message=audit['reason']))
            success=bool(audit['passed'] and e is not None and target_met(e,target,args.target_tol))
            if reason=='target_reached' and not success:
                reason='audit_failed';errors.append(dict(type='FinalTargetMismatch',message='Search target crossing not confirmed by the reporting oracle'))
            if success and audit['passed'] and reason not in ('numerical_error','interrupted'):
                reason='target_reached'
            record=dict(schema=SCHEMA,identity=identity,identity_sha256=fingerprint,
                        method=args.method,n_atoms=args.atoms,run_id=args.run,energy=e,
                        search_energy=counter.best_energy if math.isfinite(counter.best_energy) else None,
                        target_energy=target,target_tolerance=args.target_tol,
                        gap=e-target if e is not None and target is not None else None,
                        success=success,stop_reason=reason,best_x=counter.best_x.tolist() if counter.best_x is not None else None,
                        n_func_evals=counter.count,max_evals=budget,budget_exhausted=counter.count==budget,
                        budget_utilisation_pct=100.*counter.count/budget,
                        counters=dict(objective_global=counter.count_by_origin['global_search'],objective_local=counter.count_by_origin['local'],
                                      gradient_calls=counter.gradient_calls,hessian_calls=0,audit_objective_evaluations=counter.audit_calls,
                                      nonfinite_energy_evaluations=counter.nonfinite_energy_count),
                        first_target_eval=counter.first_target_eval,first_target_seconds=counter.first_target_seconds,
                        best_eval=counter.best_eval,n_attempts=len(attempts),n_restarts=max(0,len(attempts)-1),attempts=attempts,
                        n_iterations=iterations,wall_time=search_seconds,audit_seconds=audit_seconds,
                        elapsed_including_audit=search_seconds+audit_seconds,audit=audit,
                        seed=root_seed,base_seed=args.base_seed,
                        method_params=params,errors=errors,
                        local_calls=counter.local_calls,local_iterations=counter.local_iterations,
                        local_status_counts=dict(counter.local_status_counts),local_status_examples=counter.local_status_examples,
                        local_diagnostics_summary=dict(counter.local_diagnostics_summary),
                        local_anomaly_examples=counter.local_anomaly_examples,
                        hostname=__import__('socket').gethostname(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            seal(record);atomic_json(final,record)
            print(f'[done] {args.method} LJ{args.atoms} run{args.run:02d}: {reason}, E={e}, evals={counter.count}/{budget}, attempts={len(attempts)}',flush=True)
            if reason=='interrupted':return 130
            return 2 if reason in INVALID else 0


def main():
    args=parser().parse_args()
    try:return run(args)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
