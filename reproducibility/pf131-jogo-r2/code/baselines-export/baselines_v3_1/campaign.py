#!/usr/bin/env python3
"""Bounded-concurrency launcher. Child failures propagate; no hidden GPU sharing."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from baselines import METHODS,VERSION,create_optimizer
from evaluation import EvaluationCounter
from lj_oracle import LJOracle
from protocol import HERE,ATOMS,code_digest,digest,load_budget,load_overrides,method_params
from run_one_baseline import atomic_json,result_lock

THREAD_VARS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version=VERSION)
    p.add_argument('--atoms',nargs='+',type=int,default=[38,102,103,104,75,76,77,98])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--runs',type=int,default=30)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--eval-config',default=str(HERE/'baseline_eval_limits_v3.json'))
    p.add_argument('--outdir',default='results_v3')
    p.add_argument('--base-seed',type=int,default=2024)
    p.add_argument('--backend',choices=['cpu','cupy'],default='cpu')
    p.add_argument('--allow-shared-gpu',action='store_true',help='Explicitly allow >1 concurrent job on the same GPU; timings then include contention')
    p.add_argument('--potential',choices=['pf131','standard'],default='pf131')
    p.add_argument('--initialization',choices=['box','sphere'],default='box')
    p.add_argument('--center',action='store_true')
    p.add_argument('--bounds',type=float,default=5.)
    p.add_argument('--target-tol',type=float,default=1e-4)
    p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--job-seconds',type=float,default=36000.)
    p.add_argument('--restart-seconds',type=float,default=0.)
    p.add_argument('--max-attempts',type=int,default=0)
    p.add_argument('--params')
    p.add_argument('--max-evals',type=int,help='Override for explicitly labelled technical pilots, not the reference budgets')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--force',action='store_true')
    return p


def child_command(a,n,m,r):
    command=[sys.executable,str(HERE/'run_one_baseline.py'),'--atoms',str(n),'--method',m,'--run',str(r)]
    for key in ('eval_config','outdir','base_seed','backend','potential','bounds','initialization','target_tol',
                'chunk_size','job_seconds','restart_seconds','max_attempts'):
        command += ['--'+key.replace('_','-'),str(getattr(a,key))]
    if a.params:command+=['--params',a.params]
    if a.max_evals is not None:command+=['--max-evals',str(a.max_evals)]
    if a.center:command+=['--center']
    if a.force:command+=['--force']
    return command


def execute_jobs(jobs,max_parallel,logs,*,env=None):
    """jobs: list of {label, command}. Return child codes, including interruption."""
    logs=Path(logs);logs.mkdir(parents=True,exist_ok=True)
    pending=iter(jobs);active=[];done=[];exhausted=False
    try:
        while not exhausted or active:
            while len(active)<max_parallel and not exhausted:
                try:job=next(pending)
                except StopIteration:exhausted=True;break
                path=logs/(job['label']+'.log');f=path.open('w')
                proc=subprocess.Popen(job['command'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                active.append((proc,f,job,path))
            for entry in list(active):
                proc,f,job,path=entry;rc=proc.poll()
                if rc is not None:
                    f.close();active.remove(entry)
                    done.append(dict(label=job['label'],returncode=rc,log=str(path)))
                    print(f'{"OK" if rc==0 else "FAIL"} {job["label"]} rc={rc}',flush=True)
            if active:time.sleep(.05)
    except KeyboardInterrupt:
        for proc,_,_,_ in active:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        deadline=time.monotonic()+10
        for proc,f,job,path in active:
            try:rc=proc.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
            f.close();done.append(dict(label=job['label'],returncode=rc,log=str(path)))
        return 130,done
    return (1 if any(x['returncode'] for x in done) else 0),done


def launch(a):
    if a.runs<1 or a.jobs<1 or len(set(a.atoms))!=len(a.atoms) or len(set(a.methods))!=len(a.methods):
        raise ValueError('Invalid runs/jobs or duplicate selections')
    if a.backend=='cupy' and a.jobs>1 and not a.allow_shared_gpu:
        raise ValueError('CuPy requires --jobs 1 unless --allow-shared-gpu is explicitly supplied')
    if 'SMC' in a.methods and (a.center or a.initialization!='box'):
        raise ValueError('SMC requires --initialization box without --center')
    a.outdir=str(Path(a.outdir).resolve());a.eval_config=str(Path(a.eval_config).resolve())
    if a.params:a.params=str(Path(a.params).resolve())
    for n in a.atoms:
        if n<2:raise ValueError('Atoms must be >=2')
        budget,_=load_budget(a.eval_config,n,a.max_evals)
        for m in a.methods:
            params=method_params(m,n,load_overrides(a.params,m))
            create_optimizer(m,n,counter=EvaluationCounter(LJOracle(n,a.potential),budget,bounds=a.bounds),
                             bounds=a.bounds,initialization=a.initialization,remove_com=a.center,seed=0,**params)
    root=Path(a.outdir);meta=root/'_campaign';meta.mkdir(parents=True,exist_ok=True)
    entries=[dict(n_atoms=n,method=m,run_id=r,label=f'LJ{n}_{m}_run{r:02d}',command=child_command(a,n,m,r))
             for n in a.atoms for m in a.methods for r in range(1,a.runs+1)]
    plan=dict(version=VERSION,code_sha256=code_digest(),dry_run=a.dry_run,
              parameters=vars(a).copy(),job_count=len(entries),jobs=entries,
              technical_budget_override=a.max_evals is not None)
    atomic_json(meta/('plan_'+digest(plan)+'.json'),plan)
    atomic_json(meta/('plan_dry_run.json' if a.dry_run else 'plan.json'),plan)
    (meta/'joblist.tsv').write_text('method\tatoms\trun\n'+''.join(f'{x["method"]}\t{x["n_atoms"]}\t{x["run_id"]}\n' for x in entries))
    print(f'BASELINES {VERSION}: {len(entries)} jobs; workers={a.jobs}; backend={a.backend}; potential={a.potential}; output={root}',flush=True)
    if a.dry_run:return 0
    env=os.environ.copy()
    for k in THREAD_VARS:env[k]='1'
    env['OMP_DYNAMIC']='FALSE'
    env['BASELINES_CONCURRENT_JOBS']=str(a.jobs)
    # No old environment name is invented; run in the explicitly active interpreter.
    check=subprocess.run([sys.executable,str(HERE/'preflight.py'),'--backend',a.backend,
                          '--report',str(meta/'preflight.json')],env=env)
    if check.returncode:raise RuntimeError('Preflight failed; no scientific job launched')
    with result_lock(meta/'campaign.lock'):
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        logs=root/'_logs'/stamp
        rc,done=execute_jobs(entries,a.jobs,logs,env=env)
        atomic_json(meta/('execution_'+stamp+'.json'),dict(returncode=rc,planned=len(entries),finished=len(done),children=done))
    print(f'CAMPAIGN_RETURN_CODE={rc}; completed={len(done)}/{len(entries)}',flush=True)
    return rc


def main():
    a=parse().parse_args()
    try:return launch(a)
    except KeyboardInterrupt:return 130
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
