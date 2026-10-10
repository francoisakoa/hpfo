"""Frozen defaults and provenance shared by runner, planner and aggregator."""
from __future__ import annotations
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import numpy as np
import scipy
from baselines import DEFAULTS, METHODS, VERSION, KNOWN_MINIMA

HERE=Path(__file__).resolve().parent
ATOMS=[38,75,76,77,98,102,103,104]
CODE_FILES=['local_refinement.py','baselines.py','lj_oracle.py','evaluation.py','protocol.py','record_schema.py',
            'run_one_baseline.py','aggregate_baselines.py','campaign.py','benchmark_suite.py','generate_tables.py',
            'preflight.py','launch_baselines.sh']

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()
def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make_seed(base,n,m,offset=0):
    return int(hashlib.md5(f'{base}_{n}_{m}_{offset}'.encode()).hexdigest()[:8],16)
def attempt_seed(run_seed,n,attempt):
    return int(run_seed) if attempt==1 else make_seed(run_seed,n,attempt,1000)
def code_hashes():return {p:sha_file(HERE/p) for p in CODE_FILES}
def code_digest():return digest(code_hashes())
def load_budget(path,n,explicit=None):
    cfg=json.loads(Path(path).read_text())
    v=explicit if explicit is not None else cfg['eval_limits'][str(n)]['max_evals']
    if isinstance(v,bool) or int(v)!=v or v<=0:raise ValueError('Budget must be a positive integer')
    return int(v),cfg

def method_params(method,n,overrides=None):
    p=DEFAULTS[method].copy()
    if 'n_particles' in p:p['n_particles']=min(300,3*n+56)
    if method=='SMC':p['beta_max']=5.*math.log1p(n)
    if overrides:
        bad=set(overrides)-set(p)
        if bad:raise ValueError(f'{method}: unknown override(s) {sorted(bad)}')
        p.update(overrides)
    return p

def load_overrides(path,method):
    if not path:return {}
    d=json.loads(Path(path).read_text())
    if not isinstance(d,dict) or set(d)-set(METHODS):raise ValueError('Overrides must map method names to parameter dictionaries')
    p=d.get(method,{})
    if not isinstance(p,dict):raise ValueError('Method overrides must be dictionaries')
    return p

def environment(oracle=None):
    keys=['OMP_NUM_THREADS','OMP_DYNAMIC','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','CUDA_VISIBLE_DEVICES','BASELINES_CONCURRENT_JOBS']
    d=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
           system=platform.system(),machine=platform.machine(),threads={k:os.environ.get(k) for k in keys},
           logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)
    try:
        lines=Path('/proc/cpuinfo').read_text().splitlines()
        d['cpu_model']=next((line.split(':',1)[1].strip() for line in lines if line.startswith('model name')),platform.processor())
    except OSError:d['cpu_model']=platform.processor()
    if oracle is not None and oracle.cp is not None:
        d.update(cupy=oracle.cp.__version__,cuda_runtime=int(oracle.cp.cuda.runtime.runtimeGetVersion()),device=oracle.device)
    return d

COUNTING_CONVENTION={
 'energy_budget':'all completed/attempted scalar search energies plus one per candidate in each evaluated batch; local objectives included',
 'derivatives':'gradient calls separate; no gradients or Hessians added to energy budget',
 'batch_boundary':'evaluate only the remaining prefix; retain every evaluated candidate before stopping',
 'batch_target':'finish already dispatched chunk; first_target_eval identifies first hit in row order',
 'audit':'one scalar compensated reference evaluation of final incumbent, reported separately; no new search',
 'precision':'float64 on CPU and optional CuPy batches; CPU analytic local gradients',
}
