#!/usr/bin/env python3
"""Fast technical gates, not a performance benchmark. Run on the selected backend."""
import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np
from baselines import METHODS,create_optimizer,_local_minimize,adaptive_beta,normalized_logweights,effective_size_from_logweights
from evaluation import EvaluationCounter
from lj_oracle import LJOracle
from protocol import code_digest,environment
from run_one_baseline import atomic_json


def checks(backend):
    results={};started=time.monotonic()
    for model in ('pf131','standard'):
        oracle=LJOracle(2,model,backend);x=np.array([0.,0.,0.,1.5,0.,0.]);h=1e-6
        fd=np.array([(oracle.value(x+np.eye(6)[j]*h)-oracle.value(x-np.eye(6)[j]*h))/(2*h) for j in range(6)])
        rel=np.linalg.norm(fd-oracle.gradient(x))/max(1,np.linalg.norm(fd))
        assert rel<1e-7,(model,rel)
        ctr=EvaluationCounter(oracle,5000,bounds=5.)
        xx,e,_=_local_minimize(x,2,5.,1000,counter=ctr)
        assert abs(e+1)<1e-9 and abs(oracle.reference(xx)-e)<1e-10,(model,e)
        b=oracle.batch(np.array([x,xx]));assert np.allclose(b,[oracle.value(x),oracle.value(xx)],rtol=1e-12,atol=1e-12)
        # Very close atoms: PF131 and standard remain finite in double at r=1e-4.
        close=x.copy();close[3]=1e-4;assert math.isfinite(oracle.batch(close[None,:])[0])
        results['gradient_local_'+model]=dict(relative_gradient_error=float(rel),local_energy=e,evals=ctr.count)
    x0=np.load(Path(__file__).parent/'tests/data/first_false_stationarity.npz')['x0']
    ctr=EvaluationCounter(LJOracle(38,'pf131',backend),5000,bounds=5.)
    xx,e,_=_local_minimize(x0,38,5.,4000,counter=ctr)
    detail=ctr.local_status_examples[-1]['diagnostics']
    assert e<0 and detail['projected_gradient_inf']<1e-3 and detail['total_displacement_inf']>1e-3
    assert not detail['unresolved_large_residual']
    results['recorded_false_local_convergence']=dict(energy=e,evals=ctr.count,diagnostics=detail)
    for method in METHODS:
        oracle=LJOracle(5,'pf131',backend);ctr=EvaluationCounter(oracle,61,chunk_size=8,bounds=5.)
        opt=create_optimizer(method,5,counter=ctr,seed=2024)
        result=opt.optimize()
        assert ctr.count==61 and ctr.best_x is not None,(method,ctr.count)
        audit=ctr.audit();assert audit['passed'],(method,audit)
        results[method]=dict(energy=result['energy'],evals=ctr.count,audit=audit['passed'])
    smc_ctr=EvaluationCounter(LJOracle(5,'pf131',backend),10000,bounds=5.)
    smc=create_optimizer('SMC',5,counter=smc_ctr,seed=8,n_particles=20,n_iterations=4)
    smc.optimize();assert smc.diagnostics['mcmc_proposals']>=40
    E=np.linspace(-2,1e50,20);beta,rounded=adaptive_beta(0,E,8.,.7)
    ess=effective_size_from_logweights(normalized_logweights(E,beta));assert abs(ess-14)<1e-8
    results['smc']=dict(evals=smc_ctr.count,proposals=smc.diagnostics['mcmc_proposals'],extreme_energy_ess=ess)
    return dict(status='PASS',backend=backend,code_sha256=code_digest(),environment=environment(oracle),
                seconds=time.monotonic()-started,tests=results,
                scope='technical finite-budget tests; no full LJ campaign; GPU tested only if backend=cupy')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=['cpu','cupy'],default='cpu');p.add_argument('--report',default=None);a=p.parse_args()
    try:
        result=checks(a.backend);rc=0
    except Exception as exc:
        result=dict(status='FAIL',backend=a.backend,error=f'{type(exc).__name__}: {exc}',code_sha256=code_digest());rc=2
    if a.report:atomic_json(a.report,result)
    print(json.dumps(result,indent=2,allow_nan=False));return rc
if __name__=='__main__':raise SystemExit(main())
