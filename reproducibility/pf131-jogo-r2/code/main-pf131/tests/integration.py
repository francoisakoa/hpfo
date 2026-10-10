"""CLI / persistence / replay checks. Only Python standard library is required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary=Path(sys.argv[1]).resolve()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='pf_integration_') as directory:
    root=Path(directory)
    def run(name,args,expected=0):
        command=[str(binary),'--output-dir',str(root/name),*args]
        p=subprocess.run(command,text=True,capture_output=True,env=env,timeout=90)
        assert p.returncode==expected,(command,p.stdout,p.stderr)
        return root/name
    args=['--atoms','4','--particles','12','--min-particles','2','--max-iter','3','--refine-top-k','3','--local-max-iter','500','--no-target','--quiet']
    version=subprocess.check_output([str(binary),'--version'],text=True)
    if 'ipopt=1' not in version: args+=['--local-refine','projected-lbfgs']
    a=run('serial',args)
    pop=a/'LJ4_run1/initial.pfpop'
    b=run('parallel',[*args,'--population',str(pop),'--threads','1' if 'openmp=0' in version else '2','--workers','2'])
    j1=json.loads((a/'LJ4_run1/pf.json').read_text())
    j2=json.loads((b/'LJ4_run1/pf.json').read_text())
    assert j1['x']==j2['x'] and j1['history']==j2['history']
    assert j1['objective_evaluations']==j2['objective_evaluations']
    assert j1['iterations']==len(j1['history'])==3
    assert j1['objective_evaluations']==j1['objective_pf']+j1['objective_local']
    assert len((a/'LJ4_run1/pf.xyz').read_text().splitlines())==6
    if 'ipopt=1' in version: assert abs(j1['objective']+6)<1e-6
    c=run('budget',[*args,'--max-evals','80','--workers','2'])
    limited=json.loads((c/'LJ4_run1/pf.json').read_text())
    assert limited['objective_evaluations']<=80
    assert limited['status']=='evaluation_limit'
    native_args=[*args,'--local-refine','lj-newton']
    native=run('newton_serial',native_args)
    native_result=json.loads((native/'LJ4_run1/pf.json').read_text())
    assert native_result['options']['local_method']=='lj-newton'
    assert native_result['options']['local_execution']=='threads'
    assert native_result['hessian_products']>0 and abs(native_result['objective']+6)<1e-6
    if 'openmp=1' in version:
        parallel=run('newton_threads',[*native_args,'--workers','3','--worker-executable','/no/process/should/be/spawned','--population',str(native/'LJ4_run1/initial.pfpop')])
        jr=json.loads((parallel/'LJ4_run1/pf.json').read_text())
        assert jr['x']==native_result['x'] and jr['history']==native_result['history']
        assert jr['objective_evaluations']==native_result['objective_evaluations']
        low=run('newton_budget',[*native_args,'--workers','3','--max-evals','80'])
        jl=json.loads((low/'LJ4_run1/pf.json').read_text())
        assert jl['objective_evaluations']<=80 and jl['status']=='evaluation_limit'
    direct_backends=['ldl']+(['mumps'] if 'native_mumps=1' in version else [])
    combinations=[(b,'auto') for b in direct_backends]
    if 'metis=1' in version: combinations += [(b,'metis') for b in direct_backends]
    for backend,ordering in combinations:
        tag=backend+'_'+ordering
        da=[*native_args,'--newton-linear-solver',backend,'--newton-linear-ordering',ordering]
        first=run('direct_'+tag,da)
        r1=json.loads((first/'LJ4_run1/pf.json').read_text())
        assert r1['linear_factorizations']>0 and r1['linear_solves']>0
        assert r1['build']['arithmetic']=='IEEE754-binary64'
        assert r1['options']['newton_linear_ordering']==ordering
        if ordering=='metis':
            assert r1['linear_metis_analyses']==r1['linear_analyses']>0
            assert r1['mumps_user_ordering_analyses']==(r1['linear_analyses'] if backend=='mumps' else 0)
        assert abs(r1['objective']+6)<1e-6
        if backend=='mumps' or 'openmp=1' in version:
            second=run('direct_parallel_'+tag,[*da,'--workers','3','--population',str(first/'LJ4_run1/initial.pfpop')])
            r2=json.loads((second/'LJ4_run1/pf.json').read_text())
            assert r1['x']==r2['x'] and r1['history']==r2['history'],backend
            assert r1['linear_factorizations']==r2['linear_factorizations'],backend
            assert r1['linear_metis_analyses']==r2['linear_metis_analyses']
            assert r1['mumps_user_ordering_analyses']==r2['mumps_user_ordering_analyses']
            low=run('direct_budget_'+tag,[*da,'--workers','3','--max-evals','80'])
            limited=json.loads((low/'LJ4_run1/pf.json').read_text())
            assert limited['status']=='evaluation_limit' and limited['objective_evaluations']<=80
    run('bad_ordering',[*native_args,'--newton-linear-ordering','amd'],2)
    if 'metis=1' in version and 'ipopt=1' in version:
        combined=run('metis_with_ipopt',[*native_args,'--newton-linear-solver','ldl','--newton-linear-ordering','metis','--include-ipopt','--workers','2'])
        baseline=json.loads((combined/'LJ4_run1/ipopt_multistart.json').read_text())
        assert baseline['options']['local_method']=='ipopt'
        assert baseline['linear_metis_analyses']==0
        assert abs(baseline['objective']+6)<1e-6
    run('bad_cg_metis',[*native_args,'--newton-linear-ordering','metis'],2)
    if 'metis=0' in version:
        run('missing_metis',[*native_args,'--newton-linear-solver','ldl','--newton-linear-ordering','metis'],2)
    run('missing_hsl',[*native_args,'--newton-linear-solver','ma27','--hsl-library','/missing/libhsl.so'],2)
    run('bad_ipopt_ldl',[*args,'--ipopt-linear-solver','ldl'],2)
    run('bad_number',['--atoms','4','--particles','-1'],2)
    run('bad_flag',['--unknown'],2)
    run('bad_affine',['--atoms','3','--mapping','affine'],2)
    run('bad_population',[*args,'--population',str(root/'absent.pfpop')],2)
    # Reusing an output folder must fail instead of silently destroying results.
    run('serial',args,2)
print('PASS CLI schema, frozen replay, parallel replay, budgets, XYZ and invalid-input checks')
