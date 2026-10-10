"""Paired local benchmark; commands, starts, every endpoint and audits are retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

p=argparse.ArgumentParser()
p.add_argument('binary',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--atoms',type=int,nargs='+',default=[38,76])
p.add_argument('--starts',type=int,default=8)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--workers',type=int,default=4)
p.add_argument('--tol',type=float,default=1e-8)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
binary=str(a.binary.resolve())
records=[]
for n in a.atoms:
    initial=None
    endpoints={}
    for rep in range(a.repeats):
        # Alternate order to reduce a systematic order advantage.
        configurations=[('lj-newton',1),('ipopt',1),('lj-newton',a.workers),('ipopt',a.workers)]
        if rep%2:configurations.reverse()
        for method,workers in configurations:
            output=a.output/f'LJ{n}_{method}_w{workers}_r{rep+1}.json'
            command=[binary,'--atoms',str(n),'--starts',str(a.starts),'--method',method,'--workers',str(workers),
                     '--local-tol',str(a.tol),'--local-max-iter','5000','--output',str(output.resolve())]
            if initial:command+=['--population',str(initial.resolve())]
            output.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
            result=subprocess.run(command,text=True,capture_output=True,env=env,timeout=600)
            output.with_suffix('.log').write_text(result.stdout+result.stderr)
            result.check_returncode()
            if initial is None:initial=Path(str(output)+'.initial.pfpop')
            assert initial.read_bytes()==Path(str(output)+'.initial.pfpop').read_bytes()
            data=json.loads(output.read_text())
            assert data['status']=='complete'
            assert all(point['feasible'] for point in data['points'])
            endpoint=Path(str(output)+'.final.pfpop').read_bytes()
            if method in endpoints:assert endpoint==endpoints[method],(n,method,'parallel/repeat endpoint mismatch')
            else:endpoints[method]=endpoint
            data['repetition']=rep+1
            data['initial_sha256']=hashlib.sha256(initial.read_bytes()).hexdigest()
            records.append(data)
            print(result.stdout.strip(),flush=True)
summary=[]
for n in a.atoms:
    for method in ('lj-newton','ipopt'):
        for workers in (1,a.workers):
            rows=[r for r in records if r['atoms']==n and r['method']==method and r['workers']==workers]
            points=rows[0]['points']
            summary.append({'atoms':n,'method':method,'workers':workers,
                'median_seconds':statistics.median(r['core_seconds'] for r in rows),
                'seconds':[r['core_seconds'] for r in rows],
                'audit_pass_count':rows[0]['audit_pass_count'],'starts':a.starts,
                'objective_min':min(x['objective'] for x in points),
                'objective_median':statistics.median(x['objective'] for x in points),
                'max_projected_stationarity':max(x['projected_stationarity_inf'] for x in points),
                'local_failed':rows[0]['local_failed'],'local_limited':rows[0]['local_limited'],
                'initial_sha256':rows[0]['initial_sha256']})
report={'summary':summary,'local_tolerance':a.tol,'repeats':a.repeats,'parallel_endpoints_exact':True,
        'comparison_scope':'Same supplied starts and constraints. Different local basins and stopping tests; independent unscaled projected-stationarity audit retained.',
        'full_global_optimization':False}
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
