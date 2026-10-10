"""Bounded LJ38/LJ76 ordering/replay checks, preserving frozen inputs and commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('binary',type=Path)
parser.add_argument('--reference',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
summary=[]
for atoms in (38,76):
    source=args.reference/f'LJ{atoms}_run1/initial.pfpop'
    lines=source.read_text().splitlines()
    assert lines[0].split()[0]=='PFPOP1'
    pop=args.output/f'LJ{atoms}_first2.pfpop'
    pop.write_text(f'PFPOP1 2 {3*atoms}\n'+'\n'.join(lines[1:3])+'\n')
    for backend in ('ldl','mumps','ipopt'):
        previous=None
        for workers in ((1,) if backend=='ipopt' else (1,2)):
            output=args.output/f'LJ{atoms}_{backend}_w{workers}.json'
            command=[str(args.binary.resolve()),'--atoms',str(atoms),'--population',str(pop.resolve()),'--workers',str(workers),'--local-max-iter','5000','--local-tol','1e-8','--max-time','90','--output',str(output.resolve())]
            command+=['--method','ipopt','--ipopt-linear-solver','mumps'] if backend=='ipopt' else ['--method','lj-newton','--newton-linear-solver',backend,'--newton-linear-ordering','metis']
            output.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
            result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=120)
            output.with_suffix('.log').write_text(result.stdout+result.stderr)
            assert result.returncode==0,(command,result.stderr)
            data=json.loads(output.read_text())
            assert data['status']=='complete',data['status']
            initial=Path(str(output)+'.initial.pfpop')
            assert initial.read_bytes()==pop.read_bytes()
            if backend!='ipopt':
                assert data['linear_metis_analyses']==data['linear_analyses']==2
                assert data['mumps_user_ordering_analyses']==(2 if backend=='mumps' else 0)
                if previous:
                    assert data['points']==previous['points']
                    for key in ('objective_evaluations','linear_factorizations','linear_solves','linear_metis_analyses'):
                        assert data[key]==previous[key],key
            previous=data
            summary.append({key:data[key] for key in ('atoms','method','workers','newton_linear_solver','newton_linear_ordering','core_seconds','audit_pass_count','starts','local_failed','local_limited','linear_metis_analyses','mumps_user_ordering_analyses','linear_cg_fallbacks')})
            summary[-1]['energies']=[p['objective'] for p in data['points']]
            summary[-1]['input_sha256']=hashlib.sha256(pop.read_bytes()).hexdigest()
            print(atoms,backend,workers,data['audit_pass_count'],data['core_seconds'],flush=True)
(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS frozen input bytes, METIS path counters and serial/parallel replay')
