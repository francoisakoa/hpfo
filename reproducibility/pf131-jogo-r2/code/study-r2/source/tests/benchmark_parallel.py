"""Sequentially benchmark the same frozen work; no concurrent benchmark runs."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--atoms',type=int,default=38)
parser.add_argument('--particles',type=int,default=48)
parser.add_argument('--iterations',type=int,default=8)
parser.add_argument('--top-k',type=int,default=4)
parser.add_argument('--workers',type=int,default=4)
parser.add_argument('--repetitions',type=int,default=3)
args=parser.parse_args()
if args.repetitions<1 or args.workers<2: parser.error('Need repetitions >= 1 and workers >= 2')
args.output_dir.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
frozen=None
reference=None
records=[]
for repeat in range(1,args.repetitions+1):
    for workers in [1,args.workers]:
        folder=args.output_dir/f'repeat{repeat}_workers{workers}'
        command=[str(args.binary.resolve()),'--atoms',str(args.atoms),'--particles',str(args.particles),
                 '--max-iter',str(args.iterations),'--refine-top-k',str(args.top_k),
                 '--local-max-iter','1000','--no-target','--threads',str(workers),
                 '--workers',str(workers),'--output-dir',str(folder),'--quiet']
        if frozen: command+=['--population',str(frozen)]
        subprocess.run(command,check=True,env=env)
        result=json.loads((folder/f'LJ{args.atoms}_run1/pf.json').read_text())
        if reference is None:
            reference=result
            frozen=folder/f'LJ{args.atoms}_run1/initial.pfpop'
        for key in ['x','objective','history','objective_evaluations','gradient_evaluations','hessian_evaluations']:
            if result[key]!=reference[key]: raise AssertionError(f'{key} differs in {folder}')
        records.append({'repeat':repeat,'workers':workers,'seconds':result['seconds'],
                        'local_wall_seconds':result['local_wall_seconds'],
                        'objective':result['objective'],'evaluations':result['objective_evaluations'],
                        'command':command})
serial=statistics.median(r['seconds'] for r in records if r['workers']==1)
parallel=statistics.median(r['seconds'] for r in records if r['workers']==args.workers)
summary={'platform':platform.platform(),'cpu_count':os.cpu_count(),'atoms':args.atoms,
         'particles':args.particles,'iterations':args.iterations,'top_k':args.top_k,
         'workers':args.workers,'repetitions':args.repetitions,'serial_median_seconds':serial,
         'parallel_median_seconds':parallel,'speedup':serial/parallel,'exact_replay':True,'records':records}
(args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
