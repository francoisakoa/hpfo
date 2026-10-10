import subprocess, json, os
from pathlib import Path
binary=Path('build_native/pf_local_bench').resolve()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
report=[]
for n in (38,76):
 initial=Path(f'validation/direct_ldl{n}_w1.json.initial.pfpop').resolve()
 for backend,workers in (('ldl',4),('mumps',1),('mumps',4)):
  output=Path(f'validation/direct_{backend}{n}_w{workers}.json')
  command=[str(binary),'--atoms',str(n),'--starts','4','--method','lj-newton','--newton-linear-solver',backend,'--workers',str(workers),'--local-max-iter','5000','--population',str(initial),'--output',str(output)]
  result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=240)
  output.with_suffix('.log').write_text(result.stdout+result.stderr)
  result.check_returncode()
  data=json.loads(output.read_text());assert data['audit_pass_count']==4 and data['linear_factorizations']>0
  if workers==4:
   original=Path(f'validation/direct_{backend}{n}_w1.json.final.pfpop')
   assert original.read_bytes()==Path(str(output)+'.final.pfpop').read_bytes()
  print(result.stdout.strip(),flush=True)
  report.append(dict(atoms=n,backend=backend,workers=workers,audit_pass=data['audit_pass_count'],factorizations=data['linear_factorizations'],fallbacks=data['linear_cg_fallbacks']))
Path('validation/direct_checks.json').write_text(json.dumps({'parallel_replay_exact':True,'runs':report},indent=2)+'\n')
