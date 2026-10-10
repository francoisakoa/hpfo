import subprocess,json,os,hashlib
from pathlib import Path
binary=Path('build_native/pf_solver').resolve()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
rows=[]
for n in (38,76):
 initial=None
 for method in ('lj-newton','ipopt'):
  output=Path(f'validation/pf_LJ{n}_{method}')
  command=[str(binary),'--atoms',str(n),'--max-iter','20','--no-improve-stop','0','--refine-top-k','5','--workers','4','--threads','4','--max-time','120','--local-max-iter','5000','--local-refine',method,'--seed','2024','--no-target','--quiet','--output-dir',str(output)]
  if initial:command+=['--population',str(initial)]
  output.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
  result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
  output.with_suffix('.log').write_text(result.stdout+result.stderr);result.check_returncode()
  supplied=output/f'LJ{n}_run1/initial.pfpop'
  if initial:assert initial.read_bytes()==supplied.read_bytes()
  else:initial=supplied
  data=json.loads((output/f'LJ{n}_run1/pf.json').read_text())
  assert data['status']=='iteration_limit' and data['iterations']==20
  assert data['objective'] < (-160 if n==38 else -350)
  rows.append({'atoms':n,'method':method,'objective':data['objective'],'seconds':data['seconds'],'particles':data['options']['particles'],'status':data['status'],'initial_sha256':hashlib.sha256(initial.read_bytes()).hexdigest()})
  print(result.stdout.strip(),flush=True)
Path('validation/pf_checks.json').write_text(json.dumps({'scope':'20-iteration integration checks, single run, no performance claim','runs':rows},indent=2)+'\n')
