#!/usr/bin/env python3
"""Synthetic integration only: never executes a scientific solver.
Tests planning, command options, validation, all 42 distinct first-repetition
configurations, aliases, archive, and resume. Temporary fixture data are removed.
"""
import hashlib,json,os,subprocess,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STUB=r'''#!/usr/bin/env python3
import hashlib,importlib.util,json,os,sys
from pathlib import Path
args=sys.argv[1:]
if '--version' in args:print('SYNTHETIC_FIXTURE_ONLY ipopt=1');sys.exit(0)
if '--study-schema' in args:print('PF131_ARTICLE_STUDY_R1');sys.exit(0)
v={};i=0
while i<len(args):
 key=args[i]
 if i+1<len(args) and not args[i+1].startswith('--'):v[key]=args[i+1];i+=2
 else:v[key]=True;i+=1
root=Path(os.environ['PF_TEST_KIT']);spec=importlib.util.spec_from_file_location('s',root/'study.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
n=int(v['--atoms']);r=int(v.get('--run-start',1));b=int(v.get('--seed',2024));count=int(v.get('--particles',3*n+56))
f=Path(v['--output-dir'])/f'LJ{n}_run{r}';f.mkdir(parents=True)
init=dict(seed_pf=s.seed(b,n,r,1000),seed_init=s.seed(b,n,r,0),synthetic_fixture=True)
(f/'initialization.json').write_text(json.dumps(init))
if '--population' in v: (f/'initial.pfpop').write_bytes(Path(v['--population']).read_bytes())
else:(f/'initial.pfpop').write_text(f'PFPOP1 {count} {3*n}\n'+(('0 '* (3*n)).strip()+'\n')*count)
if '--generate-only' in v:sys.exit(0)
p=dict(max_iter=int(v['--max-iter']),max_time=float(v['--max-time']),max_evals=int(v['--max-evals']),threads=int(v['--threads']),workers=int(v['--workers']))
o=s.default_options(n,p)
fields={'particles':'--particles','min_particles':'--min-particles','max_iterations':'--max-iter','no_improve_stop':'--no-improve-stop','max_seconds':'--max-time','max_objective_evaluations':'--max-evals','radius':'--radius','refine_top_k':'--refine-top-k','local_method':'--local-refine','local_max_iterations':'--local-max-iter','local_tolerance':'--local-tol','batch_threads':'--threads','local_workers':'--workers','ipopt_linear_solver':'--ipopt-linear-solver','flow':'--flow','mapping':'--mapping','clustering':'--clustering','ipf_alpha_initial':'--ipf-alpha-start','ipf_alpha_final':'--ipf-alpha-end','beta_initial':'--beta-initial','beta_max':'--beta-max','mutation_sigma':'--mutation-sigma','lambda_share':'--lambda-share','lambda_novelty':'--lambda-novelty','target':'--target','target_tolerance':'--target-tol'}
for key,flag in fields.items():
 value=v[flag];o[key]=type(o[key])(value)
o['refine_all']='--refine-all' in v
for key,flag in {'ipf_enabled':'--no-ipf','adaptive_annealing':'--no-adaptive-annealing','island_exchange':'--no-island-exchange','islands':'--no-islands','kicks':'--no-kicks','diversity_injection':'--no-diversity-injection','ess_rejuvenation':'--no-ess-rejuvenation','final_refinement':'--no-final-refine'}.items():o[key]=flag not in v
o['seed_pf']=init['seed_pf'];o['seed_init']=init['seed_init']
calls=(count if o['refine_all'] else o['refine_top_k']) if o['local_method']!='none' else 0
j=dict(synthetic_fixture=True,atoms=n,run=r,status='iteration_limit',options=o,objective=0.0,seconds=.01,x=[0.0]*(3*n),objective_evaluations=1+calls,objective_pf=1,objective_local=calls,local_calls=calls,target_reached=False,iterations=1,history=[dict(weights=[1.0],objectives=[0.0],beta_used=o['beta_initial'],beta_next=o['beta_initial'],ipf_alpha=o['ipf_alpha_initial'] if o['ipf_enabled'] else 0.0,kick=False)])
(f/'pf.json').write_text(json.dumps(j));(f/'pf.xyz').write_text(str(n)+'\nSYNTHETIC ONLY\n'+'X 0 0 0\n'*n)
(f/'pf_iterations.csv').write_text('iteration,island_exchange\n1,0\n')
'''
with tempfile.TemporaryDirectory(prefix='pf_refine_all_e2e_') as td:
 base=Path(td);binpath=base/'fakebin';binpath.mkdir()
 solver=binpath/'pf_solver';solver.write_text(STUB);solver.chmod(0o755)
 (binpath/'pf_local_worker').write_text('synthetic worker never executed\n')
 out=base/'output';env=dict(os.environ,PF_TEST_KIT=str(ROOT))
 options=['--out',str(out),'--binary',str(solver),'--ablation-runs','1','--sensitivity-runs','1','--max-iter','1','--max-time','10','--threads','1','--workers','1']
 def call(mode):
  p=subprocess.run([sys.executable,str(ROOT/'study.py'),mode,*options],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
  if p.returncode:raise RuntimeError(p.stdout[-6000:])
  return p.stdout
 call('plan');text=call('run')
 plan=json.loads((out/'PLAN.json').read_text());status=json.loads((out/'reports/STATUS.json').read_text())
 assert plan['unique_solves']==42 and plan['logical_rows']==46
 assert status['unique_completed']==42 and status['logical_rows_completed']==46
 counts={'all':0,'none':0}
 for task in plan['tasks']:
  mark=json.loads((out/'runs'/task['id']/'DONE.json').read_text());raw=out/mark['folder'];j=json.loads((raw/'pf.json').read_text())
  assert j['atoms'] in (75,102) and j['synthetic_fixture']
  command=json.loads((raw.parent.parent/'EXECUTION.json').read_text())['command']
  if task['configuration'] in ('no_local','pf_only'):
   assert '--refine-all' not in command and j['local_calls']==0;counts['none']+=1
  else:
   assert '--refine-all' in command and j['options']['refine_all'] is True;counts['all']+=1
 assert counts=={'all':38,'none':4}
 files={str(p):p.stat().st_mtime_ns for p in (out/'runs').rglob('*') if p.is_file()}
 text=call('run')
 assert files=={str(p):p.stat().st_mtime_ns for p in (out/'runs').rglob('*') if p.is_file()}
 archives=list((out/'bundles').glob('*.tar.gz'));assert len(archives)==2
 for archive in archives:
  assert 'LJ75_LJ102_REFINE_ALL' in archive.name
  expected=archive.with_name(archive.name+'.sha256').read_text().split()[0]
  assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
  with tarfile.open(archive) as f:
   m=json.load(f.extractfile(next(t for t in f if t.name.endswith('/ARCHIVE_MANIFEST.json'))))
   assert m['complete'] and m['status']['unique_completed']==42
 print('SYNTHETIC_E2E=PASS; configurations=42; table_rows=46; all=38; none=4; resume_no_recompute=PASS; archives=2 verified')
 print('No real objective evaluation, no C++ solver, no scientific measurement.')
