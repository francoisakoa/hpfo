#!/usr/bin/env python3
"""Short real-C++ functional tests. These are not Table 7--10 benchmark results."""
import argparse,csv,hashlib,json,os,subprocess,tempfile
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('binary',type=Path);ap.add_argument('--original',type=Path);a=ap.parse_args()
env=dict(os.environ,OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
for k in ('LD_PRELOAD','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS'):env.pop(k,None)
checks=[]
with tempfile.TemporaryDirectory(prefix='pf_article_cli_') as tmp:
 root=Path(tmp)
 def run(name,args,expected=0,binary=None):
  dest=root/name;cmd=[str((binary or a.binary).resolve()),*args,'--output-dir',str(dest)]
  p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
  assert p.returncode==expected,(cmd,p.returncode,p.stdout[-2000:])
  return dest
 base=['--atoms','4','--particles','12','--min-particles','5','--runs','1','--run-start','7','--seed','2024',
       '--max-iter','25','--max-time','60','--no-improve-stop','0','--local-refine','none','--refine-top-k','0',
       '--no-final-refine','--no-target','--quiet']
 def result(name,args):
  dest=run(name,args);folder=dest/'LJ4_run7';return json.loads((folder/'pf.json').read_text()),folder
 for alpha in ('.95','.98','.99','1.0'):
  j,_=result('alpha'+alpha,base+['--ipf-alpha-start',alpha,'--ipf-alpha-end','.05'])
  assert j['history'][0]['ipf_alpha']==float(alpha)
  assert abs(j['history'][-1]['ipf_alpha']-.05)<1e-14
 checks.append('four actual alpha endpoints, including alpha=1')
 j,folder=result('ablation',base+['--no-ipf','--no-adaptive-annealing','--no-island-exchange','--no-kicks','--beta-initial','.001'])
 assert j['local_calls']==j['objective_local']==0
 assert all(h['ipf_alpha']==0 and h['beta_used']==h['beta_next']==.001 and not h['kick'] for h in j['history'])
 with (folder/'pf_iterations.csv').open() as f:assert all(int(h['island_exchange'])==0 for h in csv.DictReader(f))
 assert j['options']['seed_pf']==int(hashlib.md5(b'2024_4_7_1000').hexdigest()[:8],16)
 checks+=['no IPF transport; no local calls/evals; no kick/exchange','fixed beta and run-start seed derivation']
 j1,folder=result('first',base)
 j2,_=result('replayed',base+['--population',str(folder/'initial.pfpop')])
 assert (j1['x'],j1['objective'],j1['objective_evaluations'],j1['history'])==(j2['x'],j2['objective'],j2['objective_evaluations'],j2['history'])
 checks.append('byte-frozen population replay')
 run('invalid',base+['--ipf-alpha-start','1.01'],2)
 run('invalid_flow',base+['--no-ipf','--flow','mutation'],2)
 checks.append('invalid configurations rejected')
 # Small native MUMPS/Ipopt exercise, with the same measured comparator enabled.
 small=['--atoms','4','--particles','8','--min-particles','2','--max-iter','2','--max-time','60','--local-max-iter','30','--no-target','--workers','2','--threads','1','--include-ipopt','--ipf-alpha-start','.98','--quiet']
 dest=run('ipopt_smoke',small)
 for name in ('pf','ipopt_multistart'):
  q=json.loads((dest/'LJ4_run1'/f'{name}.json').read_text())
  assert q['local_calls']>0 and q['objective_local']>0
 checks.append('Ipopt/MUMPS process workers and multistart comparator')
 # Native high-dimensional path, very small limits. No global success is claimed.
 for n in (75,102):
  args=['--atoms',str(n),'--particles','12','--min-particles','5','--max-iter','1','--max-time','60',
        '--local-max-iter','2','--refine-top-k','2','--refine-all','--workers','2','--threads','1','--ipf-alpha-start','.98','--quiet']
  dest=run(f'smoke{n}',args);q=json.loads((dest/f'LJ{n}_run1/pf.json').read_text())
  assert q['atoms']==n and len(q['x'])==3*n and q['objective_evaluations']==q['objective_pf']+q['objective_local']
 checks.append('LJ75/LJ102 refine-all native bounded smoke, not benchmark results')
 if a.original:
  oldbase=[x for x in base]
  # Original CLI has no --run-start. Use run1 on both sides.
  ix=oldbase.index('--run-start');del oldbase[ix:ix+2]
  old=run('legacy_original',oldbase,binary=a.original);new=run('legacy_new',oldbase)
  x=json.loads((old/'LJ4_run1/pf.json').read_text());y=json.loads((new/'LJ4_run1/pf.json').read_text())
  for h in y['history']:h.pop('ipf_alpha')
  for key in ('x','objective','objective_evaluations','objective_pf','objective_local','history','status'):
   assert x[key]==y[key],key
  checks.append('legacy-default numerical parity against unmodified PF131 C++')
for check in checks:print('PASS',check)
print('FUNCTIONAL_GROUPS_PASSED=',len(checks))
