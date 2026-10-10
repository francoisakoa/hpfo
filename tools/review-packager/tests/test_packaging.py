#!/usr/bin/env python3
"""Synthetic packaging tests. Fixtures are NOT scientific results."""
import copy,hashlib,importlib.util,io,json,os,runpy,shutil,stat,subprocess,sys,tarfile,tempfile,unittest,zipfile
from pathlib import Path
KIT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(KIT/'payload/scripts'))
from hpfo_common import *
import rebuild_all,replay

# Verbatim-equivalent reference table algorithm of the already provided preparer.
TABLE_TOOL='''
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ABL=("full","no_ipf","no_sharing","no_novelty","no_kicks","no_island_exchange","no_local","no_adaptive_annealing","pf_local_only","pf_only")
GRIDS={7:ABL,8:("np_100","np_200","np_300","np_400","np_500"),9:("beta_1e-5","beta_1e-4","beta_1e-3","beta_1e-2"),10:("alpha_0.95","alpha_0.98","alpha_0.99","alpha_1.00")}
ALIASES={"beta_1e-4":"full","alpha_0.98":"full"}
def main(data,output):
 records=json.loads((ROOT/"provenance/RUN_INDEX.json").read_text());observed={}
 for rec in records:
  raw=(data/rec["file"]).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=rec["sha256"]:raise RuntimeError("Hash mismatch")
  j=json.loads(raw)
  if j["objective_evaluations"]!=j["objective_pf"]+j["objective_local"]:raise RuntimeError("Counter mismatch")
  if rec["dataset"]!="study":continue
  o=j["options"];s=j["objective"]<=o["target"]+o["target_tolerance"]
  if s!=j["target_reached"]:raise RuntimeError("Success mismatch")
  key=(rec["configuration"],rec["atoms"],rec["run"])
  if key in observed:raise RuntimeError("Duplicate")
  observed[key]={"success":int(s),"seconds":j["seconds"],"evaluations":j["objective_evaluations"],"objective":j["objective"],"iterations":j["iterations"]}
 if len(observed)!=820:raise RuntimeError("Expected 820")
 output.mkdir(parents=True,exist_ok=False)
 for table,configs in GRIDS.items():
  rows=[];repetitions=30 if table==7 else 10
  for cfg in configs:
   actual=ALIASES.get(cfg,cfg)
   for n in (75,102):
    group=[observed[(actual,n,r)] for r in range(1,repetitions+1)]
    row={"configuration":cfg,"atoms":n,"runs":repetitions,"successes":sum(x["success"] for x in group)}
    row["success_percent"]=100*row["successes"]/repetitions
    for field in ("seconds","evaluations","objective","iterations"):
     values=[x[field] for x in group]
     row["mean_"+field]=statistics.mean(values);row["median_"+field]=statistics.median(values);row["sample_sd_"+field]=statistics.stdev(values)
    rows.append(row)
  with (output/f"TABLE_{table}.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  if table==7:(output/"TABLE_7.md").write_text("Synthetic fixture, not production.\\n")
'''
ABL=('full','no_ipf','no_sharing','no_novelty','no_kicks','no_island_exchange','no_local','no_adaptive_annealing','pf_local_only','pf_only')
EXTRA=tuple('np_'+str(n) for n in (100,200,300,400,500))+('beta_1e-5','beta_1e-3','beta_1e-2','alpha_0.95','alpha_0.99','alpha_1.00')

def store_manifest(root):
 p=root/'SHA256SUMS.txt'
 p.write_text(''.join(sha(x)+'  '+x.relative_to(root).as_posix()+'\n' for x in sorted(root.rglob('*')) if x.is_file() and x!=p and '__pycache__' not in x.parts))

def fixture(base):
 stage=base/'stage';package=stage/'git'/SUBDIR;package.mkdir(parents=True)
 data=stage/'raw/data';idx=[]
 for ds in ('main','ipopt_multistart','study'):
  tasks=[(c,n,r) for c in ABL+EXTRA for n in (75,102) for r in range(1,(30 if c in ABL else 10)+1)] if ds=='study' else [('full',n,r) for n in ATOMS for r in range(1,31)]
  for cfg,n,r in tasks:
   folder=Path('study')/'runs'/cfg/f'LJ{n}'/f'run_{r:03d}'/'attempt_001/raw'/f'LJ{n}_run{r}' if ds=='study' else Path('main')/f'LJ{n}/mumps/LJ{n}_run{r}'
   filename='ipopt_multistart.json' if ds=='ipopt_multistart' else 'pf.json';rel=folder/filename
   j={'synthetic_fixture':True,'atoms':n,'run':r,'objective':-2.,'seconds':float(r),'iterations':1,'target_reached':True,'objective_evaluations':100,'objective_pf':20,'objective_local':80,'options':{'target':-2.,'target_tolerance':.0001}}
   write_json(data/rel,j);idx.append({'dataset':ds,'configuration':cfg,'atoms':n,'run':r,'file':rel.as_posix(),'sha256':sha(data/rel)})
 write_json(package/'provenance/RUN_INDEX.json',idx)
 write_json(package/'provenance/PREPARATION.json',{'indexed_results':{'main':240,'ipopt_multistart':240,'study':820}})
 write_json(package/'provenance/BASELINES_IMPORT.json',{'state':'NOT_IMPORTED'})
 (package/'tools').mkdir();(package/'tools/rebuild_tables.py').write_text(TABLE_TOOL)
 ns=runpy.run_path(str(package/'tools/rebuild_tables.py'));ns['main'](data,package/'tables')
 (package/'CITATION.cff').write_text('cff-version: 1.2.0\nmessage: Synthetic fixture only\n')
 for tree in ('main-pf131','study-r2/source'):
  p=package/'code'/tree/'src';p.mkdir(parents=True);(p/'solver.cpp').write_text('// synthetic fixture, not a solver\n')
  (p.parent/'CMakeLists.txt').write_text('# fixture\n')
 # Source-local data files must be included by the final manifest.
 p=package/'code/main-pf131/tests/data/fixture.txt';p.parent.mkdir(parents=True);p.write_text('fixture\n')
 (stage/'release-assets').mkdir()
 for name in ('main','study'):
  with tarfile.open(stage/'release-assets'/f'hpfo-pf131-{name}-data.tar.gz','w:gz') as t:
   for p in (data/name).rglob('*'):
    if p.is_file():t.add(p,arcname='data/'+p.relative_to(data).as_posix())
 store_manifest(stage/'release-assets');store_manifest(package)
 bd=base/'baseline';bd.mkdir();archive=bd/'BASELINES_COMPACT_001.zip';hashes=[]
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for m in METHODS:
   for n in ATOMS:
    for r in range(1,31):
     j={'synthetic_fixture':True,'method':m,'atoms':n,'run':r,'objective':-2.,'seconds':2.,'objective_evaluations':33,'target_reached':True,'target':-2.,'target_tol':.0001,'status':'target_reached'}
     raw=json.dumps(j).encode();z.writestr(f'records/{m}/LJ{n}/run_{r:03d}.json',raw);hashes.append(hashlib.sha256(raw).hexdigest())
  z.writestr('source/baselines_v3/run_one_baseline.py','# synthetic fixture only\n')
  z.writestr('source/baselines_v3/baseline_eval_limits_v3.json','{"synthetic_fixture":true}')
  source=io.BytesIO()
  with tarfile.open(fileobj=source,mode='w:gz') as t:
   for name,body in [('baselines_v3_1/run_one_baseline.py',b'# synthetic source 2\n'),('baselines_v3_1/baseline_eval_limits_v3.json',b'{"synthetic_fixture":true}')]:
    i=tarfile.TarInfo(name);i.size=len(body);t.addfile(i,io.BytesIO(body))
  z.writestr('source/frozen_sources.tar.gz',source.getvalue())
 index=base/'results_index.json';write_json(index,{'synthetic_fixture':True,'hashes':hashes});store_manifest(bd)
 return stage,bd,index

class PackagingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name);cls.stage,cls.bd,cls.index=fixture(cls.base)
  cls.dest=cls.base/'hpfo'
  cls.args=[sys.executable,str(KIT/'create_hpfo.py'),'--stage',str(cls.stage),'--baselines-dir',str(cls.bd),'--baselines-index',str(cls.index)]
  cls.before={p.relative_to(cls.base).as_posix():sha(p) for base in (cls.stage,cls.bd) for p in base.rglob('*') if p.is_file()}
  cls.result=subprocess.run(cls.args+['--destination',str(cls.dest)],text=True,capture_output=True,timeout=90)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_01_full_install(self):
  self.assertEqual(self.result.returncode,0,self.result.stdout+'\n'+self.result.stderr);self.assertTrue(self.dest.is_dir());check_package(self.dest,True)
 def test_02_originals_unchanged(self):
  for rel,h in self.before.items():self.assertEqual(sha(self.base/rel),h)
 def test_03_repeated_install_refused(self):
  p=subprocess.run(self.args+['--destination',str(self.dest)],capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('NOT be overwritten',p.stderr)
 def test_04_baseline_grid_and_hashes(self):
  j=load(self.dest/SUBDIR/'provenance/BASELINES_IMPORT.json');self.assertEqual(j['results'],1200);self.assertEqual(j['archive_hash_matches'],1200)
 def test_05_nested_sources_preserved(self):
  j=load(self.dest/SUBDIR/'provenance/BASELINES_IMPORT.json');self.assertEqual(len(j['frozen_drivers']),2)
 def test_06_source_data_not_ignored(self):
  names={x['path'] for x in load(self.dest/'PACKAGE_MANIFEST.json')['files']};self.assertIn((SUBDIR/'code/main-pf131/tests/data/fixture.txt').as_posix(),names)
 def test_07_release_bytes_preserved(self):
  self.assertEqual(sha(self.bd/'BASELINES_COMPACT_001.zip'),sha(self.dest/'release-assets/BASELINES_COMPACT_001.zip'))
 def test_08_unpack_and_rebuild(self):
  p=subprocess.run([sys.executable,str(self.dest/'scripts/unpack_data.py')],capture_output=True,text=True,timeout=60);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  p=subprocess.run([sys.executable,str(self.dest/'scripts/rebuild_all.py'),'--output',str(self.dest/'.local/test-rebuild')],capture_output=True,text=True,timeout=60);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
  for t in (7,8,9,10):self.assertEqual((self.dest/'.local/test-rebuild'/f'TABLE_{t}.csv').read_bytes(),(self.dest/SUBDIR/'tables'/f'TABLE_{t}.csv').read_bytes())
 def test_09_second_unpack_refused(self):
  if not (self.dest/'data').exists():self.skipTest('unpack not yet run')
  p=subprocess.run([sys.executable,str(self.dest/'scripts/unpack_data.py')],capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('already exists',p.stderr)
 def test_10_publish_is_dry_run(self):
  p=subprocess.run([sys.executable,str(self.dest/'scripts/publish_github.py')],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertIn('DRY_RUN=YES',p.stdout);self.assertFalse((self.dest/'.git').exists())
 def test_11_publish_requires_consent(self):
  p=subprocess.run([sys.executable,str(self.dest/'scripts/publish_github.py'),'--apply'],capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('confirm-rights-and-privacy',p.stderr)
 def test_12_publish_requires_actual_build_report(self):
  p=subprocess.run([sys.executable,str(self.dest/'scripts/verify_package.py'),'--for-publish'],capture_output=True,text=True);self.assertNotEqual(p.returncode,0)
 def test_13_zip_traversal_refused(self):
  p=self.base/'bad.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','no')
  with self.assertRaises(RuntimeError):extract_zip(p,self.base/'bad-unpack')
  self.assertFalse((self.base/'escape').exists())
 def test_14_zip_symlink_refused(self):
  p=self.base/'link.zip';i=zipfile.ZipInfo('link');i.create_system=3;i.external_attr=(stat.S_IFLNK|0o777)<<16
  with zipfile.ZipFile(p,'w') as z:z.writestr(i,'/etc/passwd')
  with self.assertRaises(RuntimeError):extract_zip(p,self.base/'link-unpack')
 def test_15_hash_tamper_detected(self):
  p=self.dest/'README.md';old=p.read_bytes()
  try:
   p.write_bytes(old+b'changed');self.assertRaises(RuntimeError,check_package,self.dest,False)
  finally:p.write_bytes(old)
 def test_16_target_not_guessed_from_scipy_success(self):
  j={'method':'GA','N':38,'run':1,'best_f':-2.,'elapsed_s':2.,'nfev':10,'success':True}
  r=baseline_record(j);self.assertIsNone(r['success']);self.assertEqual(r['atoms'],38)
 def test_17_replay_no_shell_operator(self):
  with self.assertRaises(RuntimeError):replay.find_command({'cmd':'echo no command'})
 def test_18_baseline_record_aliases(self):
  j={'method':'BH_OJ','N':38,'run':1,'best_energy':-3.,'runtime_seconds':2.,'evals_used':10,'target_energy':-3.,'target_tolerance':.0001,'target_reached':True}
  r=baseline_record(j);self.assertEqual(r['method'],'BH-OJ');self.assertTrue(r['success'])

if __name__=='__main__':unittest.main(verbosity=2)
