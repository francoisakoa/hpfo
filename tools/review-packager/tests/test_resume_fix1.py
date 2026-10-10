#!/usr/bin/env python3
"""Regression and resume tests. All experimental records here are synthetic.
The original R1 ZIP is retained solely to reproduce the reported importer failure.
"""
import copy, hashlib, json, os, re, shutil, subprocess, sys, tempfile, unittest, zipfile
from pathlib import Path
KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT/'payload/scripts'))
sys.path.insert(0, str(KIT/'tests'))
from hpfo_common import *
from test_packaging import fixture, store_manifest

R1_HASH='500464402ac3ffaefc2a91a973e60ebfeb55e1f39e0032353cd831a6d91104d8'

def native_record(method='SMC', n=102, run=5):
    # Same field names as the diagnostic supplied by the user. Values are synthetic.
    return dict(synthetic_fixture=True, schema='fixture-native-baselines-not-production',
        identity={'method':method,'n_atoms':n,'run_id':run},identity_sha256='0'*64,
        method=method,n_atoms=n,run_id=run,energy=-2.,search_energy=-2.,
        target_energy=-2.,target_tolerance=1e-4,gap=0.,success=True,
        stop_reason='target_reached',best_x=[0.]*(3*n),n_func_evals=33,max_evals=100,
        budget_exhausted=False,budget_utilisation_pct=33.,
        counters={},first_target_eval=33,first_target_seconds=2.,best_eval=33,
        n_attempts=1,n_restarts=0,attempts=[],n_iterations=1,wall_time=2.,
        audit_seconds=.125,elapsed_including_audit=2.125,audit={},seed=2024,
        base_seed=2024,method_params={},errors=[],local_calls=0)

def native_fixture(base, duplicate=False, missing=False):
    stage,bd,index=fixture(base)
    p=bd/'BASELINES_COMPACT_001.zip';values={}
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            body=z.read(name)
            if name.startswith('records/'):
                old=json.loads(body)
                if missing and (old['method'],old['atoms'],old['run'])==('SMC',104,30):continue
                native=native_record(old['method'],old['atoms'],old['run'])
                # Some JSON producers serialize the index with leading zeros.
                if native['run_id']==6:native['run_id']='06'
                version='baselines_v3_1' if native['method'] in ('BH-OJ','GA') else 'baselines_v3'
                name=(f"results_baselines_nohup_R1/phase_fixture/cohort_000_j22/results/{version}/"
                      f"LJ{native['n_atoms']}/{native['method']}/run{old['run']:02d}.json")
                body=json.dumps(native).encode()
            values[name]=body
    result_hashes={name:hashlib.sha256(body).hexdigest() for name,body in values.items()
                   if '/results/' in name}
    write_json(index,{'synthetic_fixture':True,'records':[
        {'path':name,'sha256':h} for name,h in result_hashes.items()]})
    values['EXPORT/MANIFEST.json']=json.dumps({'format':'synthetic-export',
        'expected_results':1200,'exported_results':len(result_hashes),'files':result_hashes}).encode()
    if duplicate:
        name=next(name for name in values if '/results/' in name)
        values['duplicate/run.json']=values[name]
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,body in values.items():z.writestr(name,body)
    store_manifest(bd)
    return stage,bd,index

def command(kit, stage, bd, index, dest, resume=None):
    args=[sys.executable,'-B',str(kit/'create_hpfo.py'),'--stage',str(stage),
          '--baselines-dir',str(bd),'--baselines-index',str(index),'--destination',str(dest)]
    if resume:args+=['--resume-work',str(resume)]
    return args

def call(args):
    return subprocess.run(args,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
                          capture_output=True,text=True,timeout=90)

class NativeRecordTests(unittest.TestCase):
    def test_native_fields_and_accounting_scope(self):
        r=baseline_record(native_record())
        self.assertIsNotNone(r)
        self.assertEqual((r['method'],r['atoms'],r['run']),('SMC',102,5))
        self.assertEqual((r['seconds'],r['evaluations'],r['objective']),(2.,33,-2.))
        self.assertEqual(r['field_paths'],dict(method='method',atoms='n_atoms',run='run_id',
                         objective='energy',seconds='wall_time',evaluations='n_func_evals'))
        self.assertEqual(r['status'],'target_reached')
        self.assertTrue(r['success'])
    def test_recorded_target_is_not_a_scipy_success_flag(self):
        j=native_record();j['energy']=-1.;j['success']=True
        self.assertFalse(baseline_record(j)['success'])
    def test_numeric_string_identity_is_exact(self):
        j=native_record();j['run_id']='05';j['n_atoms']='102'
        r=baseline_record(j);self.assertEqual((r['atoms'],r['run']),(102,5))
    def test_fractional_identity_not_truncated(self):
        j=native_record();j['run_id']=5.7;self.assertIsNone(baseline_record(j))
    def test_negative_counter_is_rejected(self):
        j=native_record();j['n_func_evals']=-1
        with self.assertRaises(RuntimeError):baseline_record(j)
    def test_generic_success_does_not_create_a_target(self):
        j=native_record();del j['target_energy'];del j['target_tolerance']
        self.assertIsNone(baseline_record(j)['success'])
    def test_missing_run_has_no_path_based_fallback(self):
        j=native_record();del j['run_id'];self.assertIsNone(baseline_record(j))

class ResumeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name)
        cls.stage,cls.bd,cls.index=native_fixture(cls.base)
        archive=KIT/'tests/reference/HPFO_TRX_REVIEW_R1.zip'
        assert sha(archive)==R1_HASH
        extract_zip(archive,cls.base/'old-kit');cls.old=cls.base/'old-kit/HPFO_TRX_REVIEW_R1'
        cls.dest=cls.base/'hpfo'
        cls.before={str(p):sha(p) for tree in (cls.stage,cls.bd) for p in tree.rglob('*') if p.is_file()}
        cls.before[str(cls.index)]=sha(cls.index)
        cls.r1=call(command(cls.old,cls.stage,cls.bd,cls.index,cls.dest))
        match=re.search(r'^PREPARATION_TEMPORAIRE=(.+)$',cls.r1.stdout,re.M)
        assert match,cls.r1.stdout+cls.r1.stderr
        cls.work=Path(match.group(1))
        cls.diagnostic=load(cls.work/'.local/BASELINES_IMPORT_DIAGNOSTIC.json')
        cls.asset_state={p.name:(p.stat().st_ino,p.stat().st_mtime_ns,sha(p))
                         for p in (cls.work/'release-assets').iterdir() if p.is_file()}
        # A changed extracted result must not be silently overwritten during reuse.
        rec=next((cls.work/'.local/baseline-unpack').rglob('run05.json'))
        body=rec.read_bytes();rec.write_bytes(body+b' ')
        cls.bad_resume=call(command(KIT,cls.stage,cls.bd,cls.index,cls.dest,cls.work))
        cls.tamper_preserved=(rec.read_bytes()==body+b' ' and not cls.dest.exists())
        rec.write_bytes(body)
        cls.fixed=call(command(KIT,cls.stage,cls.bd,cls.index,cls.dest,cls.work))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_r1_failure_reproduced_zero_of_1200(self):
        self.assertNotEqual(self.r1.returncode,0,self.r1.stdout)
        self.assertEqual(self.diagnostic['found'],0)
        self.assertEqual(len(self.diagnostic['missing']),1200)
        self.assertEqual(self.diagnostic['invalid_json'],[])
    def test_changed_extracted_file_rejected_without_overwrite(self):
        self.assertNotEqual(self.bad_resume.returncode,0)
        self.assertIn('differs from archive',self.bad_resume.stderr)
        self.assertTrue(self.tamper_preserved)
    def test_full_resume_to_hpfo(self):
        self.assertEqual(self.fixed.returncode,0,self.fixed.stdout+self.fixed.stderr)
        self.assertTrue(self.dest.is_dir());self.assertFalse(self.work.exists())
        self.assertIn('RECOPIE_GROSSES_ARCHIVES=NON',self.fixed.stdout)
        check_package(self.dest,True)
    def test_archives_reused_in_place_not_recopied(self):
        for name,state in self.asset_state.items():
            p=self.dest/'release-assets'/name
            self.assertEqual((p.stat().st_ino,p.stat().st_mtime_ns,sha(p)),state)
    def test_sources_inputs_and_results_unchanged(self):
        for name,h in self.before.items():self.assertEqual(sha(Path(name)),h)
        for h,name in manifest_read(self.stage/'git'/SUBDIR/'SHA256SUMS.txt'):
            if name.startswith('code/'):
                self.assertEqual(sha(self.dest/SUBDIR/name),h)
    def test_all_1200_native_records_with_hashes(self):
        info=load(self.dest/SUBDIR/'provenance/BASELINES_IMPORT.json')
        self.assertEqual((info['results'],info['archive_hash_matches']),(1200,1200))
        rows=load(self.dest/SUBDIR/'provenance/BASELINES_RUN_INDEX.json')
        self.assertEqual({r['field_paths']['evaluations'] for r in rows},{'n_func_evals'})
        self.assertEqual(len(rows),1200)
        self.assertEqual(len({(r['method'],r['atoms'],r['run']) for r in rows}),1200)
    def test_resume_not_published_or_compiled(self):
        self.assertFalse((self.dest/'.git').exists())
        p=load(self.dest/SUBDIR/'provenance/PREPARATION.json')
        self.assertFalse(p['cpp_build_executed']);self.assertFalse(p['optimization_executed'])
        self.assertTrue(p['resumed_preparation'])
    def test_native_reviewer_unpack_rebuild(self):
        p=call([sys.executable,'-B',str(self.dest/'scripts/unpack_data.py')])
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        p=call([sys.executable,'-B',str(self.dest/'scripts/rebuild_all.py'),'--output',str(self.dest/'.local/native-rebuilt')])
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        for t in (7,8,9,10):
            self.assertEqual((self.dest/'.local/native-rebuilt'/f'TABLE_{t}.csv').read_bytes(),
                             (self.dest/SUBDIR/'tables'/f'TABLE_{t}.csv').read_bytes())
    def test_export_checksum_list_does_not_include_itself(self):
        rows=load(self.dest/SUBDIR/'provenance/RELEASE_ASSETS.json')
        self.assertEqual(len(rows),3)
        self.assertNotIn('SHA256SUMS.txt',[r['name'] for r in rows])

class RefusalTests(unittest.TestCase):
    def test_duplicate_native_identity_is_not_selected_arbitrarily(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);stage,bd,index=native_fixture(base,duplicate=True)
            dest=base/'hpfo';p=call(command(KIT,stage,bd,index,dest))
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Duplicate baseline result',p.stderr)
            self.assertFalse(dest.exists())
    def test_missing_native_record_cannot_be_declared_complete(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);stage,bd,index=native_fixture(base,missing=True)
            dest=base/'hpfo';p=call(command(KIT,stage,bd,index,dest))
            self.assertNotEqual(p.returncode,0)
            work=Path(re.search(r'^PREPARATION_TEMPORAIRE=(.+)$',p.stdout,re.M).group(1))
            j=load(work/'.local/BASELINES_IMPORT_DIAGNOSTIC.json')
            self.assertEqual(j['found'],1199);self.assertEqual(j['missing'],[['SMC',104,30]])
            self.assertFalse(dest.exists())

if __name__=='__main__':unittest.main(verbosity=2)
