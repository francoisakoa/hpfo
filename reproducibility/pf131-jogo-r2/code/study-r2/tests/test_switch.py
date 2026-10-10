import importlib.util,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
stop=module('stop_campaign');reuse=module('reuse_binaries')
class SwitchTests(unittest.TestCase):
 def test_source_identity_and_tamper(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);kit=d/'kit';old=d/'old';src=kit/'source';src.mkdir(parents=True)
   (src/'file.cpp').write_text('int main() { return 0; }\n')
   frozen=old/'provenance/kit/source';frozen.mkdir(parents=True)
   (frozen/'file.cpp').write_bytes((src/'file.cpp').read_bytes())
   (old/'provenance/SOURCE_SHA256.json').write_text(json.dumps({'source/file.cpp':reuse.sha(src/'file.cpp')}))
   self.assertEqual(len(reuse.verify_sources(kit,old)),1)
   (src/'file.cpp').write_text('changed')
   with self.assertRaises(RuntimeError):reuse.verify_sources(kit,old)
 def test_binary_role_exact_path(self):
  out=Path('/tmp/pf_unit_role')
  self.assertEqual(stop.role(dict(pid=os.getpid(),args=[str(out/'provenance/bin/pf_solver'),'--atoms','75']),out),'pf_solver')
  self.assertIsNone(stop.role(dict(pid=os.getpid(),args=['/tmp/other/provenance/bin/pf_solver','--log',str(out)]),out))
 def test_runner_role_exact_out(self):
  out=Path('/tmp/pf_unit_role')
  info=dict(pid=os.getpid(),args=[sys.executable,'/tmp/some/study.py','run','--out',str(out)])
  self.assertEqual(stop.role(info,out),'runner')
  info['args'][-1]='/tmp/pf_unit_role_other';self.assertIsNone(stop.role(info,out))
  info['args'][-1]=str(out);info['args'][2]='plan';self.assertIsNone(stop.role(info,out))
 def test_scoped_sigterm_children_and_unrelated_process(self):
  with tempfile.TemporaryDirectory(prefix='pf_stop_test_') as td:
   root=Path(td);out=root/'target';other=root/'unrelated'
   for o in (out,other):
    (o/'provenance/bin').mkdir(parents=True)
    (o/'PLAN.json').write_text('{}\n');(o/'preserved.txt').write_text('must not change\n')
   script=root/'study.py'
   script.write_text('''import os,signal,subprocess,sys,time,json
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--out')+1])
child=subprocess.Popen(['/bin/sleep','60'],start_new_session=True)
def done(*args):
 try: os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError: pass
 child.wait(timeout=10)
 sys.exit(0)
signal.signal(signal.SIGTERM,done)
(out/'child.json').write_text(json.dumps({'pid':child.pid}))
while True:time.sleep(.05)
''')
   procs=[subprocess.Popen([sys.executable,str(script),'run','--out',str(o)]) for o in (out,other)]
   try:
    deadline=time.monotonic()+5
    while not all((o/'child.json').exists() for o in (out,other)) and time.monotonic()<deadline:time.sleep(.02)
    self.assertTrue((out/'child.json').exists());child=json.loads((out/'child.json').read_text())['pid']
    self.assertEqual(stop.main(['--out',str(out),'--check']),2)
    self.assertEqual(stop.main(['--out',str(out),'--stop','--timeout','5']),0)
    self.assertEqual(procs[0].wait(timeout=3),0);self.assertIsNone(procs[1].poll())
    self.assertIsNone(stop.proc_info(child));self.assertEqual((out/'preserved.txt').read_text(),'must not change\n')
    self.assertEqual(stop.main(['--out',str(out),'--check']),0)
   finally:
    for proc in procs:
     if proc.poll() is None:proc.terminate()
    for proc in procs:
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
 def test_absent_campaign_no_signal(self):
  with tempfile.TemporaryDirectory() as td:self.assertEqual(stop.main(['--out',str(Path(td)/'absent'),'--check']),0)
if __name__=='__main__':unittest.main()
