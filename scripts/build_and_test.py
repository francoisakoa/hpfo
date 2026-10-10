#!/usr/bin/env python3
"""Fresh separate builds of main PF131 and study R2; run their existing CTest tests.
This runs numerical UNIT/INTEGRATION TESTS, not a benchmark campaign.
"""
from __future__ import annotations
import argparse,datetime,json,os,shlex,subprocess,sys,tempfile
from pathlib import Path
from hpfo_common import *

def build(root,jobs,native):
    check_package(root,assets=False)
    require(jobs>=1,'jobs must be positive')
    for tool in ('cmake','ctest'):
        require(shutil.which(tool),'Missing required executable: '+tool)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (root/'.local/builds').mkdir(parents=True,exist_ok=True)
    base=Path(tempfile.mkdtemp(prefix=stamp+'-',dir=root/'.local/builds'))
    evidence=root/'validation/build'/base.name;evidence.mkdir(parents=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for k in ('LD_PRELOAD','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS'):env.pop(k,None)
    if env.get('CONDA_PREFIX'):
        env['LD_LIBRARY_PATH']=env['CONDA_PREFIX']+'/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env['PKG_CONFIG_PATH']=env['CONDA_PREFIX']+'/lib/pkgconfig'+(':'+env['PKG_CONFIG_PATH'] if env.get('PKG_CONFIG_PATH') else '')
    report={'created_utc':stamp,'status':'RUNNING','source_inventory_sha256':sha(root/SUBDIR/'provenance/SOURCE_INVENTORY.json'),
            'benchmarks_rerun':False,'tests_run':True,'native':native,'builds':[]}
    try:
        py_sources=[r for r in load(root/SUBDIR/'provenance/SOURCE_INVENTORY.json') if r['path'].endswith('.py')]
        for r in py_sources:
            compile((root/r['path']).read_bytes(),r['path'],'exec')
        report['python_source_syntax']={'status':'PASS','files':len(py_sources),'imports_or_experiments_executed':False}
        for variant,src in (('main',root/SUBDIR/'code/main-pf131'),('study',root/SUBDIR/'code/study-r2/source')):
            bd=base/variant;rec={'variant':variant,'build_dir':bd.relative_to(root).as_posix(),'status':'RUNNING','tests':0};report['builds'].append(rec)
            def run(args,name,capture=False):
                print('+ '+shlex.join([str(x) for x in args]),flush=True)
                if capture:
                    v=subprocess.run(list(map(str,args)),env=env,text=True,capture_output=True)
                    (evidence/(variant+'-'+name+'.log')).write_text(v.stdout+v.stderr)
                    require(v.returncode==0,'Command failed: '+name);return v.stdout
                with (evidence/(variant+'-'+name+'.log')).open('w') as f:
                    proc=subprocess.Popen(list(map(str,args)),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                    for line in proc.stdout:print(line,end='');f.write(line)
                    rc=proc.wait()
                require(rc==0,'Command failed: '+name+'; see '+str(evidence))
            config=['cmake','-S',src,'-B',bd,'-DCMAKE_BUILD_TYPE=Release','-DBUILD_TESTING=ON',
                    '-DPF_WITH_IPOPT=ON','-DPF_WITH_OPENMP=ON','-DPF_WITH_MUMPS=ON','-DPF_WITH_METIS=ON',
                    '-DPF_NATIVE='+('ON' if native else 'OFF'),'-DPython3_EXECUTABLE='+sys.executable]
            if env.get('CONDA_PREFIX'):config+=['-DCMAKE_PREFIX_PATH='+env['CONDA_PREFIX']]
            run(config,'configure');run(['cmake','--build',bd,'--parallel',str(jobs)],'build')
            tests=json.loads(run(['ctest','--test-dir',bd,'--show-only=json-v1'],'test-list',True))['tests']
            rec['tests']=len(tests);require(tests,'No tests configured')
            required={'core','local_incumbent','cli_integration','linear_metis_mumps'}
            require(required.issubset({t['name'] for t in tests}),'Required tests were not configured')
            version=run([bd/'pf_solver','--version'],'version',True).strip()
            require(all(x in version for x in ('ipopt=1','native_mumps=1','metis=1')),'Required backends not enabled: '+version)
            rec['version']=version
            run(['ctest','--test-dir',bd,'--output-on-failure','--no-tests=error','--parallel','1','--timeout','300'],'ctest')
            rec['status']='PASS'
        report['status']='PASS'
    except Exception as e:
        report['status']='FAIL';report['error']=str(e)
        raise
    finally:
        write_json(evidence/'BUILD_REPORT.json',report);write_json(root/'validation/build/LATEST.json',report);seal(root)
    print('CPP_MAIN=PASS ; CPP_STUDY=PASS ; TESTS=PASS ; BENCHMARK_CAMPAIGNS=NOT_RUN')
    print('BUILD_REPORT='+str(evidence/'BUILD_REPORT.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=4);p.add_argument('--native',action='store_true');a=p.parse_args()
    try:build(Path(__file__).resolve().parents[1],a.jobs,a.native)
    except Exception as e:sys.exit('BUILD_ERROR: '+str(e))
