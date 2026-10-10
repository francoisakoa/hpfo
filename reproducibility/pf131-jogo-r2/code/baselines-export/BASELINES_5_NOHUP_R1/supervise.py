#!/usr/bin/env python3
"""Five-baseline orchestration only; original solvers and results stay unchanged.

The scheduler retains a separate execution phase for resource changes. All 1200
identities are considered, but completed valid originals are not rerun. Interrupted
attempts remain archived at their original paths; a retry starts at the same seed,
not from a nonexistent algorithm checkpoint. Numerical errors block automatic retry.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
ATOMS=[38,75,76,77,98,102,103,104]
ORDER=[98,77,76,75,104,103,102,38]  # long-budget cells enter early, fixed before outcomes
METHODS=['BH-OJ','GA','PSO','CLPSO','SMC']
SOURCES={'baselines_v3_1':('BH-OJ','GA'), 'baselines_v3':('PSO','CLPSO','SMC')}
NORMAL={'target_reached','evaluation_limit'}
CENSORED={'time_limit','restart_limit','single_attempt_complete'}
THREADS=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']


def utc():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def event(root,name,**kw):
    with (root/'scheduler.events.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=utc(),event=name,**kw),allow_nan=False)+'\n')
def env_for(jobs):
    env=os.environ.copy()
    for k in ('LD_PRELOAD','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','LJ_USE_GPU'):env.pop(k,None)
    for k in THREADS:env[k]='1'
    env.update(OMP_DYNAMIC='FALSE',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',BASELINES_CONCURRENT_JOBS=str(jobs))
    return env

def topology():
    cpus=sorted(os.sched_getaffinity(0));cores=set()
    for n in cpus:
        p=Path(f'/sys/devices/system/cpu/cpu{n}/topology')
        try:cores.add(((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip()))
        except OSError:cores.add(('unknown',str(n)))
    return dict(logical_available=len(cpus),physical_available=len(cores),affinity=cpus)
def check_jobs(jobs,allow_smt=False):
    t=topology()
    if type(jobs) is not int or not 1<=jobs<=t['logical_available']:
        raise ValueError(f'jobs must be in 1..{t["logical_available"]}')
    if jobs>t['physical_available'] and not allow_smt:
        raise ValueError(f'{jobs} > {t["physical_available"]} physical cores: explicit --allow-smt required')
    return t

def option(args,k):
    for i,x in enumerate(args):
        if x==k and i+1<len(args):return args[i+1]
        if x.startswith(k+'='):return x[len(k)+1:]
    return None

def proc_stat(pid):
    try:
        line=Path(f'/proc/{pid}/stat').read_text();v=line[line.rfind(')')+2:].split()
        return dict(state=v[0],ppid=int(v[1]),ticks=int(v[11])+int(v[12]),start_ticks=int(v[19]),rss_pages=int(v[21]))
    except (OSError,ValueError,IndexError):return None

def active_originals(base,root,ignore=()):
    rows=[];base=base.resolve();root=root.resolve()
    oldroots={base/'results_baselines_v3_cpu',base/'results_baselines_v3_1_localfix_cpu'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in ignore:continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if not any(Path(x).name in ('campaign.py','run_one_baseline.py') for x in args[1:]):continue
            out=option(args,'--outdir')
            if out is None:continue
            out=Path(out);cwd=(p/'cwd').resolve();out=(out if out.is_absolute() else cwd/out).resolve()
            if out not in oldroots and not out.is_relative_to(root):continue
            stat=proc_stat(int(p.name))
            if not stat or stat['state']=='Z':continue
            rows.append(dict(pid=int(p.name),method=option(args,'--method'),atoms=option(args,'--atoms'),run=option(args,'--run'),outdir=str(out)))
        except (OSError,RuntimeError,ValueError):continue
    return rows

@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'LOCKED: {path}; no duplicate launch')
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def check_sources(a):
    expected=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for name in SOURCES:
        folder=a.base/name
        for fn,h in expected[name]['files'].items():
            p=folder/fn
            if not p.is_file() or sha(p)!=h:
                raise RuntimeError(f'Source absent or modified: {p}. No solver patch is applied by this kit.')
    return expected

def parameters(a):
    return dict(atoms=[38,75] if a.technical_smoke else ATOMS,runs=1 if a.technical_smoke else 30,
                technical_max_evals=61 if a.technical_smoke else None,job_seconds=120 if a.technical_smoke else 36000)

def candidates(a,source,methods):
    old=a.base/('results_baselines_v3_1_localfix_cpu' if source.endswith('_1') else 'results_baselines_v3_cpu')
    roots=[] if a.technical_smoke else [old]
    roots.extend(sorted(a.root.glob(f'phase_*/cohort_*/results/{source}')))
    ps=[]
    for root in roots:
        for n in parameters(a)['atoms']:
            for m in methods:ps.extend(sorted((root/f'LJ{n}'/m).glob('run*.json')))
    return ps

def probe(a,source,files,work):
    work.mkdir(parents=True,exist_ok=True)
    p=work/f'request_{source}_{uuid.uuid4().hex}.json'
    req={**parameters(a),'methods':list(SOURCES[source]),'files':[str(x) for x in files]}
    atomic(p,req)
    cmd=[a.python,str(HERE/'probe_records.py'),'--source',str(a.base/source),'--request',str(p)]
    r=subprocess.run(cmd,env=env_for(a.jobs),text=True,capture_output=True,timeout=180)
    if r.returncode:raise RuntimeError('Record probe failed: '+r.stderr[-6000:])
    return json.loads(r.stdout)

def software_signature(env):
    return {k:(env or {}).get(k) for k in ('python','numpy','scipy','system','machine')}

def select_records(allrecords):
    kept={};interrupted=[];issues=[]
    for r in allrecords:
        if r.get('errors'):
            issues.append(dict(path=r['path'],errors=r['errors']));continue
        key=tuple(r['key']);reason=r['stop_reason']
        if reason=='interrupted':interrupted.append(r);continue
        if reason not in NORMAL|CENSORED:
            issues.append(dict(path=r['path'],errors=[f'run ended with {reason}; inspect, no silent retry']));continue
        if key in kept:
            issues.append(dict(path=r['path'],errors=['duplicate final result'],other=kept[key]['path']));continue
        kept[key]=r
    return kept,interrupted,issues

def audit(a,work):
    sources=check_sources(a);allrecords=[];meta={}
    for src,methods in SOURCES.items():
        p=probe(a,src,candidates(a,src,methods),work);meta[src]={k:p[k] for k in ('templates','current_environment')}
        for r in p['records']:
            if r.get('environment') and software_signature(r['environment'])!=software_signature(p['current_environment']):
                r['errors'].append('Python/NumPy/SciPy/OS differs: explicit reconciliation required')
            allrecords.append(r)
    kept,interrupted,issues=select_records(allrecords)
    expected={(n,m,r) for n in parameters(a)['atoms'] for m in METHODS for r in range(1,parameters(a)['runs']+1)}
    return dict(expected=expected,kept=kept,interrupted=interrupted,issues=issues,meta=meta,sources=sources)

def inventory_json(inv):
    return dict(expected_count=len(inv['expected']),retained=[v for _,v in sorted(inv['kept'].items())],
                missing=[list(k) for k in sorted(inv['expected']-set(inv['kept']))],
                interrupted=inv['interrupted'],issues=inv['issues'],sources=inv['sources'],meta=inv['meta'])

def make_queues(keys):
    rank={n:i for i,n in enumerate(ORDER)}
    return {m:deque(sorted([k for k in keys if k[1]==m],key=lambda k:(rank[k[0]],k[2]))) for m in METHODS}

def next_job(queues,active,dispatched=None):
    counts=Counter(v['key'][1] for v in active.values())
    usable=[m for m in METHODS if queues[m]]
    if not usable:return None
    # A shared pool gives every method a presence and automatically redistributes capacity.
    dispatched=dispatched if dispatched is not None else Counter()
    m=min(usable,key=lambda m:(counts[m],dispatched[m],METHODS.index(m)))
    dispatched[m]+=1
    return queues[m].popleft()

def read_last_event(path):
    try:
        with path.open('rb') as f:
            f.seek(0,2);n=f.tell();f.seek(max(0,n-262144));raw=f.read()
        lines=raw.splitlines()
        for line in reversed(lines):
            try:d=json.loads(line)
            except (ValueError,TypeError):continue
            if d.get('event')=='incumbent':return d
    except OSError:pass
    return None

def resource_room(root):
    import shutil
    free=shutil.disk_usage(root).free
    if free<1024**3:return False,'less than 1 GiB disk free'
    try:
        info=dict((k,int(v.strip().split()[0])) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()))
        if info.get('MemAvailable',2**30)<1024**2:return False,'less than 1 GiB RAM available'
    except (OSError,ValueError):pass
    return True,''

def command(a,key,dest):
    n,m,r=key;src='baselines_v3_1' if m in SOURCES['baselines_v3_1'] else 'baselines_v3'
    c=[a.python,'-u',str(a.base/src/'run_one_baseline.py'),'--atoms',str(n),'--method',m,'--run',str(r),
       '--outdir',str(dest),'--eval-config',str(a.base/src/'baseline_eval_limits_v3.json'),
       '--base-seed','2024','--backend','cpu','--potential','pf131','--bounds','5.0','--initialization','box',
       '--target-tol','0.0001','--chunk-size','32','--job-seconds',str(parameters(a)['job_seconds']),
       '--restart-seconds','0','--max-attempts','0']
    if a.technical_smoke:c+=['--max-evals','61']
    return src,c

def rows(kept,expected,active):
    out=[]
    for m in METHODS:
        rs=[r for k,r in kept.items() if k[1]==m];want=sum(k[1]==m for k in expected)
        out.append(dict(method=m,completed=len(rs),expected=want,successes=sum(r['success'] for r in rs),
                        censored=sum(r['stop_reason'] in CENSORED for r in rs),
                        active=sum(v['key'][1]==m for v in active.values())))
    return out

def snapshot(a,phase,inv,kept,active,finished_new,failed,started,jobs,state,paused=None):
    now=time.monotonic();worker=[];hz=os.sysconf('SC_CLK_TCK');up=float(Path('/proc/uptime').read_text().split()[0])
    for pid,v in active.items():
        ps=proc_stat(pid);cpu=None
        if ps and v.get('sample'):
            previous_t,previous_ticks=v['sample'];dt=now-previous_t
            if dt>0:cpu=100*(ps['ticks']-previous_ticks)/hz/dt
        if ps:v['sample']=(now,ps['ticks'])
        e=read_last_event(v['events'])
        worker.append(dict(pid=pid,key=list(v['key']),state=ps['state'] if ps else 'exited',cpu_percent_interval=cpu,
                           elapsed_seconds=now-v['started'],last_incumbent_event=e,log=str(v['log']),
                           events=str(v['events']),cohort=v['cohort']))
    elapsed=now-started;rate=finished_new/elapsed*3600 if elapsed else None
    left=len(inv['expected'])-len(kept)
    deadline=json.loads((a.root/'deadline.json').read_text())['deadline_epoch']
    hoursleft=max(0,(deadline-time.time())/3600)
    d=dict(at=utc(),status=state,pid=os.getpid(),phase=str(phase),jobs_limit=jobs,elapsed_hours=elapsed/3600,
           expected=len(inv['expected']),retained_before=len(inv['kept']),completed=len(kept),missing=left,
           completed_this_session=finished_new,failed=failed,active=len(active),paused=paused,
           required_results_per_hour_for_week=left/hoursleft if hoursleft else None,
           completed_per_hour_this_session=rate,
           pace_scope='Observed throughput is not an ETA: cells/methods have different costs. The 7-day target does not terminate runs.',
           remaining_upper_nominal_run_hours=left*parameters(a)['job_seconds']/3600,
           workers=worker,methods=rows(kept,inv['expected'],active))
    atomic(a.root/'status.json',d);return d

def show(a):
    d=json.loads((a.root/'status.json').read_text());age=time.time()-(a.root/'status.json').stat().st_mtime
    ps=proc_stat(d['pid']);live=bool(ps and ps['state']!='Z')
    if 'methods' not in d:
        print(json.dumps(d,indent=2));print(f'Supervisor present={live}; age={age:.1f}s');return 0
    print(f"{d['at']} | {d['status']} | supervisor PID={d['pid']} present={live} | snapshot age={age:.0f}s")
    print(f"RESULTS {d['completed']}/{d['expected']} | missing={d['missing']} | active={d['active']} | slots={d['jobs_limit']} | failures={len(d['failed'])}")
    print('METHOD   DONE/TOTAL HITS CENS ACTIVE')
    for r in d['methods']:print(f"{r['method']:7s} {r['completed']:4d}/{r['expected']:<5d} {r['successes']:4d} {r['censored']:4d} {r['active']:6d}")
    print('PID       RUN                     STATE CPU%    LAST ENERGY / COUNT (lower bound)')
    for w in d['workers']:
        e=w['last_incumbent_event'] or {};cpu=w['cpu_percent_interval'];cpu='-' if cpu is None else f'{cpu:.1f}'
        print(f"{w['pid']:<9} LJ{w['key'][0]} {w['key'][1]:6s} run{w['key'][2]:02d} {w['state']:>6s} {cpu:>7s} {json.dumps(e)}")
    need=d.get('required_results_per_hour_for_week');rate=d.get('completed_per_hour_this_session')
    print(f'Week target: required remaining throughput={need}; observed completions/hour={rate}. Not a guaranteed ETA.')
    if d.get('paused'):print('PAUSED:',d['paused'])
    if d['failed']:print('FAILED:',json.dumps(d['failed'],indent=2))
    print('Read-only snapshot; CPU activity is not a proof of objective progress.')
    return 0


def run(a):
    try:
        if os.getsid(0)!=os.getpid():os.setsid()
    except PermissionError:
        pass  # nohup already protects SIGHUP; streams are redirected by the shell wrapper.
    a.root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(lock(a.root/'supervisor.lock'))
        if not (a.root/'deadline.json').exists():
            atomic(a.root/'deadline.json',dict(created=utc(),deadline_epoch=time.time()+7*86400,policy='planning target only; never cuts a scientific run'))
        existing=active_originals(a.base,a.root)
        if existing:raise RuntimeError('Old launcher/workers still active. Stop the original launcher with Ctrl+C, wait, then restart. '+json.dumps(existing))
        for old in (a.base/'results_baselines_v3_cpu',a.base/'results_baselines_v3_1_localfix_cpu'):
            if old.exists():stack.enter_context(lock(old/'_campaign'/'campaign.lock'))
        phase=a.root/('phase_'+stamp());phase.mkdir()
        started=time.monotonic();atomic(a.root/'status.json',dict(status='PREPARING',at=utc(),pid=os.getpid(),phase=str(phase)))
        inv=audit(a,phase/'intake');atomic(phase/'inventory_before.json',inventory_json(inv))
        if inv['issues']:raise RuntimeError(f'{len(inv["issues"])} original-data issues; see {phase}/inventory_before.json. Nothing rerun.')
        atomic(phase/'SESSION.json',dict(started=utc(),pid=os.getpid(),mode='TECHNICAL_SMOKE' if a.technical_smoke else 'PRODUCTION',
             python=a.python,parameters=parameters(a),jobs_initial=a.jobs,topology=topology(),source_lock=inv['sources'],
             original_folders_read_only=True,resource_change='New concurrency cohort; original results and checksums unchanged',
             seven_day_target='Scheduling objective, not an extra optimizer termination criterion'))
        for src in SOURCES:
            p=subprocess.run([a.python,str(a.base/src/'preflight.py'),'--backend','cpu','--report',str(phase/f'preflight_{src}.json')],
                             env=env_for(a.jobs),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (phase/f'preflight_{src}.log').write_text(p.stdout)
            if p.returncode:raise RuntimeError(f'Preflight failed for {src}: {p.stdout[-3000:]}')
        atomic(a.root/'control.json',dict(jobs=a.jobs,allow_smt=a.allow_smt,at=utc()))
        kept=dict(inv['kept']);queues=make_queues(inv['expected']-set(kept));active={};failed=[];finished_new=0
        jobs=a.jobs;cohort=0;dispatch_counts=Counter();last_write=0.;last_pulse=0.;last_verify=time.monotonic();stop=[False];sent_stop=False
        def request_stop(sig,frame):stop[0]=True
        previous={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
        print(f'READY: five baselines, {jobs} global slots, {len(kept)} retained, {len(inv["expected"])-len(kept)} pending; phase={phase}',flush=True)
        event(a.root,'ready',phase=str(phase),jobs=jobs,retained=len(kept))
        try:
            while any(queues.values()) or active:
                ctl=json.loads((a.root/'control.json').read_text());newjobs=ctl['jobs'];check_jobs(newjobs,ctl.get('allow_smt',False))
                if newjobs!=jobs:
                    jobs=newjobs;cohort+=1;event(a.root,'resource_change',jobs=jobs,cohort=cohort,active=len(active))
                    print(f'RESOURCES: limit now {jobs}; existing runs are not interrupted',flush=True)
                if time.monotonic()-last_verify>60:
                    check_sources(a);last_verify=time.monotonic()
                    aliens=active_originals(a.base,a.root,ignore=active)
                    if aliens:raise RuntimeError('Another solver launcher appeared: '+json.dumps(aliens))
                if stop[0] and not sent_stop:
                    for v in active.values():
                        if v['proc'].poll() is None:os.killpg(v['proc'].pid,signal.SIGTERM)
                    sent_stop=True
                room,why=resource_room(a.root)
                while not stop[0] and room and len(active)<jobs:
                    key=next_job(queues,active,dispatch_counts)
                    if key is None:break
                    n,m,r=key;src='baselines_v3_1' if m in ('BH-OJ','GA') else 'baselines_v3'
                    # Each concurrency epoch has a separate output tree; no hash editing.
                    dest=phase/f'cohort_{cohort:03d}_j{jobs}'/'results'/src
                    _,cmd=command(a,key,dest);log=phase/'logs'/f'LJ{n}_{m}_run{r:02d}.log';log.parent.mkdir(exist_ok=True)
                    fh=log.open('w');proc=subprocess.Popen(cmd,cwd=a.base/src,env=env_for(jobs),stdin=subprocess.DEVNULL,
                        stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
                    entry=dict(proc=proc,fh=fh,key=key,source=src,dest=dest,log=log,started=time.monotonic(),cohort=cohort,
                        result=dest/f'LJ{n}'/m/f'run{r:02d}.json',events=dest/f'LJ{n}'/m/f'run{r:02d}.events.jsonl')
                    active[proc.pid]=entry;event(a.root,'start',pid=proc.pid,key=list(key),command=cmd,cohort=cohort,jobs=jobs)
                    print(f'START PID={proc.pid} LJ{n}_{m}_run{r:02d} active={len(active)}/{jobs}',flush=True)
                for pid,v in list(active.items()):
                    rc=v['proc'].poll()
                    if rc is None:continue
                    v['fh'].close();del active[pid]
                    error=None;rec=None
                    if not v['result'].is_file():error='No final JSON'
                    else:
                        p=probe(a,v['source'],[v['result']],phase/'checks');rec=p['records'][0]
                        if rec['errors']:error='; '.join(rec['errors'])
                        elif rec['stop_reason'] not in NORMAL|CENSORED:error='stop_reason='+str(rec['stop_reason'])
                    if rc or error:
                        failed.append(dict(key=list(v['key']),returncode=rc,error=error,path=str(v['result']),log=str(v['log'])))
                        print(f'FAIL {v["key"]} rc={rc} {error}',flush=True)
                    else:
                        kept[v['key']]=rec;finished_new+=1
                        print(f'DONE {v["key"]} {rec["stop_reason"]} E={rec["energy"]} evals={rec["evaluations"]} sec={rec["seconds"]:.1f}',flush=True)
                    event(a.root,'finish',pid=pid,key=list(v['key']),returncode=rc,error=error)
                if time.monotonic()-last_write>=10 or stop[0]:
                    d=snapshot(a,phase,inv,kept,active,finished_new,failed,started,jobs,'DRAINING' if stop[0] else 'RUNNING',None if room else why)
                    if time.monotonic()-last_pulse>60:
                        print(f"PROGRESS {d['completed']}/{d['expected']} active={d['active']}/{jobs} failures={len(failed)}",flush=True);last_pulse=time.monotonic()
                    last_write=time.monotonic()
                if stop[0] and not active:break
                time.sleep(.5)
        finally:
            for sig,h in previous.items():signal.signal(sig,h)
            if active:
                # On a supervisor exception request orderly stops, not SIGKILL.
                for v in active.values():
                    if v['proc'].poll() is None:os.killpg(v['proc'].pid,signal.SIGTERM)
                for v in active.values():
                    try:v['proc'].wait(timeout=60)
                    except subprocess.TimeoutExpired:print(f'Worker {v["proc"].pid} still stopping; do not relaunch yet',flush=True)
                    v['fh'].close()
        state='INTERRUPTED' if stop[0] else ('FINISHED_WITH_ERRORS' if failed else 'FINISHED')
        snapshot(a,phase,inv,kept,{},finished_new,failed,started,jobs,state)
        # Consolidation preserves original paths, hashes, versions and execution cohorts.
        atomic(a.root/'results_index.json',dict(at=utc(),scope='COHORT_AWARE_INDEX_NOT_FINAL_MANUSCRIPT_AUDIT',
            expected_count=len(inv['expected']),records=[v for _,v in sorted(kept.items())],failed=failed,
            interrupted_before=inv['interrupted'],technical=a.technical_smoke,
            timing='Do not pool timings across concurrency cohorts without disclosure. Time-limit outcomes kept explicitly.'))
        print(f'{state}: {len(kept)}/{len(inv["expected"])} retained; failures={len(failed)}',flush=True)
        return 130 if stop[0] else (2 if failed else 0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['plan','run','status','set-jobs','stop'])
    ap.add_argument('--base',type=Path,default=Path('/data/projects/PF_GlobalOptimization'))
    ap.add_argument('--root',type=Path)
    ap.add_argument('--python',default=sys.executable)
    ap.add_argument('--jobs',type=int,default=22)
    ap.add_argument('--allow-smt',action='store_true')
    ap.add_argument('--technical-smoke',action='store_true',help='10 tiny tests, 61 evaluations; NEVER manuscript data')
    a=ap.parse_args();a.base=a.base.resolve();a.python=str(Path(a.python).absolute())
    a.root=(a.root or a.base/('results_baselines_nohup_TECHNICAL' if a.technical_smoke else 'results_baselines_nohup_R1')).resolve()
    if a.action=='status':return show(a)
    if a.action=='stop':
        d=json.loads((a.root/'status.json').read_text());pid=d['pid'];p=Path(f'/proc/{pid}/cmdline')
        args=p.read_bytes().split(b'\0')
        if str(HERE/'supervise.py').encode() not in args or b'run' not in args:raise RuntimeError('PID does not identify this supervisor')
        os.kill(pid,signal.SIGTERM);print(f'Stop requested for PID {pid}; partial runs are preserved.');return 0
    check_jobs(a.jobs,a.allow_smt)
    if a.action=='set-jobs':
        atomic(a.root/'control.json',dict(jobs=a.jobs,allow_smt=a.allow_smt,at=utc()))
        print(f'Concurrency requested: {a.jobs}; effective at next scheduler poll. Existing work is not killed.');return 0
    if a.action=='plan':
        work=a.root/'plans'/stamp();inv=audit(a,work);d=inventory_json(inv);d['active_processes']=active_originals(a.base,a.root)
        atomic(work/'PLAN.json',d)
        print(f'PLAN: {len(inv["kept"])}/{len(inv["expected"])} retained; {len(d["missing"])} to compute; {len(inv["interrupted"])} interrupted records preserved; issues={len(inv["issues"])}')
        print(f'All five: {METHODS}; TOTAL concurrency={a.jobs}, not {a.jobs} per method.')
        print(f'PLAN_FILE={work}/PLAN.json')
        if d['active_processes']:print('ACTIVE_OLD_JOBS:',json.dumps(d['active_processes']))
        return 2 if inv['issues'] else 0
    return run(a)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(2)
