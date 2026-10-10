#!/usr/bin/env python3
"""PF131: grids of Tables 7--10, LJ75/LJ102, refine-all in the PF loop. Stdlib runner; real output only, never invented.
The original PF131 installation is never modified. See PROTOCOLE.md before use.
"""
from __future__ import annotations
import argparse, contextlib, csv, datetime as dt, fcntl, hashlib, io, json, math
import os, platform, random, shutil, signal, statistics, subprocess, sys, tarfile, time
from pathlib import Path

KIT = Path(__file__).resolve().parent
SCHEMA = 'PF131_ARTICLE_STUDY_R1'  # Unchanged C++ interface.
CAMPAIGN_SCHEMA = 'PF131_LJ75_LJ102_REFINE_ALL_R2'
ATOMS = (75, 102)
TARGETS = {75: -397.492331, 102: -569.363653}
TERMINAL = {'target_reached', 'iteration_limit', 'stagnation_limit', 'evaluation_limit', 'time_limit'}
ENV = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
for key in ('LD_PRELOAD', 'ASAN_OPTIONS', 'UBSAN_OPTIONS', 'LSAN_OPTIONS'):
    ENV.pop(key, None)

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='microseconds')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, x):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)

def log(message):
    print(f'[{utc()}] {message}', flush=True)

@contextlib.contextmanager
def lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.study.lock').open('a') as f:
        try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'Campagne deja active : {out}')
        yield

def configurations():
    # Empty overrides mean the full control specified in default_options().
    reduced = dict(ipf_enabled=False, lambda_share=0.0, lambda_novelty=0.0,
                   kicks=False, islands=False, island_exchange=False,
                   diversity_injection=False, ess_rejuvenation=False)
    nolocal = dict(local_method='none', refine_top_k=0, refine_all=False, final_refinement=False)
    items = [
        ('full','Full HPF',{}),
        ('no_ipf','- IPF operator',dict(ipf_enabled=False)),
        ('no_sharing','- Sharing metric',dict(lambda_share=0.0)),
        ('no_novelty','- Novelty metric',dict(lambda_novelty=0.0)),
        ('no_kicks','- Kick mechanism',dict(kicks=False)),
        ('no_island_exchange','- Island exchange',dict(island_exchange=False)),
        ('no_local','- Local optimization',nolocal),
        ('no_adaptive_annealing','- Adaptive annealing',dict(adaptive_annealing=False)),
        ('pf_local_only','Only PF + local opt (no diversity)',reduced),
        ('pf_only','Only PF (no local, no diversity)',dict(reduced, **nolocal))]
    result = [dict(id=i,label=label,table=7,changes=changes) for i,label,changes in items]
    for n in (100,200,300,400,500):
        result.append(dict(id=f'np_{n}',label=f'Np={n}',table=8,changes=dict(particles=n)))
    for name, b in [('1e-5',1e-5),('1e-4',1e-4),('1e-3',1e-3),('1e-2',1e-2)]:
        result.append(dict(id='beta_'+name,label=f'beta_min={name}',table=9,
                           changes=dict(beta_initial=b), alias='full' if b==1e-4 else None))
    for name,a in [('0.95',.95),('0.98',.98),('0.99',.99),('1.00',1.0)]:
        result.append(dict(id='alpha_'+name,label=f'alpha_0={name}',table=10,
                           changes=dict(ipf_alpha_initial=a),alias='full' if a==.98 else None))
    return result

def default_options(n, p):
    return dict(particles=max(3*n+56,50), min_particles=5, max_iterations=p['max_iter'],
        no_improve_stop=0, max_seconds=p['max_time'], max_objective_evaluations=p['max_evals'],
        radius=5.0, refine_top_k=5, refine_all=True, final_refinement=True,
        local_method='ipopt', local_max_iterations=5000, local_tolerance=1e-8,
        batch_threads=p['threads'], local_workers=p['workers'], ipopt_linear_solver='mumps',
        ipopt_ball_constraints=True, ipopt_exact_hessian=True, flow='invertible',mapping='linear',
        clustering='kmeans', ipf_enabled=True, ipf_schedule='linear-endpoints',
        ipf_alpha_initial=.98, ipf_alpha_final=.05, adaptive_annealing=True,
        beta_initial=1e-4,beta_max=5.0,mutation_sigma=.05,lambda_share=.01,lambda_novelty=.01,
        kicks=True,islands=True,island_exchange=True,diversity_injection=True,ess_rejuvenation=True,
        target=TARGETS[n],target_tolerance=1e-4)

def seed(base,n,r,offset):
    return int(hashlib.md5(f'{base}_{n}_{r}_{offset}'.encode()).hexdigest()[:8],16)

def task_id(config,n,r): return f'{config}/LJ{n}/run_{r:03d}'

def make_plan(p):
    configs=configurations(); tasks=[]; rows=[]
    for c in configs:
        count=p['ablation_runs'] if c['table']==7 else p['sensitivity_runs']
        for n in ATOMS:
            opts=default_options(n,p); opts.update(c['changes'])
            if c['id']=='no_adaptive_annealing': opts['beta_initial']=p['fixed_beta']
            for r in range(1,count+1):
                tid=task_id(c.get('alias') or c['id'],n,r)
                rows.append(dict(table=c['table'],configuration=c['id'],label=c['label'],
                                 atoms=n,run=r,task=tid,expected_runs=count))
                if not c.get('alias'):
                    tasks.append(dict(id=tid,configuration=c['id'],atoms=n,run=r,options=opts,
                        seed_init=seed(p['base_seed'],n,r,0),seed_pf=seed(p['base_seed'],n,r,1000)))
    # Randomized blocks: no concurrent solves, paired run IDs; actual order is saved.
    tasks.sort(key=lambda t:(t['run'],t['atoms'],t['configuration']))
    ordered=[]
    for r in range(1,p['ablation_runs']+1):
        for n in ATOMS:
            group=[t for t in tasks if t['run']==r and t['atoms']==n]
            random.Random(seed(p['base_seed'],n,r,2000)).shuffle(group)
            ordered.extend(group)
    return dict(schema=CAMPAIGN_SCHEMA,solver_interface=SCHEMA,
                study_design=dict(atoms=list(ATOMS),refinement='all-current-particles-in-PF-loop',
                    terminal_refinement='native-PF131-top5-if-target-not-reached',
                    local_disabled=['no_local','pf_only'],
                    amendment='LJ98 replaced by LJ102; top5 replaced by refine-all; no reuse of top5 results'),
                protocol=p,configurations=configs,tasks=ordered,table_rows=rows,
                unique_solves=len(ordered),logical_rows=len(rows))

def solver_args(options):
    numeric = {'particles':'--particles','min_particles':'--min-particles',
        'max_iterations':'--max-iter','no_improve_stop':'--no-improve-stop',
        'max_seconds':'--max-time','max_objective_evaluations':'--max-evals','radius':'--radius',
        'refine_top_k':'--refine-top-k','local_method':'--local-refine',
        'local_max_iterations':'--local-max-iter','local_tolerance':'--local-tol',
        'batch_threads':'--threads','local_workers':'--workers',
        'ipopt_linear_solver':'--ipopt-linear-solver','flow':'--flow','mapping':'--mapping',
        'clustering':'--clustering','ipf_alpha_initial':'--ipf-alpha-start',
        'ipf_alpha_final':'--ipf-alpha-end','beta_initial':'--beta-initial','beta_max':'--beta-max',
        'mutation_sigma':'--mutation-sigma','lambda_share':'--lambda-share',
        'lambda_novelty':'--lambda-novelty','target':'--target','target_tolerance':'--target-tol'}
    args=[]
    for key,flag in numeric.items(): args.extend((flag,str(options[key])))
    # Native PF131 uses refine_all in the PF loop; refine_top_k remains the
    # size of the terminal refinement. Never enable all on a no-local arm.
    if options['refine_all']:
        if options['local_method']=='none': raise RuntimeError('refine_all incompatible with no-local arm')
        args.append('--refine-all')
    args.extend(('--ipf-schedule','linear','--algorithm','pf'))
    off={'ipf_enabled':'--no-ipf','adaptive_annealing':'--no-adaptive-annealing',
         'island_exchange':'--no-island-exchange','islands':'--no-islands','kicks':'--no-kicks',
         'diversity_injection':'--no-diversity-injection','ess_rejuvenation':'--no-ess-rejuvenation',
         'final_refinement':'--no-final-refine'}
    for key,flag in off.items():
        if not options[key]: args.append(flag)
    return args

def source_inventory():
    paths=[p for root in ('source','reference') for p in (KIT/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in KIT.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md')]
    return {str(p.relative_to(KIT)):sha(p) for p in sorted(paths)}

def initialize(out, p, binary):
    requested=make_plan(p); requested['kit_source_hash']=digest(source_inventory())
    path=out/'PLAN.json'
    if path.exists():
        old=load(path)
        if old != requested: raise RuntimeError('Plan/configuration/source differents : nouveau OUT requis.')
    else:
        leftovers=[x for x in out.iterdir() if x.name!='.study.lock']
        if leftovers: raise RuntimeError('OUT non vide sans PLAN.json : aucune sortie remplacee.')
        write_json(path,requested)
    if not binary.is_file(): raise RuntimeError(f'Binaire absent : {binary}; compiler avec build.sh.')
    worker=binary.with_name('pf_local_worker')
    if not worker.is_file(): raise RuntimeError(f'Worker absent : {worker}')
    identity=dict(solver_sha256=sha(binary),worker_sha256=sha(worker))
    prov=out/'provenance'; identity_file=prov/'BINARY.json'
    if identity_file.exists():
        if load(identity_file)['identity']!=identity: raise RuntimeError('Binaire modifie depuis le debut : nouveau OUT requis.')
    else:
        dst=prov/'bin'; dst.mkdir(parents=True,exist_ok=True)
        for path in (binary,worker): shutil.copy2(path,dst/path.name)
        version=subprocess.check_output([str(dst/'pf_solver'),'--version'],env=ENV,text=True,timeout=20)
        schema=subprocess.check_output([str(dst/'pf_solver'),'--study-schema'],env=ENV,text=True,timeout=20).strip()
        if schema!=SCHEMA or 'ipopt=1' not in version: raise RuntimeError('Binaire incompatible : schema/Ipopt.')
        for name in ('source','reference','tests'):
            if (KIT/name).exists(): shutil.copytree(KIT/name,prov/'kit'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
        for path in KIT.iterdir():
            if path.is_file(): shutil.copy2(path,prov/'kit'/path.name)
        build=prov/'build'; build.mkdir(exist_ok=True)
        for pattern in ('REUSE_BINARY.json','reference_build/**/*','CMakeCache.txt','compile_commands.json','*.log','CMakeFiles/*/flags.make','CMakeFiles/*/link.txt','Testing/Temporary/LastTest.log'):
            for path in binary.parent.glob(pattern):
                if not path.is_file(): continue
                target=build/path.relative_to(binary.parent); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
        info=dict(captured_utc=utc(),platform=platform.platform(),python=sys.version,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                  environment={k:ENV.get(k) for k in ('CONDA_PREFIX','OMP_NUM_THREADS','OMP_DYNAMIC','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')})
        for cmd in (['lscpu'],['free','-h'],['readelf','-d',str(binary)]):
            try:
                proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20,env=ENV)
                info[cmd[0]]=dict(returncode=proc.returncode,output=proc.stdout)
            except (OSError,subprocess.TimeoutExpired) as e: info[cmd[0]]=str(e)
        prefix=ENV.get('CONDA_PREFIX')
        if prefix and (Path(prefix)/'conda-meta').is_dir():
            shutil.copytree(Path(prefix)/'conda-meta',prov/'conda-meta',dirs_exist_ok=True)
        write_json(prov/'ENVIRONMENT.json',info)
        write_json(prov/'SOURCE_SHA256.json',source_inventory())
        write_json(identity_file,dict(identity=identity,version=version,collected_utc=utc(),original_path=str(binary)))
    for name,key in [('pf_solver','solver_sha256'),('pf_local_worker','worker_sha256')]:
        if sha(prov/'bin'/name)!=identity[key]: raise RuntimeError('Copie binaire corrompue.')
    return requested,prov/'bin'/'pf_solver'

def execute(cmd, logfile, timeout=None):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('wb') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=ENV,start_new_session=True)
        try: return p.wait(timeout=timeout)
        except BaseException:
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                p.wait()
            raise

def population_shape(path, n, count):
    with path.open() as f:
        header=f.readline().split()
        if header!=['PFPOP1',str(count),str(3*n)]: raise RuntimeError(f'Population mal dimensionnee : {path}')
        rows=0
        for line in f:
            if not line.strip(): continue
            values=[float(v) for v in line.split()]
            if len(values)!=3*n or not all(math.isfinite(v) and -5<=v<=5 for v in values):
                raise RuntimeError(f'Population invalide : {path}')
            rows+=1
        if rows!=count: raise RuntimeError(f'Nombre de particules invalide : {path}')

def next_attempt(root):
    root.mkdir(parents=True,exist_ok=True)
    k=1
    while (root/f'attempt_{k:03d}').exists(): k+=1
    path=root/f'attempt_{k:03d}'; path.mkdir(); return path

def frozen_population(out,binary,p,t):
    n,r,count=t['atoms'],t['run'],t['options']['particles']
    root=out/'inputs'/f'LJ{n}'/f'np_{count}'/f'run_{r:03d}'
    mark=root/'READY.json'
    if mark.exists():
        m=load(mark); pop=out/m['file']
        if sha(pop)!=m['sha256']: raise RuntimeError(f'Population geleee modifiee : {pop}')
        population_shape(pop,n,count); return pop
    attempt=next_attempt(root)
    raw=attempt/'raw'
    cmd=[str(binary),'--atoms',str(n),'--run-start',str(r),'--runs','1','--seed',str(p['base_seed']),
         '--particles',str(count),'--generate-only','--output-dir',str(raw)]
    rc=execute(cmd,attempt/'generation.log',timeout=120)
    if rc: raise RuntimeError(f'Generation population echouee rc={rc} : {attempt}')
    pop=raw/f'LJ{n}_run{r}'/'initial.pfpop'; population_shape(pop,n,count)
    init=load(pop.with_name('initialization.json'))
    if (init['seed_pf'],init['seed_init'])!=(t['seed_pf'],t['seed_init']): raise RuntimeError('Graines generation incoherentes.')
    write_json(mark,dict(file=str(pop.relative_to(out)),sha256=sha(pop),created_utc=utc(),command=cmd,
        policy='native PF131: population generated from seed_pf; x0 from seed_init is unused'))
    return pop

def validate_result(folder,t,pop):
    needed=('pf.json','pf.xyz','pf_iterations.csv','initial.pfpop','initialization.json')
    for name in needed:
        if not (folder/name).is_file() or (folder/name).stat().st_size==0: raise RuntimeError(f'Piece manquante : {folder/name}')
    j=load(folder/'pf.json'); o=j.get('options',{})
    if j.get('atoms')!=t['atoms'] or j.get('run')!=t['run'] or j.get('status') not in TERMINAL:
        raise RuntimeError('Identite/statut du resultat invalide.')
    if o.get('seed_pf')!=t['seed_pf'] or o.get('seed_init')!=t['seed_init']: raise RuntimeError('Graines du resultat invalides.')
    for key,value in t['options'].items():
        if o.get(key)!=value: raise RuntimeError(f'Option non appliquee {key}: {o.get(key)} != {value}')
    f=j.get('objective'); seconds=j.get('seconds'); x=j.get('x',[])
    if not isinstance(f,(int,float)) or not math.isfinite(f): raise RuntimeError('Objectif non fini.')
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0: raise RuntimeError('Temps invalide.')
    if len(x)!=3*t['atoms'] or not all(math.isfinite(v) for v in x): raise RuntimeError('Coordonnees invalides.')
    if j['objective_evaluations']!=j['objective_pf']+j['objective_local']: raise RuntimeError('Compteurs incoherents.')
    if sha(folder/'initial.pfpop')!=sha(pop): raise RuntimeError('Population initiale non identique.')
    history=j.get('history',[])
    if len(history)!=j['iterations']: raise RuntimeError('Historique tronque.')
    with (folder/'pf_iterations.csv').open() as stream: trace=list(csv.DictReader(stream))
    if len(trace)!=len(history): raise RuntimeError('Historique CSV/JSON incoherent.')
    for h in history:
        if not h['weights'] or len(h['weights'])!=len(h['objectives']): raise RuntimeError('Poids/objectifs mal dimensionnes.')
        if not all(math.isfinite(v) for v in h['weights']+h['objectives']): raise RuntimeError('Trace non finie.')
        if not o['adaptive_annealing'] and (h['beta_used']!=o['beta_initial'] or h['beta_next']!=o['beta_initial']):
            raise RuntimeError('Ablation annealing inactive.')
        if not o['ipf_enabled'] and h['ipf_alpha']!=0: raise RuntimeError('Ablation IPF inactive.')
        if not o['kicks'] and h['kick']: raise RuntimeError('Ablation kicks inactive.')
    if not o['island_exchange'] and any(int(h['island_exchange']) for h in trace): raise RuntimeError('Ablation islands inactive.')
    if o['local_method']=='none' and (j['local_calls'] or j['objective_local']): raise RuntimeError('Ablation locale inactive.')
    success=f<=o['target']+o['target_tolerance']
    if j.get('target_reached')!=success: raise RuntimeError('Indicateur de succes incoherent.')
    coords=[]
    with (folder/'pf.xyz').open() as stream:
        if int(stream.readline())!=t['atoms']: raise RuntimeError('XYZ invalide.')
        stream.readline()
        for line in stream:
            if line.strip(): coords.extend(float(v) for v in line.split()[1:])
    if coords!=x: raise RuntimeError('XYZ et x JSON differents.')
    return j,{name:sha(folder/name) for name in needed}

def done_result(out,t,verify=True):
    root=out/'runs'/t['id']; marker=root/'DONE.json'
    if not marker.exists(): return None
    m=load(marker)
    if m['task_sha256']!=digest(t): raise RuntimeError(f'Identite tache modifiee : {marker}')
    folder=out/m['folder']
    if verify:
        for name,expected in m['hashes'].items():
            if sha(folder/name)!=expected: raise RuntimeError(f'Resultat modifie : {folder/name}')
    j=load(folder/'pf.json') if verify else m['summary']
    return dict(record=j,marker=m)

def run_task(out,binary,p,t):
    if done_result(out,t): return False
    pop=frozen_population(out,binary,p,t)
    root=out/'runs'/t['id']; attempt=next_attempt(root)
    raw=attempt/'raw'; n,r=t['atoms'],t['run']
    cmd=[str(binary),'--atoms',str(n),'--runs','1','--run-start',str(r),
         '--seed',str(p['base_seed']),'--population',str(pop),
         '--worker-executable',str(binary.with_name('pf_local_worker')), *solver_args(t['options']),
         '--output-dir',str(raw)]
    meta=dict(task=t,command=cmd,command_shell=__import__('shlex').join(cmd),started_utc=utc(),
        binary_sha256=sha(binary),worker_sha256=sha(binary.with_name('pf_local_worker')),
        population_sha256=sha(pop),wall_seconds_scope='whole CLI, including serialization; not solver.seconds')
    write_json(attempt/'EXECUTION.json',meta)
    started=time.monotonic(); log(f"DEBUT {t['id']} seed_pf={t['seed_pf']} LOG={attempt/'solver.log'}")
    try:
        rc=execute(cmd,attempt/'solver.log')
        meta.update(returncode=rc,finished_utc=utc(),cli_wall_seconds=time.monotonic()-started)
        write_json(attempt/'EXECUTION.json',meta)
        if rc: raise RuntimeError(f'Solveur rc={rc}; voir {attempt}/solver.log')
        folder=raw/f'LJ{n}_run{r}'
        j,hashes=validate_result(folder,t,pop)
        mark=dict(task_sha256=digest(t),folder=str(folder.relative_to(out)),hashes=hashes,
            finished_utc=meta['finished_utc'],cli_wall_seconds=meta['cli_wall_seconds'],
            status=j['status'],success=j['target_reached'],
            summary={key:j[key] for key in ('status','target_reached','objective','seconds','objective_evaluations','iterations','options')})
        write_json(root/'DONE.json',mark)
        log(f"FIN {t['id']} {j['status']} f={j['objective']:.12f} seconds={j['seconds']:.3f} evals={j['objective_evaluations']}")
        return True
    except BaseException as e:
        meta.update(finished_utc=utc(),cli_wall_seconds=time.monotonic()-started,error=str(e),completed=False)
        write_json(attempt/'EXECUTION.json',meta); raise

def csv_write(path,rows,fields):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp,path)

def summary(out,plan,verbose=False):
    data={t['id']:done_result(out,t,verify=False) for t in plan['tasks']}
    report=out/'reports'; report.mkdir(exist_ok=True)
    detail=[]
    for row in plan['table_rows']:
        result=data[row['task']]
        if not result: continue
        j,m=result['record'],result['marker']
        detail.append(dict(table=row['table'],configuration=row['configuration'],atoms=row['atoms'],run=row['run'],
            status=j['status'],success=int(j['target_reached']),objective=j['objective'],seconds=j['seconds'],
            objective_evaluations=j['objective_evaluations'],iterations=j['iterations'],
            seed_pf=j['options']['seed_pf'],seed_init=j['options']['seed_init'],finished_utc=m['finished_utc'],
            raw_result=m['folder']+'/pf.json',task=row['task']))
    fields=['table','configuration','atoms','run','status','success','objective','seconds','objective_evaluations',
            'iterations','seed_pf','seed_init','finished_utc','raw_result','task']
    csv_write(report/'RUNS.csv',detail,fields)
    aggregates=[]; by_id={c['id']:c for c in plan['configurations']}
    for config,c in by_id.items():
        expected=plan['protocol']['ablation_runs'] if c['table']==7 else plan['protocol']['sensitivity_runs']
        for n in ATOMS:
            rows=[d for d in detail if d['configuration']==config and d['atoms']==n]
            success=sum(d['success'] for d in rows)
            obj=[d['objective'] for d in rows]; sec=[d['seconds'] for d in rows]; ev=[d['objective_evaluations'] for d in rows]
            record=dict(table=c['table'],configuration=config,atoms=n,completed=len(rows),expected=expected,success=success,
                success_percent=100*success/expected if len(rows)==expected else None,
                success_among_completed_percent=100*success/len(rows) if rows else None,
                best_objective=min(obj) if obj else None,mean_objective=statistics.mean(obj) if obj else None,
                sd_objective=statistics.stdev(obj) if len(obj)>1 else None,
                mean_seconds=statistics.mean(sec) if sec else None,median_seconds=statistics.median(sec) if sec else None,
                mean_evaluations=statistics.mean(ev) if ev else None,median_evaluations=statistics.median(ev) if ev else None,
                mean_iterations=statistics.mean([d['iterations'] for d in rows]) if rows else None,
                mean_seconds_success_only=statistics.mean([d['seconds'] for d in rows if d['success']]) if success else None)
            aggregates.append(record)
    csv_write(report/'SUMMARY.csv',aggregates,list(aggregates[0]))
    completed=sum(v is not None for v in data.values())
    state=dict(unique_completed=completed,unique_expected=len(plan['tasks']),
               logical_rows_completed=len(detail),logical_rows_expected=len(plan['table_rows']),
               complete=completed==len(plan['tasks']),generated_utc=utc())
    write_json(report/'STATUS.json',state)
    for table in (7,8,9,10):
        subset=[x for x in aggregates if x['table']==table]
        csv_write(report/f'TABLE_{table}.csv',subset,list(aggregates[0]))
    text=['# Etudes PF131 — LJ75 / LJ102 — REFINE ALL', '',
          'Boucle PF : toutes les particules courantes. Exceptions no_local/pf_only : aucun appel local.',
          'Raffinement terminal natif conserve : top5, uniquement si la cible n’est pas deja atteinte.',
          'Protocole amende : autres tailles/reglage que le manuscrit initial; pas de fusion avec les resultats top5.', '',f"Calculs uniques termines : {completed}/{len(plan['tasks'])}",
          'Succes = objectif <= cible + 1e-4. Moyennes et medianes : tous les runs termines, sauf colonne explicite.',
          'Un resultat manquant n’est pas un echec scientifique. Pourcentage final vide avant completude.',
          'Defaults beta=1e-4 et alpha=0.98 : premieres repetitions du controle full, pas de nouveaux calculs.', '',
          '|Table|Configuration|LJ|Termines|Succes|Meilleur f|Mediane s|Mediane evals f|',
          '|---:|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v): return 'n/d' if v is None else f'{v:.6g}'
    for a in aggregates:
        text.append(f"|{a['table']}|{a['configuration']}|{a['atoms']}|{a['completed']}/{a['expected']}|{a['success']}/{a['expected']}|{fmt(a['best_objective'])}|{fmt(a['median_seconds'])}|{fmt(a['median_evaluations'])}|")
    tmp=report/'SUMMARY.md.tmp'; tmp.write_text('\n'.join(text)+'\n',encoding='utf-8')
    os.replace(tmp,report/'SUMMARY.md')
    if verbose: print('\n'.join(text))
    return state

def make_bundle(out,plan,allow_partial=False):
    state=summary(out,plan)
    for task in plan['tasks']: done_result(out,task,verify=True)
    if not state['complete'] and not allow_partial:
        raise RuntimeError('Campagne incomplete : utiliser bundle --allow-partial pour une archive explicitement PARTIELLE.')
    dest=out/'bundles'; dest.mkdir(exist_ok=True)
    suffix='COMPLET' if state['complete'] else 'PARTIEL'
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=dest/f'PF131_ABLATION_SENSIBILITE_LJ75_LJ102_REFINE_ALL_{suffix}_{stamp}.tar.gz'; temp=path.with_name(path.name+'.partial')
    files=sorted(p for p in out.rglob('*') if p.is_file() and dest not in p.parents and p.name!='.study.lock')
    prefix='PF131_ABLATION_SENSIBILITE_LJ75_LJ102_REFINE_ALL'
    manifest={}
    log(f'ARCHIVAGE {suffix} : {len(files)} fichiers')
    try:
        with tarfile.open(temp,'w:gz',compresslevel=3) as tar:
            for p in files:
                if p.is_symlink(): raise RuntimeError(f'Lien symbolique non archive : {p}')
                before=p.stat(); name=prefix+'/'+p.relative_to(out).as_posix()
                h=hashlib.sha256()
                class Reader:
                    def __init__(self,f): self.f=f
                    def read(self,n=-1):
                        b=self.f.read(n); h.update(b); return b
                with p.open('rb') as f:
                    info=tar.gettarinfo(str(p),arcname=name); info.uid=info.gid=0; info.uname=info.gname=''
                    tar.addfile(info,Reader(f))
                after=p.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError(f'Fichier change pendant archivage : {p}')
                manifest[name]=dict(sha256=h.hexdigest(),bytes=after.st_size)
            raw=canonical(dict(schema=CAMPAIGN_SCHEMA,solver_interface=SCHEMA,complete=state['complete'],status=state,files=manifest))
            info=tarfile.TarInfo(prefix+'/ARCHIVE_MANIFEST.json'); info.size=len(raw); tar.addfile(info,io.BytesIO(raw))
        seen=set()
        with tarfile.open(temp,'r|gz') as tar:
            for member in tar:
                if member.name.endswith('/ARCHIVE_MANIFEST.json'): continue
                if member.name not in manifest or member.name in seen: raise RuntimeError('Membre archive inattendu.')
                f=tar.extractfile(member); h=hashlib.sha256()
                for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
                if h.hexdigest()!=manifest[member.name]['sha256']: raise RuntimeError('SHA interne incorrect.')
                seen.add(member.name)
        if seen!=set(manifest): raise RuntimeError('Archive incomplete.')
        os.replace(temp,path); value=sha(path)
        path.with_name(path.name+'.sha256').write_text(f'{value}  {path.name}\n')
        log(f'BUNDLE_UNIQUE={path}'); print(f'SHA256={value}\nINTEGRITE_BUNDLE=OK\nBUNDLE_COMPLET={state["complete"]}',flush=True)
        return path
    except BaseException:
        temp.unlink(missing_ok=True); raise

def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['plan','run','status','bundle'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--binary',type=Path,default=KIT/'build'/'pf_solver')
    ap.add_argument('--ablation-runs',type=int,default=30)
    ap.add_argument('--sensitivity-runs',type=int,default=10)
    ap.add_argument('--base-seed',type=int,default=2024)
    ap.add_argument('--max-iter',type=int,default=1000)
    ap.add_argument('--max-time',type=float,default=36000)
    ap.add_argument('--max-evals',type=int,default=0)
    ap.add_argument('--threads',type=int,default=10)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--fixed-beta',type=float,default=1e-4,help='Convention non specifiee par Table 7 : beta constant pour -annealing')
    ap.add_argument('--max-tasks',type=int,default=0,help='0=all; otherwise stop after this many new tasks, then resume same OUT')
    ap.add_argument('--allow-partial',action='store_true')
    return ap

def main(argv=None):
    a=parser().parse_args(argv); out=a.out.resolve()
    if a.mode=='status':
        if not (out/'PLAN.json').exists(): raise RuntimeError('PLAN.json absent.')
        report=out/'reports'/'SUMMARY.md'
        if not report.exists(): raise RuntimeError('SUMMARY.md pas encore disponible.')
        print(report.read_text(encoding='utf-8')); return 0
    with lock(out):
        if a.mode=='bundle':
            make_bundle(out,load(out/'PLAN.json'),a.allow_partial); return 0
        p={k:getattr(a,k) for k in ('ablation_runs','sensitivity_runs','base_seed','max_iter','max_time','max_evals','threads','workers','fixed_beta')}
        if not (1<=p['sensitivity_runs']<=p['ablation_runs']<=100000): raise RuntimeError('1 <= sensibilite <= ablation <= 100000 requis.')
        if any(p[k]<1 for k in ('max_iter','threads','workers')) or not math.isfinite(p['max_time']) or p['max_time']<=0 or p['max_evals']<0 or a.max_tasks<0:
            raise RuntimeError('Budget/concurrence invalide.')
        if not 0<=p['base_seed']<2**64 or not 0<p['fixed_beta']<=5: raise RuntimeError('Seed/beta invalides.')
        plan,binary=initialize(out,p,a.binary.resolve())
        log(f'OUT={out} ATOMS=75,102 REFINE_LOOP=ALL FINAL_REFINE=TOP5_NATIVE CALCULS_UNIQUES={plan["unique_solves"]} LIGNES_TABLES={plan["logical_rows"]}')
        if a.mode=='plan':
            summary(out,plan,True); return 0
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        launched=0
        try:
            for t in plan['tasks']:
                if run_task(out,binary,p,t):
                    launched+=1; summary(out,plan)
                    if a.max_tasks and launched>=a.max_tasks: break
        finally:
            summary(out,plan)
        state=summary(out,plan,True)
        if state['complete']: make_bundle(out,plan)
        else: log('Campagne partielle conservee. Reprendre avec le meme OUT et les memes parametres.')
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: print('INTERRUPTION : pieces preservees; reprise possible.',file=sys.stderr); sys.exit(130)
    except Exception as e: print(f'ERREUR : {e}',file=sys.stderr); sys.exit(2)
