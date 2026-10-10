#!/usr/bin/env python3
"""Print a replay command; only --execute launches an experiment into a NEW directory.
Main PF131 has no --run-start: main replay runs 1..K, including Ipopt multistart.
Study replay reuses the saved command and frozen population of the chosen task.
"""
from __future__ import annotations
import argparse,os,shlex,subprocess,sys
from pathlib import Path
from hpfo_common import *

def built_solver(root,variant):
    j=load(root/'validation/build/LATEST.json')
    require(j['status']=='PASS','Build/test the two source snapshots first')
    r=next(r for r in j['builds'] if r['variant']==variant)
    path=root/r['build_dir']/'pf_solver';require(path.is_file(),'Local rebuilt binary unavailable: '+str(path))
    return path

def replace_flag(cmd,flag,value,required=True):
    require(cmd.count(flag)<=1,'Duplicate command option '+flag)
    if flag in cmd:
        i=cmd.index(flag);require(i+1<len(cmd),'Missing option value');cmd[i+1]=str(value)
    elif required:raise RuntimeError('Recorded command lacks '+flag)
    return cmd

VALUES={'particles':'--particles','min_particles':'--min-particles','max_iterations':'--max-iter',
        'refine_top_k':'--refine-top-k','radius':'--radius','batch_threads':'--threads','local_workers':'--workers',
        'local_method':'--local-refine','flow':'--flow','mapping':'--mapping','clustering':'--clustering',
        'mutation_sigma':'--mutation-sigma','beta_initial':'--beta-initial','beta_max':'--beta-max',
        'lambda_share':'--lambda-share','lambda_novelty':'--lambda-novelty','target':'--target',
        'target_tolerance':'--target-tol','max_seconds':'--max-time','max_objective_evaluations':'--max-evals',
        'no_improve_stop':'--no-improve-stop','gmm_max_iterations':'--gmm-max-iter',
        'local_max_iterations':'--local-max-iter','local_tolerance':'--local-tol',
        'newton_linear_solver':'--newton-linear-solver','newton_linear_ordering':'--newton-linear-ordering',
        'newton_cg_max_iterations':'--newton-cg-max-iter','newton_initial_radius':'--newton-radius',
        'ipopt_linear_solver':'--ipopt-linear-solver','ipopt_print_level':'--ipopt-print-level',
        'kmeans_restarts':'--kmeans-restarts','cluster_max_iterations':'--cluster-max-iter',
        'dbscan_epsilon':'--dbscan-eps','dbscan_min_samples':'--dbscan-min-samples'}
FALSE_FLAGS={'final_refinement':'--no-final-refine','kicks':'--no-kicks','islands':'--no-islands',
             'diversity_injection':'--no-diversity-injection','ipopt_ball_constraints':'--no-ipopt-ball',
             'ipopt_exact_hessian':'--ipopt-limited-memory'}

def find_command(obj):
    found=[];todo=[obj]
    while todo:
        x=todo.pop()
        if isinstance(x,dict):
            for k,v in x.items():
                if k.lower() in ('command','argv','command_line','command_shell','cmd'):
                    a=shlex.split(v) if isinstance(v,str) else v
                    if isinstance(a,list) and all(isinstance(t,str) for t in a) and any(Path(t).name=='run_one_baseline.py' for t in a):
                        found.append(a)
                if isinstance(v,(dict,list)):todo.append(v)
        elif isinstance(x,list):todo.extend(v for v in x if isinstance(v,(dict,list)))
    unique={tuple(x) for x in found}
    require(len(unique)==1,'No unique original baseline command in this JSON. Use the archived driver and configuration; an exact command is not invented.')
    return list(next(iter(unique)))

def remap_source(old,source_root):
    old=Path(old);candidates=list(source_root.rglob(old.name))
    require(candidates,'Archived source/config not found for '+str(old))
    ranked=[]
    for p in candidates:
        a,b=list(reversed(p.parts)),list(reversed(old.parts));score=0
        for x,y in zip(a,b):
            if x!=y:break
            score+=1
        ranked.append((score,p))
    best=max(s for s,p in ranked);bestpaths=[p for s,p in ranked if s==best]
    require(len(bestpaths)==1,'Ambiguous archived source version for '+str(old))
    return bestpaths[0]

def command(root,a):
    data=a.data.resolve();out=a.output.absolute()
    require(not out.exists() and not out.is_symlink(),'Replay output exists: '+str(out))
    # Never write below the archived data, code or prepared evidence.
    for protected in (data,root/SUBDIR,root/'release-assets',root/'validation'):
        require(out!=protected and protected not in out.parents,'Output overlaps protected material')
    if a.mode=='main':
        require(1<=a.runs<=30,'main runs must be 1..30')
        folder=data/'main'/f'LJ{a.atoms}/mumps/LJ{a.atoms}_run1'
        j=load(folder/'pf.json');o=j['options'];init=load(folder/'initialization.json')
        require(o['base_seed']==init['base_seed'],'Base seed mismatch')
        pop=folder/'initial.pfpop';popsha=sha(pop)
        varying={'seed','seed_init','seed_pf'}
        for r in range(1,a.runs+1):
            f=data/'main'/f'LJ{a.atoms}/mumps/LJ{a.atoms}_run{r}'
            require(sha(f/'initial.pfpop')==popsha,'Main run populations differ; not silently pooled')
            oo=load(f/'pf.json')['options']
            require({k:v for k,v in oo.items() if k not in varying}=={k:v for k,v in o.items() if k not in varying},'Main per-run parameters differ')
        binary=built_solver(root,'main')
        cmd=[str(binary),'--atoms',str(a.atoms),'--runs',str(a.runs),'--seed',str(o['base_seed']),
             '--population',str(pop),'--worker-executable',str(binary.with_name('pf_local_worker'))]
        for k,flag in VALUES.items():require(k in o,'Missing archived option '+k);cmd += [flag,str(o[k])]
        for k,flag in FALSE_FLAGS.items():
            if not o[k]:cmd.append(flag)
        if o['refine_all']:cmd.append('--refine-all')
        if o['covariance_mutation']:cmd.append('--cma')
        require(not o.get('hsl_library'),'Historical HSL path cannot be reused silently')
        cmd += ['--include-ipopt','--output-dir',str(out)]
    elif a.mode=='study':
        require(a.configuration,'--configuration is required')
        row=next((r for r in load(root/SUBDIR/'provenance/RUN_INDEX.json') if r['dataset']=='study' and r['atoms']==a.atoms and r['run']==a.run and r['configuration']==a.configuration),None)
        require(row,'Study task absent from recorded index');folder=(data/row['file']).parent
        meta=load(folder.parent.parent/'EXECUTION.json');cmd=meta['command'][:]
        require(isinstance(cmd,list) and all(isinstance(s,str) for s in cmd),'Invalid recorded command')
        binary=built_solver(root,'study');cmd[0]=str(binary)
        replace_flag(cmd,'--worker-executable',binary.with_name('pf_local_worker'))
        replace_flag(cmd,'--population',folder/'initial.pfpop');replace_flag(cmd,'--output-dir',out)
    else:
        require(a.method in METHODS,'--method required for a baseline')
        r=next((r for r in load(root/SUBDIR/'provenance/BASELINES_RUN_INDEX.json') if r['method']==a.method and r['atoms']==a.atoms and r['run']==a.run),None)
        require(r,'Baseline result not found');cmd=find_command(load(data/r['file']))
        k=next(i for i,x in enumerate(cmd) if Path(x).name=='run_one_baseline.py')
        require(k in (0,1,2,3),'Unexpected baseline launcher; inspect original command')
        src=root/SUBDIR/'code/baselines-export';driver=remap_source(cmd[k],src)
        if k:
            require(Path(cmd[0]).name.startswith('python'), 'Unsupported baseline interpreter prefix')
            require(all(x in ('-u','-B','-O','-OO','-E','-I') for x in cmd[1:k]), 'Unsupported Python launcher options')
        cmd=[sys.executable]+cmd[1:k]+[str(driver)]+cmd[k+1:] if k else [sys.executable,str(driver)]+cmd[1:]
        replace_flag(cmd,'--outdir',out)
        if '--eval-config' in cmd:replace_flag(cmd,'--eval-config',remap_source(cmd[cmd.index('--eval-config')+1],src))
    require(not any(x in cmd for x in (';','&&','|','>','>>','<')),'Shell operators refused')
    return cmd

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('main','study','baseline'))
    p.add_argument('--atoms',type=int,choices=ATOMS,required=True);p.add_argument('--run',type=int,default=1)
    p.add_argument('--runs',type=int,default=30);p.add_argument('--configuration');p.add_argument('--method',choices=METHODS)
    p.add_argument('--data',type=Path,default=root/'data');p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    try:
        check_package(root,False);cmd=command(root,a);print(shlex.join(cmd),flush=True)
        if not a.execute:print('DRY_RUN=YES ; no optimization launched')
        else:
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            a.output.parent.mkdir(parents=True,exist_ok=True)
            sys.exit(subprocess.call(cmd,env=env))
    except Exception as e:sys.exit('REPLAY_ERROR: '+str(e))
