#!/usr/bin/env python3
"""Validate raw results and export summaries; never silently discard bad runs.

Default: exactly 8 sizes x 5 methods x 30 unique runs, no invalid/censored run.
--preview permits an incomplete, explicitly labelled diagnostic summary.
--allow-censored permits recorded time/restart limits, counted separately.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from baselines import METHODS
from protocol import ATOMS,digest,sha_file
from record_schema import validate_record,CENSORED,INVALID
from run_one_baseline import atomic_json


def inspect_results(root,atoms,methods,runs,*,verify_energy=True):
    root=Path(root);expected={(n,m,r) for n in atoms for m in methods for r in range(1,runs+1)}
    records={};issues=[];all_files=[];duplicates=[]
    for path in sorted(root.glob('LJ*/*/run*.json')):
        all_files.append(str(path.relative_to(root)))
        try:
            r=json.loads(path.read_text())
            key=(r['n_atoms'],r['method'],r['run_id'])
            if type(key[0]) is not int or not isinstance(key[1],str) or type(key[2]) is not int:
                raise ValueError('malformed run identity types')
            errors=validate_record(r,verify_energy=verify_energy)
            if key in records:duplicates.append(key);errors.append('duplicate method/instance/run identifier')
            if key not in expected:errors.append('unexpected method/instance/run identifier')
            if errors:issues.append(dict(file=str(path.relative_to(root)),errors=errors))
            if key not in records:records[key]=r
        except Exception as exc:
            issues.append(dict(file=str(path.relative_to(root)),errors=[f'unreadable result: {exc}']))
    missing=sorted(expected-set(records));unexpected=sorted(set(records)-expected)
    if missing:issues.append(dict(errors=[f'{len(missing)} missing runs'],missing=missing))
    if unexpected:issues.append(dict(errors=['unexpected runs'],unexpected=unexpected))
    # Compare all protocol fields that should be identical across a campaign.
    global_fields=('version','base_seed','potential','backend','bounds','initialization','remove_com',
                   'target_tolerance','chunk_size','restart_seconds','job_seconds','max_attempts',
                   'code_sha256','environment','counting','budget_config_sha256')
    baseline=None;cell_refs={}
    for key,r in records.items():
        cfg=r.get('identity',{})
        if not isinstance(cfg,dict):cfg={}
        global_cfg={k:cfg.get(k) for k in global_fields}
        if baseline is None:baseline=global_cfg
        elif global_cfg!=baseline:issues.append(dict(run=key,errors=['mixed campaign protocol/code/environment']))
        cell=key[:2];cell_cfg={k:cfg.get(k) for k in ('budget','target_energy','method_params','oracle')}
        if cell not in cell_refs:cell_refs[cell]=cell_cfg
        elif cell_refs[cell]!=cell_cfg:issues.append(dict(run=key,errors=['mixed parameters/budgets inside a cell']))
    return records,dict(expected_count=len(expected),observed_files=len(all_files),unique_runs=len(records),
                        missing=missing,unexpected=unexpected,duplicates=duplicates,issues=issues,
                        protocol=baseline,raw_files=all_files,results_root=str(root.resolve()),
                        raw_file_sha256={p:sha_file(root/p) for p in all_files})


def summarize(records,atoms,methods,runs):
    rows=[]
    for n in atoms:
        for method in methods:
            rs=[r for (a,m,_),r in sorted(records.items()) if a==n and m==method]
            finite=[float(r['energy']) for r in rs if type(r.get('energy')) in (float,int) and math.isfinite(r['energy'])]
            times=[float(r['wall_time']) for r in rs if isinstance(r.get('wall_time'),(int,float)) and math.isfinite(r['wall_time'])]
            evals=[r['n_func_evals'] for r in rs if type(r.get('n_func_evals')) is int]
            def mean(v):return float(np.mean(v)) if v else None
            def sd(v):return float(np.std(v,ddof=1)) if len(v)>1 else None
            reasons=Counter(str(r.get('stop_reason','missing')) for r in rs)
            success=sum(bool(r.get('success',False)) for r in rs)
            row=dict(n_atoms=n,method=method,runs=len(rs),expected_runs=runs,successes=success,
                     success_rate_pct=100.*success/len(rs) if rs else None,finite_energy_count=len(finite),
                     best_energy=min(finite) if finite else None,mean_energy=mean(finite),std_energy=sd(finite),
                     mean_wall_time=mean(times),std_wall_time=sd(times),median_wall_time=float(np.median(times)) if times else None,
                     mean_evaluations=mean(evals),median_evaluations=float(np.median(evals)) if evals else None,
                     maximum_evaluations=max(evals) if evals else None,
                     budget=rs[0].get('max_evals') if rs else None,
                     target_energy=rs[0].get('target_energy') if rs else None,
                     target_tolerance=rs[0].get('target_tolerance') if rs else None,
                     evaluation_limit_runs=reasons.get('evaluation_limit',0),
                     time_limit_runs=reasons.get('time_limit',0),
                     censored_runs=sum(reasons[k] for k in CENSORED),
                     invalid_runs=sum(reasons[k] for k in INVALID),
                     runs_with_errors=sum(bool(r.get('errors')) for r in rs),
                     gradient_calls_total=sum(r.get('counters',{}).get('gradient_calls',0) for r in rs if isinstance(r.get('counters'),dict) and type(r['counters'].get('gradient_calls')) is int),
                     audit_evaluations_total=sum(r.get('counters',{}).get('audit_objective_evaluations',0) for r in rs if isinstance(r.get('counters'),dict) and type(r['counters'].get('audit_objective_evaluations')) is int),
                     stop_reasons=dict(reasons))
            rows.append(row)
    return rows


def aggregate(root,output,atoms=ATOMS,methods=METHODS,runs=30,preview=False,allow_censored=False):
    if not atoms or not methods or runs<1:raise ValueError('Empty/invalid campaign selection')
    if len(set(atoms))!=len(atoms) or len(set(methods))!=len(methods):raise ValueError('Duplicate requested selections')
    records,report=inspect_results(root,atoms,methods,runs)
    rows=summarize(records,atoms,methods,runs)
    invalid=sum(row['invalid_runs']+row['runs_with_errors'] for row in rows)
    censored=sum(row['censored_runs'] for row in rows)
    ready=not report['issues'] and invalid==0 and (censored==0 or allow_censored)
    report.update(invalid_or_error_count=invalid,censored_count=censored,censored_accepted=allow_censored,
                  ready_for_export=ready,preview=preview)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'validation.json',report)
    if not ready and not preview:
        print(f'BLOCKED: {len(report["issues"])} validation issues, {invalid} invalid/error flags, {censored} censored runs. See {out/"validation.json"}',file=sys.stderr)
        return 2
    status='PREVIEW_NOT_FOR_MANUSCRIPT' if preview else ('VALIDATED_WITH_CENSORING' if censored else 'VALIDATED')
    payload=dict(schema='jogo-baseline-summary-v3',status=status,atoms=list(atoms),methods=list(methods),runs_per_cell=runs,
                 convention='all runs; sample SD ddof=1; separate stop reasons; no imputation',
                 protocol=report['protocol'],validation=report,rows=rows)
    payload['summary_sha256']=digest(payload)
    stem='summary_preview' if preview else 'summary'
    atomic_json(out/(stem+'.json'),payload)
    keys=[k for k in rows[0] if k!='stop_reasons']
    with (out/(stem+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:row[k] for k in keys})
    print(f'{status}: {len(records)} unique results -> {out/stem}')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir','--outdir',dest='results_dir',default='results_v3')
    p.add_argument('--output',default=None)
    p.add_argument('--atoms',type=int,nargs='+',default=ATOMS)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--runs',type=int,default=30)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--allow-censored',action='store_true')
    a=p.parse_args()
    try:return aggregate(a.results_dir,a.output or str(Path(a.results_dir)/'_summary'),a.atoms,a.methods,a.runs,a.preview,a.allow_censored)
    except Exception as e:print(f'ERROR: {e}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
