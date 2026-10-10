#!/usr/bin/env python3
"""Rebuild run summaries from archived JSONs without executing an optimizer."""
from __future__ import annotations
import argparse, csv, runpy, statistics, sys
from collections import defaultdict
from pathlib import Path
from hpfo_common import *

def csv_save(path,rows):
    require(rows,'Empty output table')
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def aggregate(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,g in sorted(groups.items()):
        row=dict(zip(keys,key));row['runs']=len(g)
        row['successes']=sum(r['success'] for r in g) if all(type(r['success']) is bool for r in g) else ''
        for k in ('seconds','evaluations','objective'):
            v=[r[k] for r in g];row['mean_'+k]=statistics.mean(v);row['median_'+k]=statistics.median(v)
            row['sample_sd_'+k]=statistics.stdev(v) if len(v)>1 else 0.0
        out.append(row)
    return out

def rebuild(root,data,baselines,output):
    root,data,baselines,output=map(Path,(root,data,baselines,output))
    require(not output.exists(),'Choose a NEW output directory: '+str(output))
    # This is the exact table rebuild tool already used in the verified preparation.
    ns=runpy.run_path(str(root/SUBDIR/'tools/rebuild_tables.py'),run_name='hpfo_table_module')
    ns['main'](data,output)
    groups={'main':[],'ipopt_multistart':[]}
    for r in load(root/SUBDIR/'provenance/RUN_INDEX.json'):
        if r['dataset'] not in groups:continue
        j=load(in_root(data,r['file']));o=j['options']
        hit=j['objective']<=o['target']+o['target_tolerance']
        require(hit==j['target_reached'],'Target flag mismatch')
        groups[r['dataset']].append(dict(atoms=r['atoms'],run=r['run'],success=hit,
                seconds=j['seconds'],evaluations=j['objective_evaluations'],objective=j['objective']))
    csv_save(output/'MAIN_SUMMARY.csv',aggregate(groups['main'],('atoms',)))
    csv_save(output/'IPOPT_MULTISTART_SUMMARY.csv',aggregate(groups['ipopt_multistart'],('atoms',)))
    rows=[]
    for r in load(root/SUBDIR/'provenance/BASELINES_RUN_INDEX.json'):
        rel=Path(r['file']).relative_to('baselines-export');p=in_root(baselines,rel.as_posix())
        require(sha(p)==r['sha256'],'Baseline JSON hash mismatch: '+str(p))
        got=baseline_record(load(p));require(got is not None,'Unrecognized baseline record')
        for k in ('method','atoms','run','seconds','evaluations','objective','success'):
            require(got[k]==r[k],'Baseline metadata changed: '+str(p)+' / '+k)
        rows.append({k:r[k] for k in ('method','atoms','run','status','success','success_source','objective','seconds','evaluations','file','sha256')})
    require(len(rows)==1200,'Expected 1200 baseline records')
    csv_save(output/'BASELINES_RUNS.csv',rows);summary=aggregate(rows,('method','atoms'))
    csv_save(output/'BASELINES_SUMMARY.csv',summary)
    lines=['# External baseline summaries','',
           'All completed runs are included in cost means and sample standard deviations.',
           'A blank success count means explicit target-attainment metadata was unavailable; it is not zero.',
           '', '|Method|LJ|Runs|Target hits|Mean seconds|Mean evaluations|', '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['method']}|{r['atoms']}|{r['runs']}|{r['successes']}|{r['mean_seconds']:.3f}|{r['mean_evaluations']:.1f}|")
    (output/'BASELINES_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('TABLES_REBUILT=7,8,9,10,MAIN,IPOPT,BASELINES ; OPTIMIZATION=NOT_RUN')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--data',type=Path,default=root/'data')
    p.add_argument('--baselines',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:rebuild(root,a.data,a.baselines or a.data/'baselines-export',a.output)
    except Exception as e:sys.exit('ERROR: '+str(e))
