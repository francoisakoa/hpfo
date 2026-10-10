#!/usr/bin/env python3
"""Generate baseline-only LaTeX tables from a validated v3 summary.

No old HPF values are inserted. Add the matched HPF column separately after
its campaign and counting provenance have been checked.
"""
import argparse
import json
from pathlib import Path
import sys
from protocol import digest,sha_file


def export(summary,output):
    path=Path(summary);s=json.loads(path.read_text());unsigned=dict(s);h=unsigned.pop('summary_sha256',None)
    if h!=digest(unsigned):raise ValueError('Summary checksum mismatch')
    if s.get('status') not in ('VALIDATED','VALIDATED_WITH_CENSORING'):raise ValueError('Only validated summaries can generate manuscript tables')
    report_path=path.parent/'validation.json'
    if not report_path.exists():raise ValueError('Missing validation report')
    if json.loads(report_path.read_text())!=s['validation'] or not s['validation']['ready_for_export']:
        raise ValueError('Latest validation report is not the one authorising this export')
    raw_root=Path(s['validation']['results_root'])
    for relative,expected_hash in s['validation']['raw_file_sha256'].items():
        if not (raw_root/relative).is_file() or sha_file(raw_root/relative)!=expected_hash:
            raise ValueError('Raw results changed/moved since aggregation; aggregate again before exporting tables')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows={(r['n_atoms'],r['method']):r for r in s['rows']};methods=s['methods'];atoms=s['atoms']
    def val(x,dec=6):return '--' if x is None else f'{x:.{dec}f}'
    prefix=(f'% Automatically generated; SHA256 {h}\n'
            '% Specific implementations under the recorded protocol; NOT algorithm-family superiority.\n'
            '% Energies and times include ALL runs. SD uses ddof=1. Audit outside search budget.\n'
            f'% Status: {s["status"]}; model={s["protocol"]["potential"]}; backend={s["protocol"]["backend"]}.\n')
    def make(name,heading,cell):
        txt=prefix+'\\begin{tabular}{r'+'r'*len(methods)+'}\n\\hline\n'
        txt+='N & '+' & '.join(methods)+' \\\\\n\\hline\n'
        for n in atoms:txt+=str(n)+' & '+' & '.join(cell(rows[n,m]) for m in methods)+' \\\\\n'
        txt+='\\hline\n\\end{tabular}\n'
        (out/name).write_text('% '+heading+'\n'+txt)
    make('table_baseline_success_best.tex','Successes/runs; best energy',lambda r:f'{r["successes"]}/{r["runs"]} / {val(r["best_energy"])}')
    make('table_baseline_mean_energy.tex','Mean energy +/- sample SD',lambda r:f'{val(r["mean_energy"])} $\\pm$ {val(r["std_energy"])}')
    make('table_baseline_times.tex','Mean search elapsed seconds +/- sample SD',lambda r:f'{val(r["mean_wall_time"],3)} $\\pm$ {val(r["std_wall_time"],3)}')
    make('table_baseline_censoring.tex','Counts: evaluation limit / time limit / all other censoring',lambda r:f'{r["evaluation_limit_runs"]} / {r["time_limit_runs"]} / {r["censored_runs"]-r["time_limit_runs"]}')
    make('table_baseline_evaluations.tex','Mean consumed objective evaluations / budget',lambda r:f'{val(r["mean_evaluations"],1)} / {r["budget"]}')
    (out/'README_TABLES.txt').write_text('Generated baseline columns only. No HPF data inferred. Preserve model, budget and censoring notes.\nThe timing is implementation/hardware dependent. Tables do not certify global optimality.\n')
    print(f'Tables generated in {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',required=True);p.add_argument('--output',default='tables_v3');a=p.parse_args()
    try:export(a.summary,a.output);return 0
    except Exception as e:print(f'ERROR: {e}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
