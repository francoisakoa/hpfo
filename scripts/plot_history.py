#!/usr/bin/env python3
"""Plot a saved PF CSV: diagnostic plots, not replacements for submitted figures."""
import argparse,csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Choose a new output directory')
    with a.csv.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise SystemExit('Empty trace')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.output.mkdir(parents=True)
    for x,y,name in (('iteration','best','best_by_iteration'),('objective_evaluations','best','best_by_evaluations'),('seconds','best','best_by_time'),('iteration','ess','ess_by_iteration')):
        fig,ax=plt.subplots(figsize=(7,4));ax.plot([float(r[x]) for r in rows],[float(r[y]) for r in rows]);ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(True,alpha=.3);fig.tight_layout();fig.savefig(a.output/(name+'.png'),dpi=180);plt.close(fig)
    print('Diagnostic plots written. No original manuscript figure identity is claimed.')
if __name__=='__main__':main()
