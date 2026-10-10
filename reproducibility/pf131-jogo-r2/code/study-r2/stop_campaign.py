#!/usr/bin/env python3
"""Stop only the matching PF131 study runner and its children. No global pkill.
Default is --check. --stop requests SIGTERM and lets the R1 handler save the
interrupted attempt and terminate its solver process group. Completed runs stay.
"""
from __future__ import annotations
import argparse, datetime, json, os, signal, sys, time
from pathlib import Path


def proc_info(pid):
    p=Path('/proc')/str(pid)
    try:
        if p.stat().st_uid != os.geteuid(): return None
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0] in ('Z','X'): return None
        args=[os.fsdecode(a) for a in (p/'cmdline').read_bytes().split(b'\0') if a]
        return dict(pid=int(pid),ppid=int(stat[1]),start_ticks=int(stat[19]),args=args)
    except (OSError,ValueError,IndexError): return None


def resolve_arg(pid,value):
    p=Path(value)
    if not p.is_absolute():
        try: p=Path(os.readlink(f'/proc/{pid}/cwd'))/p
        except OSError: return None
    return str(p.resolve())


def role(info,out):
    args=info['args'];pid=info['pid']
    if not args: return None
    # Exact study.py + run + --out, not a substring in a log or shell command.
    idx=next((i for i,a in enumerate(args) if Path(a).name=='study.py'),None)
    if idx is not None and len(args)>idx+1 and args[idx+1]=='run':
        dest=None
        for i,a in enumerate(args[idx+2:],idx+2):
            if a=='--out' and i+1<len(args): dest=args[i+1]
            elif a.startswith('--out='): dest=a.split('=',1)[1]
        if dest is not None and resolve_arg(pid,dest)==str(out): return 'runner'
    name=Path(args[0]).name
    if name in ('pf_solver','pf_local_worker'):
        if resolve_arg(pid,args[0])==str((out/'provenance'/'bin'/name).resolve()): return name
    return None


def scan(out):
    infos={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            info=proc_info(int(p.name))
            if info: infos[info['pid']]=info
    selected={pid:dict(info,role=role(info,out)) for pid,info in infos.items() if role(info,out)}
    # Track children already created, even if they exec or are later reparented.
    changed=True
    while changed:
        changed=False
        for pid,info in infos.items():
            if pid not in selected and info['ppid'] in selected:
                selected[pid]=dict(info,role='descendant'); changed=True
    return selected


def alive(info):
    current=proc_info(info['pid'])
    return current is not None and current['start_ticks']==info['start_ticks']


def terminate(info):
    if not alive(info): return False
    if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
        try: fd=os.pidfd_open(info['pid'])
        except ProcessLookupError: return False
        try:
            if not alive(info): return False
            signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally: os.close(fd)
    else:
        if not alive(info): return False
        try: os.kill(info['pid'],signal.SIGTERM)
        except ProcessLookupError: return False
    print(f"SIGTERM pid={info['pid']} role={info['role']}",flush=True)
    return True


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    g=ap.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--stop',action='store_true')
    ap.add_argument('--timeout',type=float,default=45)
    a=ap.parse_args(argv);out=a.out.resolve()
    if a.timeout<1 or a.timeout>300: raise RuntimeError('timeout entre 1 et 300 secondes requis')
    if len(out.parts)<3: raise RuntimeError('Chemin de campagne trop general')
    selected=scan(out)
    print(f'ANCIENNE_CAMPAGNE={out}',flush=True)
    for i in selected.values(): print(f"ACTIF pid={i['pid']} role={i['role']} {' '.join(i['args'])}",flush=True)
    if not selected:
        print('ANCIENNE_CAMPAGNE_ARRETEE=OUI; aucun processus correspondant visible',flush=True);return 0
    if not a.stop:
        print('ANCIENNE_CAMPAGNE_ENCORE_ACTIVE=OUI; ne pas lancer la nouvelle',flush=True);return 2
    if not (out/'PLAN.json').is_file(): raise RuntimeError('PLAN.json absent : refus d\u2019envoyer des signaux')
    # The runner must be stopped first: killing only its solver could allow a next task.
    roots=[i for i in selected.values() if i['role']=='runner']
    to_signal=roots or [i for i in selected.values() if i['role'] in ('pf_solver','pf_local_worker')]
    sent=[]
    for info in to_signal:
        if terminate(info): sent.append(dict(pid=info['pid'],role=info['role'],start_ticks=info['start_ticks']))
    deadline=time.monotonic()+a.timeout
    while time.monotonic()<deadline:
        time.sleep(.2)
        current=scan(out);selected.update(current)
        remaining={pid:i for pid,i in selected.items() if alive(i)}
        if not remaining: break
        # If the runner is gone, terminate only surviving matching descendants.
        if not any(i['role']=='runner' and alive(i) for i in selected.values()):
            for info in remaining.values():
                if not any(x['pid']==info['pid'] and x['start_ticks']==info['start_ticks'] for x in sent):
                    if terminate(info):sent.append(dict(pid=info['pid'],role=info['role'],start_ticks=info['start_ticks']))
    remaining={pid:i for pid,i in selected.items() if alive(i)}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record=dict(requested_utc=stamp,out=str(out),reason='User requested LJ75/LJ102 with refine-all in a separate campaign',
                signals=sent,remaining=list(remaining.values()),no_results_deleted=True)
    with (out/f'ARRET_POUR_REFINE_ALL_{stamp}.json').open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2,ensure_ascii=False);f.write('\n')
    if remaining:
        print('ARRET_NON_CONFIRME; processus restants : '+','.join(map(str,remaining)),flush=True);return 2
    print('ANCIENNE_CAMPAGNE_ARRETEE=OUI; resultats et tentatives conserves',flush=True);return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as e: print(f'ERREUR_ARRET : {e}',file=sys.stderr);sys.exit(2)
