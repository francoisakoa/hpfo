#!/usr/bin/env python3
"""Isolated, read-only use of the original solver's schema (one version per process).
No optimizer is run here. No original result is edited or resealed.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--request',required=True)
    a=ap.parse_args();source=Path(a.source).resolve();sys.path.insert(0,str(source))
    from baselines import VERSION
    from protocol import (code_digest, method_params, load_budget, sha_file, COUNTING_CONVENTION,
                          environment, digest)
    from lj_oracle import LJOracle
    from record_schema import validate_record
    request=json.loads(Path(a.request).read_text()); templates={}
    for n in request['atoms']:
        for method in request['methods']:
            oracle=LJOracle(n,'pf131','cpu')
            from protocol import KNOWN_MINIMA
            budget,_=load_budget(source/'baseline_eval_limits_v3.json',n,request.get('technical_max_evals'))
            t=dict(version=VERSION,method=method,n_atoms=n,base_seed=2024,budget=budget,
                   budget_config_sha256=sha_file(source/'baseline_eval_limits_v3.json'),
                   target_energy=KNOWN_MINIMA.get(n),target_tolerance=1e-4,potential='pf131',
                   oracle=oracle.metadata(),backend='cpu',bounds=5.,initialization='box',remove_com=False,
                   chunk_size=32,method_params=method_params(method,n),restart_seconds=0.,
                   job_seconds=float(request.get('job_seconds',36000)),max_attempts=0,
                   code_sha256=code_digest(),
                   seed_protocol='md5-base-N-run-1000-first8; first attempt uses root directly; later attempts MD5(root,N,a,1000)',
                   restart_protocol='distinct-seed restart on algorithm completion or explicit attempt deadline; shared budget/incumbent (also available to BH-OJ jumps)',
                   counting=COUNTING_CONVENTION)
            if method in ('BH-OJ','GA'):
                from local_refinement import POLICY
                t['local_refinement_policy']=POLICY
            templates[f'{n}|{method}']=t
    records=[]
    for s in request.get('files',[]):
        p=Path(s)
        try:
            raw=p.read_bytes();r=json.loads(raw);errors=validate_record(r,verify_energy=True)
            identity=r.get('identity',{})
            key=(r.get('n_atoms'),r.get('method'),r.get('run_id'))
            t=templates.get(f'{key[0]}|{key[1]}')
            expected=dict(t or {});expected['run_id']=key[2]
            actual={k:v for k,v in identity.items() if k!='environment'}
            if t is None:errors.append('instance/method outside selected source version')
            elif actual!=expected:
                fields=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
                errors.append('algorithm/code/budget identity differs in: '+','.join(fields))
            if type(key[2]) is not int or not 1<=key[2]<=request['runs']:errors.append('unexpected run index')
            # Do not silently accept a file under another cell/run path.
            if key[0] is not None and key[1] is not None and type(key[2]) is int:
                if p.name!=f'run{key[2]:02d}.json' or p.parent.name!=key[1] or p.parent.parent.name!=f'LJ{key[0]}':
                    errors.append('path does not match run identity')
            import hashlib
            record=dict(path=str(p.resolve()),sha256=hashlib.sha256(raw).hexdigest(),key=list(key),errors=errors,
                        stop_reason=r.get('stop_reason'),success=r.get('success'),energy=r.get('energy'),
                        evaluations=r.get('n_func_evals'),budget=r.get('max_evals'),seconds=r.get('wall_time'),
                        finished_utc=r.get('finished_utc'),identity_sha256=r.get('identity_sha256'),
                        environment=identity.get('environment'),version=VERSION,
                        code_sha256=identity.get('code_sha256'))
            records.append(record)
        except Exception as exc:
            records.append(dict(path=str(p.resolve()),errors=[f'{type(exc).__name__}: {exc}']))
    print(json.dumps(dict(templates=templates,records=records,current_environment=environment()),allow_nan=False))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'PROBE_ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);raise SystemExit(2)
