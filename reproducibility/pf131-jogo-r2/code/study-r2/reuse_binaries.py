#!/usr/bin/env python3
"""Reuse the frozen study-R1 executables after source and binary checks. No build."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
KIT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def inventory(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}

def verify_sources(kit,old):
    root=old/'provenance'/'kit'/'source'
    if not root.is_dir(): raise RuntimeError(f'Source figee absente : {root}')
    expected=inventory(kit/'source');got=inventory(root)
    if not expected or expected!=got:
        diff=sorted(k for k in set(expected)|set(got) if expected.get(k)!=got.get(k))
        raise RuntimeError('Sources differentes : aucune reutilisation binaire. Fichiers : '+', '.join(diff[:15]))
    manifest=json.loads((old/'provenance'/'SOURCE_SHA256.json').read_text())
    for name,h in expected.items():
        if manifest.get('source/'+name)!=h: raise RuntimeError('Source non conforme au manifeste R1 : '+name)
    return expected

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--previous-out',type=Path,required=True)
    a=ap.parse_args(argv);old=a.previous_out.resolve()
    hashes=verify_sources(KIT,old)
    meta=json.loads((old/'provenance'/'BINARY.json').read_text());identity=meta['identity']
    files=[('pf_solver','solver_sha256'),('pf_local_worker','worker_sha256')]
    for name,key in files:
        src=old/'provenance'/'bin'/name
        if not src.is_file() or sha(src)!=identity[key]: raise RuntimeError('Binaire R1 absent ou modifie : '+str(src))
        if not os.access(src,os.X_OK): raise RuntimeError('Binaire non executable : '+str(src))
    # Existing destination must agree; never overwrite a different executable.
    build=KIT/'build';build.mkdir(exist_ok=True)
    for name,key in files:
        dest=build/name
        if dest.exists() and sha(dest)!=identity[key]: raise RuntimeError('Binaire different deja present : '+str(dest))
    for name,key in files:
        src=old/'provenance'/'bin'/name;dest=build/name
        if not dest.exists(): shutil.copy2(src,dest)
        if sha(dest)!=identity[key]: raise RuntimeError('Copie incorrecte : '+str(dest))
    schema=subprocess.check_output([str(build/'pf_solver'),'--study-schema'],text=True,timeout=20).strip()
    version=subprocess.check_output([str(build/'pf_solver'),'--version'],text=True,timeout=20).strip()
    if schema!='PF131_ARTICLE_STUDY_R1' or 'ipopt=1' not in version: raise RuntimeError('Interface/Ipopt incompatible')
    ref=old/'provenance'/'build'
    if ref.is_dir() and not (build/'reference_build').exists(): shutil.copytree(ref,build/'reference_build')
    info=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_out=str(old),
              binary_identity=identity,source_files_identical=len(hashes),cpp_changed=False,
              recompiled=False,version=version,solver_schema=schema,
              note='Commands --version and --study-schema only. No optimization run.')
    (build/'REUSE_BINARY.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    print('SOURCE_IDENTIQUE=OUI FICHIERS='+str(len(hashes)))
    print('BINAIRE_REUTILISE='+str(build/'pf_solver'))
    print('VERSION='+version+'\nRECOMPILATION=NON\nPREPARATION_OK=OUI')
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print(f'ERREUR_PREPARATION : {e}',file=sys.stderr);sys.exit(2)
