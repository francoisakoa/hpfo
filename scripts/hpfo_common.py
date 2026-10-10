"""Small, dependency-free helpers used by the review-package tools."""
from __future__ import annotations
import hashlib, json, math, os, re, shutil, stat, tarfile, zipfile
from pathlib import Path, PurePosixPath

SUBDIR = Path('reproducibility/pf131-jogo-r2')
ATOMS = (38, 75, 76, 77, 98, 102, 103, 104)
METHODS = ('BH-OJ', 'GA', 'PSO', 'CLPSO', 'SMC')
IGNORED = {'.git', '.local', 'data', 'release-assets', '_build', '__pycache__'}

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def safe_name(name):
    require(isinstance(name,str) and name and '\\' not in name and '\x00' not in name,
            'Invalid archive/manifest name')
    require(not any(ord(c)<32 for c in name), 'Control character in path')
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and not re.match(r'^[A-Za-z]:',name),
            'Unsafe archive/manifest path: '+name)
    require(p.parts and p.parts[0] != '.git','Refusing .git content')
    return p

def in_root(root, name):
    p=Path(root)
    require(not p.is_symlink(),'Symlink root refused: '+str(p))
    for part in safe_name(name).parts:
        p=p/part;require(not p.is_symlink(),'Symlink refused: '+str(p))
    return p

def snapshot_copy(src,dst):
    src,dst=Path(src),Path(dst)
    require(src.is_file() and not src.is_symlink(),'Missing/nonregular source: '+str(src))
    require(not dst.exists() and not dst.is_symlink(),'Destination exists: '+str(dst))
    a=src.stat();dst.parent.mkdir(parents=True,exist_ok=True)
    # A real copy, never a hard link to mutable research data.
    shutil.copyfile(src,dst)
    b=src.stat()
    require((a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_size,b.st_mtime_ns,b.st_ctime_ns),
            'Source changed during copy: '+str(src))
    require(sha(src)==sha(dst),'Copy hash mismatch: '+str(src))
    dst.chmod(0o755 if a.st_mode&0o111 else 0o644)

def manifest_read(path):
    rows=[];seen=set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        m=re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)',line)
        require(m is not None,'Unsupported SHA256 manifest line in '+str(path))
        digest,name=m.groups();name=safe_name(name).as_posix()
        require(name not in seen,'Duplicate manifest name: '+name);seen.add(name)
        rows.append((digest.lower(),name))
    require(rows,'Empty SHA256 manifest: '+str(path))
    return rows

def check_manifest(path,root=None):
    root=Path(root) if root else Path(path).parent
    rows=manifest_read(path)
    for digest,name in rows:
        p=in_root(root,name)
        require(p.is_file() and sha(p)==digest,'SHA256 mismatch/missing: '+str(p))
    return rows

def write_or_verify_stream(stream, path, reuse_identical=False):
    """Read the complete archive stream, including CRC checks, before reuse."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        require(reuse_identical and path.is_file() and not path.is_symlink(),
                'Archive destination collision: ' + str(path))
        expected = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            expected.update(block)
        require(sha(path) == expected.hexdigest(),
                'Existing extracted file differs from archive: ' + str(path))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out:
        shutil.copyfileobj(stream, out)
    return True

def extract_zip(archive,dest,max_bytes=8*1024**3,reuse_identical=False):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        require(sum(i.file_size for i in z.infolist())<=max_bytes,'ZIP exceeds extraction limit')
        seen=set()
        for i in z.infolist():
            name=safe_name(i.filename).as_posix()
            require(name not in seen,'Duplicate ZIP member: '+name);seen.add(name)
            mode=(i.external_attr>>16)&0xffff
            require(not stat.S_ISLNK(mode),'ZIP symlink refused: '+name)
            require(not (i.flag_bits&1),'Encrypted ZIP refused')
            p=in_root(dest,name)
            if i.is_dir(): p.mkdir(parents=True,exist_ok=True);continue
            with z.open(i) as a:
                created=write_or_verify_stream(a,p,reuse_identical)
            if created:p.chmod(0o755 if mode&0o111 else 0o644)
    return dest

def extract_tar(archive,dest,max_bytes=30*1024**3,allowed_prefix=None,reuse_identical=False):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);total=0;seen=set()
    with tarfile.open(archive,'r:*') as t:
        for m in t:
            name=safe_name(m.name).as_posix()
            require(name not in seen,'Duplicate TAR member: '+name);seen.add(name)
            if allowed_prefix:
                require(name.startswith(allowed_prefix+'/') or name==allowed_prefix,
                        'Unexpected TAR prefix: '+name)
            require(m.isdir() or m.isfile(),'TAR link/special member refused: '+name)
            p=in_root(dest,name)
            if m.isdir(): p.mkdir(parents=True,exist_ok=True);continue
            total+=m.size;require(total<=max_bytes,'TAR exceeds extraction limit')
            with t.extractfile(m) as a:
                created=write_or_verify_stream(a,p,reuse_identical)
            if created:p.chmod(0o755 if m.mode&0o111 else 0o644)
    return dest

def first(obj, paths, numeric=False):
    for path in paths:
        value=obj
        for key in path.split('.'):
            if not isinstance(value,dict) or key not in value: value=None;break
            value=value[key]
        if numeric:
            if type(value) not in (int,float) or not math.isfinite(value): continue
        elif value is None: continue
        return value,path
    return None,None

def identity_field(obj, paths):
    """Read an integer identity from JSON, never from a filename or a guessed row."""
    for name in paths:
        value, key = first(obj, (name,))
        if type(value) is str and re.fullmatch(r"[0-9]+", value):
            return int(value), key
        if type(value) in (int, float) and math.isfinite(value) and int(value) == value:
            return int(value), key
    return None, None

def baseline_record(j):
    if not isinstance(j,dict): return None
    meth,mk=first(j,('method','algorithm','algo','config.method','result.method','record.method','summary.method'))
    if not isinstance(meth,str): return None
    canon={'BHOJ':'BH-OJ','BH-OJ':'BH-OJ','GA':'GA','PSO':'PSO','CLPSO':'CLPSO','SMC':'SMC'}
    method=canon.get(meth.upper().replace('_','-'))
    if method is None: return None
    atoms,nk=identity_field(j,('atoms','N','n_atoms','n','config.atoms','config.N','result.atoms','record.atoms','summary.atoms'))
    run,rk=identity_field(j,('run','run_index','run_id','repetition','config.run','result.run','record.run'))
    objective,ek=first(j,('objective','best_energy','best_objective','best_f','f_best','fun','f','energy','best.energy','result.objective','result.best_energy','record.objective','summary.best_energy'),True)
    # Only result-shaped records; index metadata and task configs are not results.
    if atoms is None or run is None or objective is None: return None
    if int(atoms)!=atoms or int(run)!=run: return None
    seconds,tk=first(j,('seconds','elapsed_seconds','wall_seconds','runtime_seconds','duration_seconds','elapsed_time','time_s','time_sec','elapsed_s','wall_time_s','wall_time','runtime','elapsed','time_seconds','time','result.seconds','metrics.seconds','summary.seconds'),True)
    evals,ck=first(j,('n_func_evals','objective_evaluations','nfev','n_eval','evals_total','evaluations_used','objective_evals','function_evaluations','evals','evals_used','eval_count','objective_calls','evaluations','counters.objective','counters.objective_evaluations','counters.nfev','evaluations.objective','result.objective_evaluations','summary.objective_evaluations'),True)
    if seconds is None or evals is None: return None
    status,_=first(j,('status','termination','stop_reason','result.status'))
    require(atoms in ATOMS and 1<=run<=30,'Unexpected baseline result identity')
    require(seconds>=0 and evals>=0 and int(evals)==evals,'Invalid baseline cost')
    target,_=first(j,('target','target_energy','reference_energy','config.target','config.target_energy'),True)
    tol,_=first(j,('target_tol','target_tolerance','config.target_tol','config.target_tolerance'),True)
    hit,hk=first(j,('target_reached','hit_target','reached_target','result.target_reached'))
    if target is not None and tol is not None:
        success=objective<=target+tol; success_source='recorded target + recorded tolerance'
        if isinstance(hit,bool): require(hit==success,'Baseline target flag inconsistent')
    elif isinstance(hit,bool): success=hit;success_source=hk
    else: success=None;success_source='not reconstructed: no explicit target-attainment fields'
    return dict(method=method,atoms=int(atoms),run=int(run),objective=objective,seconds=seconds,
                evaluations=int(evals),status=status,success=success,success_source=success_source,
                field_paths=dict(method=mk,atoms=nk,run=rk,objective=ek,seconds=tk,evaluations=ck))

def hashes_in(obj):
    result=set();todo=[obj]
    while todo:
        x=todo.pop()
        if isinstance(x,dict):todo.extend(x.values())
        elif isinstance(x,list):todo.extend(x)
        elif isinstance(x,str) and re.fullmatch('[0-9a-fA-F]{64}',x):result.add(x.lower())
    return result

def managed_files(root):
    root=Path(root)
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if rel.parts[0] in IGNORED or '__pycache__' in rel.parts:continue
        if rel.as_posix()=='PACKAGE_MANIFEST.json':continue
        require(not p.is_symlink(),'Managed symlink refused: '+str(p))
        if p.is_file():yield rel.as_posix(),p

def seal(root):
    write_json(Path(root)/'PACKAGE_MANIFEST.json',{
        'schema':'hpfo-review-manifest-v1',
        'files':[dict(path=r,sha256=sha(p),bytes=p.stat().st_size) for r,p in managed_files(root)]})

def check_package(root,assets=True):
    root=Path(root);m=load(root/'PACKAGE_MANIFEST.json');seen=set()
    for row in m['files']:
        p=in_root(root,row['path']);seen.add(row['path'])
        require(p.is_file() and sha(p)==row['sha256'],'Managed file mismatch: '+str(p))
    require(seen=={r for r,p in managed_files(root)},'Unsealed added or missing file')
    if assets:
        require((root/'release-assets/SHA256SUMS.txt').read_bytes()==(root/SUBDIR/'provenance/RELEASE_SHA256SUMS.txt').read_bytes(), 'Release checksum list differs from sealed provenance')
        for row in load(root/SUBDIR/'provenance/RELEASE_ASSETS.json'):
            p=in_root(root/'release-assets',row['name'])
            require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],
                    'Release asset mismatch: '+str(p))
    return len(seen)
