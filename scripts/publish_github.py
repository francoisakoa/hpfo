#!/usr/bin/env python3
"""Publish the review artifact to a NEW branch/tag in francoisakoa/hpfo.
Default: print a plan only. --apply --confirm-rights-and-privacy enables publication.
No forced push, deletion, visibility change, or default-branch overwrite.
"""
from __future__ import annotations
import argparse,json,os,shlex,subprocess,sys,tempfile
from pathlib import Path
from hpfo_common import *
from verify_package import verify
REPO='francoisakoa/hpfo'

def cmd(args,cwd=None,capture=False):
    print('+ '+shlex.join(list(map(str,args))),flush=True)
    v=subprocess.run(list(map(str,args)),cwd=cwd,text=True,capture_output=capture)
    if v.returncode:raise RuntimeError('Command failed'+(': '+v.stderr.strip() if capture else ''))
    return v.stdout if capture else ''

def ref_exists(repo,ref):
    p=subprocess.run(['git','ls-remote','--exit-code','--refs','origin',ref],cwd=repo,text=True,capture_output=True)
    require(p.returncode in (0,2),'Cannot query remote references: '+p.stderr)
    return p.returncode==0

def publish(root,a):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*',a.branch) and '..' not in a.branch,'Invalid branch')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',a.tag),'Invalid tag')
    print('REPOSITORY='+REPO+'\nNEW_BRANCH='+a.branch+'\nNEW_TAG='+a.tag)
    print('Plan: verify local package/build evidence; clone existing repository; preserve existing files;')
    print('create a new review branch; commit; push branch/tag; upload draft release assets; publish release.')
    if not a.apply:
        print('DRY_RUN=YES ; no network/write operation performed');return
    require(a.confirm_rights_and_privacy,'Explicit --confirm-rights-and-privacy is required')
    verify(root,assets=True,for_publish=True)
    require(shutil.which('git') and shutil.which('gh'),'git and GitHub CLI gh are required')
    cmd(['gh','auth','status'])
    meta=json.loads(cmd(['gh','repo','view',REPO,'--json','nameWithOwner,isPrivate,url'],capture=True))
    require(meta['nameWithOwner'].lower()==REPO,'Wrong repository')
    print('REMOTE_PRIVATE='+str(meta['isPrivate']))
    (root/'.local').mkdir(exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='publish-',dir=root/'.local'));checkout=temp/'checkout'
    cmd(['git','clone','https://github.com/'+REPO+'.git',checkout])
    require(not ref_exists(checkout,'refs/heads/'+a.branch),'Remote review branch exists; no overwrite')
    require(not ref_exists(checkout,'refs/tags/'+a.tag),'Remote tag exists; no overwrite')
    require(not (checkout/SUBDIR).exists(),'Remote artifact directory already exists; no overwrite')
    cmd(['git','switch','-c',a.branch],cwd=checkout)
    manifest=load(root/'PACKAGE_MANIFEST.json')
    for entry in manifest['files']:
        rel=entry['path'];src=root/rel;dst=checkout/rel
        if dst.exists():
            require(dst.is_file() and not dst.is_symlink(),'Remote path conflict: '+rel)
            if sha(dst)==sha(src):continue
            if rel=='README.md':
                dst.write_text(dst.read_text()+'\n\n---\n\n'+src.read_text(),encoding='utf-8');continue
            if rel=='.gitignore':
                dst.write_text(dst.read_text()+'\n# PF131 review artifact\n'+src.read_text(),encoding='utf-8');continue
            if rel=='CITATION.cff':
                # Keep the repository-wide citation; artifact citation stays inside SUBDIR.
                continue
            raise RuntimeError('Remote file conflict; refusing overwrite: '+rel)
        snapshot_copy(src,dst)
    require(not (checkout/'PACKAGE_MANIFEST.json').exists(),'Remote package manifest exists; no overwrite')
    seal(checkout);check_package(checkout,assets=False)
    # Stage ONLY our copied paths and explicitly merged top-level documentation.
    paths=[r['path'] for r in manifest['files']]+['PACKAGE_MANIFEST.json']
    for k in range(0,len(paths),150):cmd(['git','add','--',*paths[k:k+150]],cwd=checkout)
    names=cmd(['git','diff','--cached','--name-only'],cwd=checkout,capture=True)
    require(names.strip(),'No staged files')
    require(not any(n.startswith(('release-assets/','data/','.local/')) for n in names.splitlines()),'Large/local files accidentally staged')
    cmd(['git','commit','-m','Add PF C++ 1.3.1 JOGO reviewer artifact and completed experiments'],cwd=checkout)
    commit=cmd(['git','rev-parse','HEAD'],cwd=checkout,capture=True).strip()
    cmd(['git','tag','-a',a.tag,'-m','PF131 JOGO reproducibility artifact'],cwd=checkout)
    cmd(['git','push','--atomic','origin','HEAD:refs/heads/'+a.branch,'refs/tags/'+a.tag],cwd=checkout)
    assets=[root/'release-assets'/r['name'] for r in load(root/SUBDIR/'provenance/RELEASE_ASSETS.json')]+[root/'release-assets/SHA256SUMS.txt']
    cmd(['gh','release','create',a.tag,*assets,'--repo',REPO,'--verify-tag','--draft','--latest=false',
         '--title','PF C++ 1.3.1 — JOGO review artifact','--notes-file',root/'docs/RELEASE_NOTES.md'])
    release=json.loads(cmd(['gh','api','repos/'+REPO+'/releases/tags/'+a.tag],capture=True))
    actual={x['name']:x for x in release['assets']}
    for p in assets:
        require(p.name in actual and actual[p.name]['size']==p.stat().st_size,'Remote asset missing/wrong size; release left as draft')
        digest=actual[p.name].get('digest')
        if digest:require(digest=='sha256:'+sha(p),'Remote digest mismatch; release left as draft')
    cmd(['gh','release','edit',a.tag,'--repo',REPO,'--draft=false'])
    result={'repository':REPO,'branch':a.branch,'tag':a.tag,'commit':commit,'is_private':meta['isPrivate'],
            'code_url':'https://github.com/'+REPO+'/tree/'+a.tag,
            'release_url':'https://github.com/'+REPO+'/releases/tag/'+a.tag,
            'reviewer_account_access_verified':False}
    write_json(temp/'PUBLICATION.json',result)
    print(json.dumps(result,indent=2))
    print('PUBLICATION=COMPLETE ; DEFAULT_BRANCH_UNCHANGED ; REVIEWER_ACCOUNT_ACCESS=NOT_TESTED')
    if meta['isPrivate']:print('Grant the Editor/reviewers repository access before sending this private URL.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');p.add_argument('--confirm-rights-and-privacy',action='store_true')
    p.add_argument('--branch',default='review/pf131-jogo-r2');p.add_argument('--tag',default='pf131-jogo-r2');a=p.parse_args()
    try:publish(Path(__file__).resolve().parents[1],a)
    except Exception as e:sys.exit('PUBLICATION_ERROR: '+str(e))
