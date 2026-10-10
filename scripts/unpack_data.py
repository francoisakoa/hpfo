#!/usr/bin/env python3
"""Verify and extract release assets locally. Never overwrite data/."""
import sys,tempfile
from pathlib import Path
from hpfo_common import *

def unpack(root):
    check_package(root,True)
    require(not (root/'data').exists(),'data/ already exists; no overwrite performed')
    (root/'.local').mkdir(exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='unpack-',dir=root/'.local'))
    for item in load(root/SUBDIR/'provenance/RELEASE_ASSETS.json'):
        p=root/'release-assets'/item['name'];print('EXTRACT='+p.name,flush=True)
        if p.name=='hpfo-pf131-main-data.tar.gz':extract_tar(p,tmp,allowed_prefix='data/main')
        elif p.name=='hpfo-pf131-study-data.tar.gz':extract_tar(p,tmp,allowed_prefix='data/study')
        elif p.name.startswith('BASELINES_COMPACT_') and p.suffix=='.zip':extract_zip(p,tmp/'data/baselines-export')
        else:raise RuntimeError('Unknown asset format: '+p.name)
    seen=set()
    for depth in range(3):
        nested=[p for p in (tmp/'data/baselines-export').rglob('*') if p.is_file() and p.name.endswith(('.zip','.tar.gz','.tgz','.tar')) and p not in seen]
        if not nested:break
        for p in nested:
            seen.add(p);d=p.parent/(p.name+'.contents')
            if p.suffix=='.zip':extract_zip(p,d,max_bytes=1024**3)
            else:extract_tar(p,d,max_bytes=1024**3)
    for index_name in ('RUN_INDEX.json','BASELINES_RUN_INDEX.json'):
        for r in load(root/SUBDIR/'provenance'/index_name):
            p=in_root(tmp/'data',r['file']);require(sha(p)==r['sha256'],'Extracted JSON hash mismatch')
    require(not (root/'data').exists(),'Destination appeared during extraction')
    (tmp/'data').rename(root/'data')
    print('DATA_READY='+str(root/'data')+' ; JSON_HASHES=2500/2500 ; OPTIMIZATION=NOT_RUN')

if __name__=='__main__':
    try:unpack(Path(__file__).resolve().parents[1])
    except Exception as e:sys.exit('ERROR: '+str(e))
