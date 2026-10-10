#!/usr/bin/env python3
"""Verify sealed sources/docs, optionally release assets and build evidence."""
import argparse,sys
from pathlib import Path
from hpfo_common import *

def verify(root,assets=False,for_publish=False):
    count=check_package(root,assets or for_publish)
    idx=load(root/SUBDIR/'provenance/RUN_INDEX.json')
    from collections import Counter
    require(Counter(r['dataset'] for r in idx)=={'main':240,'ipopt_multistart':240,'study':820},'PF index counts wrong')
    b=load(root/SUBDIR/'provenance/BASELINES_RUN_INDEX.json')
    require({(r['method'],r['atoms'],r['run']) for r in b}=={(m,n,r) for m in METHODS for n in ATOMS for r in range(1,31)} and len(b)==1200,'Baseline grid incomplete')
    if for_publish:
        report=load(root/'validation/build/LATEST.json')
        require(report.get('status')=='PASS' and {r['variant'] for r in report['builds']}=={'main','study'},'Clean build/test evidence missing or failed')
        require(report.get('python_source_syntax',{}).get('status')=='PASS','Python source syntax check missing/failed')
        inv=load(root/SUBDIR/'provenance/SOURCE_INVENTORY.json')
        require(report['source_inventory_sha256']==sha(root/SUBDIR/'provenance/SOURCE_INVENTORY.json'),'Build evidence refers to another source inventory')
        for r in inv:require(sha(root/r['path'])==r['sha256'],'Source differs from build inventory')
        for r in report['builds']:
            require(r['tests']>0 and r['status']=='PASS','Incomplete CTest evidence')
    print(f'PACKAGE_MANIFEST=OK ({count} files); PF=240; IPOPT=240; STUDY=820; BASELINES=1200')
    print('RELEASE_ASSETS='+('OK' if assets or for_publish else 'NOT_CHECKED'))
    print('BUILD_EVIDENCE='+('PASS' if for_publish else 'NOT_REQUIRED_FOR_THIS_CHECK'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',action='store_true');p.add_argument('--for-publish',action='store_true');a=p.parse_args()
    try:verify(Path(__file__).resolve().parents[1],a.assets,a.for_publish)
    except Exception as e:sys.exit('ERROR: '+str(e))
