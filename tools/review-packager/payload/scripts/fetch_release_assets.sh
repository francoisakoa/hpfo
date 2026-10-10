#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${1:-pf131-jogo-r2}
command -v gh >/dev/null || { echo 'Install GitHub CLI (gh) first.' >&2; exit 2; }
mkdir -p "$ROOT/.local"
TMP=$(mktemp -d "$ROOT/.local/download_XXXXXXXX")
[[ ! -e "$ROOT/release-assets" ]] || { echo 'release-assets/ already exists; not overwritten.' >&2; exit 2; }
gh release download "$TAG" --repo francoisakoa/hpfo --dir "$TMP" --pattern 'hpfo-pf131-*-data.tar.gz' --pattern 'BASELINES_COMPACT_*.zip' --pattern 'SHA256SUMS.txt'
python3 - "$ROOT" "$TMP" <<'PY'
import sys
from pathlib import Path
root,tmp=map(Path,sys.argv[1:]);sys.path.insert(0,str(root/'scripts'))
from hpfo_common import load,sha,require,SUBDIR
for r in load(root/SUBDIR/'provenance/RELEASE_ASSETS.json'):
 p=tmp/r['name'];require(p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Downloaded asset mismatch: '+r['name'])
require(not (root/'release-assets').exists(),'Destination appeared during download')
tmp.rename(root/'release-assets')
print('DOWNLOADED_ASSETS=VERIFIED')
PY
