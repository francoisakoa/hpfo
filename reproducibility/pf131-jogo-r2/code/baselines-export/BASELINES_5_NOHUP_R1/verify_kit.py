#!/usr/bin/env python3
"""Verify immutable delivered orchestration files (not solver results)."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
expected=json.loads((root/'MANIFEST_SHA256.json').read_text())
bad=[n for n,h in expected.items() if not (root/n).is_file() or hashlib.sha256((root/n).read_bytes()).hexdigest()!=h]
if bad:print('KIT_INTEGRITY_FAIL:',bad);raise SystemExit(2)
print(f'KIT_INTEGRITY_PASS: {len(expected)} files')
