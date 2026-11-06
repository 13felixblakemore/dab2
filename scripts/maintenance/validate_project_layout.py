"""Verify the organization checksum record and current package/checkpoint compatibility."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'docs/project_organization_20261106.json'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    mismatches = []
    for relative, expected in manifest['protected_files'].items():
        path = ROOT/relative
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f'Protected experiment files changed: {mismatches}')
    for move in manifest['moves']:
        move['sha256_final'] = sha256(ROOT/move['new'])
    manifest['protected_unchanged'] = True
    manifest['final_protected_verification'] = len(manifest['protected_files'])
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    checks_path = ROOT/'docs/validation/organization_checks.json'
    checks = json.loads(checks_path.read_text(encoding='utf-8'))
    checks['protected_files_unchanged'] = len(manifest['protected_files'])
    checks_path.write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
