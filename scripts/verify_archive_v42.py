"""Reproducibility check for an archived snapshot manifest."""
import argparse,hashlib,json
from pathlib import Path

def verify(root=Path('.')):
    root=Path(root); manifest=root/'results/archive_v42/MANIFEST.json'; data=json.loads(manifest.read_text(encoding='utf-8'))
    for rel,meta in data['files'].items():
        p=root/'results/archive_v42'/rel
        b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=meta['sha256'] or len(b)!=meta['size_bytes']:
            raise ValueError(f"archive mismatch: {rel}")
    return len(data['files'])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');a=ap.parse_args();print(f"verified {verify(a.root)} archived files")
