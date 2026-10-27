"""Read-only Tier 0/1 and v4.2 preflight for Tier 2."""
import json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.abstention_classifier import classify_texts
from scripts.verify_archive_v42 import verify as verify_archive

def verify_preflight():
    required=[Path('src/tokenization_contract.py'),Path('src/abstention_classifier.py'),Path('results/archive_v42/MANIFEST.json'),Path('results/stage8_popqa_core.json'),Path('results/stage9_summary.json')]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit(f'missing preflight artifacts: {missing}')
    if not callable(classify_texts): raise SystemExit('Tier 1 judged classifier interface missing')
    stage8=json.loads(Path('results/stage8_popqa_core.json').read_text(encoding='utf-8'))
    interp=stage8.get('interpretation',{})
    if 'overall_as_registered_v4' not in interp or 'rule_version' not in interp: raise SystemExit('v4.1 interpretation output missing')
    rows=[json.loads(s) for s in Path('data/popqa_sample.jsonl').read_text(encoding='utf-8').splitlines() if s]
    missing_hash=sum(not r.get('input_ids_sha256') for r in rows)
    missing_judged=sum(r.get('label_source')!='canonical_v1_batch1_judged' for r in rows)
    if missing_hash or missing_judged: raise SystemExit(f'Tier 1 PopQA prerequisite failed: rows={len(rows)}, missing input hash={missing_hash}, missing canonical judged label source={missing_judged}')
    n_archive=verify_archive()
    subprocess.check_call([sys.executable,str(Path(__file__).with_name('verify_tier2_commit_order.py'))])
    return {'n_popqa_rows':len(rows),'archive_files_verified':n_archive}
def main():
    result=verify_preflight()
    print(f"preflight passed: canonical tokenization, judged classifier, v4.1 interpretation, {result['n_popqa_rows']} hashed judged PopQA rows, {result['archive_files_verified']} verified v4.2 archive files")
if __name__=='__main__':main()
