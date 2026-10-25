"""Read-only Tier 0/1 and v4.2 preflight for Tier 2."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.abstention_classifier import classify_texts
from src.popqa_uncertainty import classify_interpretation_v5

def main():
    required=[Path('src/tokenization_contract.py'),Path('src/abstention_classifier.py'),Path('results/archive_v42/MANIFEST.json'),Path('results/stage8_popqa_core.json'),Path('results/stage9_summary.json')]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit(f'missing preflight artifacts: {missing}')
    stage8=json.loads(Path('results/stage8_popqa_core.json').read_text(encoding='utf-8'))
    interp=stage8.get('interpretation',{})
    if 'overall_as_registered_v4' not in interp or 'rule_version' not in interp: raise SystemExit('v4.1 interpretation output missing')
    rows=[json.loads(s) for s in Path('data/popqa_sample.jsonl').read_text(encoding='utf-8').splitlines() if s]
    missing_hash=sum(not r.get('input_ids_sha256') for r in rows)
    missing_judged=sum(r.get('label_source')!='canonical_v1_batch1_judged' for r in rows)
    if missing_hash or missing_judged: raise SystemExit(f'Tier 1 PopQA prerequisite failed: rows={len(rows)}, missing input hash={missing_hash}, missing canonical judged label source={missing_judged}')
    print(f'preflight passed: canonical tokenization, judged classifier, v4.1 interpretation, {len(rows)} hashed judged PopQA rows, v4.2 archive')
if __name__=='__main__':main()
