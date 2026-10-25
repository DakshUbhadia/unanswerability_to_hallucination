"""Resolve and persist the immutable Hugging Face revision used by the local judge."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from huggingface_hub import HfApi
from src.config import CFG

def main():
    p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true',help='Resolve a new revision explicitly; requires a new validation run')
    a=p.parse_args();path=CFG.judge_model_pin_path
    if path.exists() and not a.refresh:
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('model_id')!=CFG.judge_model_id or len(row.get('revision_sha',''))<32:raise SystemExit('Existing judge pin is invalid; inspect it before refreshing')
        print(f"Pinned judge already set: {row['model_id']} @ {row['revision_sha']}");return
    info=HfApi().model_info(CFG.judge_model_id,revision='main')
    sha=str(info.sha)
    if len(sha)<32:raise RuntimeError(f"Hugging Face returned a non-commit revision: {sha}")
    payload={'model_id':CFG.judge_model_id,'revision_sha':sha,'resolved_from':'main',
             'resolved_at_utc':datetime.now(timezone.utc).isoformat(),'revision_validation_required':True}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(f"Pinned {CFG.judge_model_id} @ {sha}; validate this exact revision before labeling study data")
if __name__=='__main__':main()
