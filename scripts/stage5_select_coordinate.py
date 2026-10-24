#!/usr/bin/env python3
"""Run the Stage 5 validation coordinate selection sweep."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

import torch
from nnsight import LanguageModel
from src.config import CFG
from src.stage_contract import load_stage2_model_contract
from src.direction_extraction import verify_stage4_outputs
from src.coordinate_selection import (load_and_validate_val_pairs, run_validation_coordinate_sweep,
 save_stage5_artifacts, verify_stage5_outputs)
from src.eval_common import load_stage_model


def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
 return h.hexdigest()

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--force",action="store_true"); p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
 p.add_argument("--dtype",default="auto"); p.add_argument("--batch-size",type=int,default=CFG.batch_size_forward)
 p.add_argument("--pairs-path",type=Path,default=CFG.data_dir/"pairs.jsonl"); p.add_argument("--splits-path",type=Path,default=CFG.data_dir/"splits.json")
 p.add_argument("--stage0-path",type=Path,default=CFG.stage0_out_path); p.add_argument("--stage2-path",type=Path,default=CFG.stage2_out_path)
 p.add_argument("--stage4-directions-path",type=Path,default=CFG.stage4_out_path); p.add_argument("--stage4-manifest-path",type=Path,default=CFG.stage4_manifest_path)
 p.add_argument("--output-path",type=Path,default=CFG.stage5_out_path); p.add_argument("--table2-csv-path",type=Path,default=CFG.table2_csv_path); p.add_argument("--table2-md-path",type=Path,default=CFG.table2_md_path)
 a=p.parse_args(argv)
 if not a.force and all(x.exists() for x in (a.output_path,a.table2_csv_path,a.table2_md_path)):
  try: verify_stage5_outputs(a.output_path,a.table2_csv_path,a.table2_md_path,a.stage0_path,a.stage4_manifest_path)
  except (AssertionError,KeyError,ValueError,RuntimeError): pass
  else: print(f"[SKIP] Stage 5 already verified in {a.output_path}"); return 0
 if not a.stage4_directions_path.exists():
  repo=os.environ.get("HF_TENSOR_REPO_ID","")
  if not repo: raise FileNotFoundError(f"Missing {a.stage4_directions_path}; set HF_TENSOR_REPO_ID to restore it")
  from scripts.kaggle_sync import ensure_tensor_cached_from_hf
  ensure_tensor_cached_from_hf("stage4_directions.pt",repo,a.stage4_directions_path.parent.parent)
 verify_stage4_outputs(a.stage4_directions_path,a.stage4_manifest_path,a.stage0_path)
 directions=torch.load(a.stage4_directions_path,map_location="cpu",weights_only=False)
 stage2=load_stage2_model_contract(a.stage2_path)
 stage0=json.loads(a.stage0_path.read_text(encoding="utf-8")); contract={**stage0,**stage2}
 val=load_and_validate_val_pairs(a.pairs_path,a.splits_path)
 raw,tokenizer,_=load_stage_model(contract,a.device,a.dtype)
 llm=LanguageModel(raw, tokenizer=tokenizer, dispatch=True)
 result=run_validation_coordinate_sweep(llm,contract,directions,val,a.batch_size,CFG.stage5_checkpoint_path,a.force,stage2,_sha(a.stage4_directions_path))
 save_stage5_artifacts(result,a.output_path,a.table2_csv_path,a.table2_md_path,None)
 checked=verify_stage5_outputs(a.output_path,a.table2_csv_path,a.table2_md_path,a.stage0_path,a.stage4_manifest_path)
 if CFG.stage5_checkpoint_path.exists(): CFG.stage5_checkpoint_path.unlink()
 print(f"[Stage 5] coordinate=({checked['l_star']}, {checked['i_star']}, {checked['pos_idx']}) norm={checked['r_norm_at_star']:.6g}")
 print(f"ablation={checked['selected_ablation_log_odds']:.4f}, steering={checked['selected_steering_log_odds']:.4f}, KL={checked['selected_kl_val']:.6f}")
 print(a.table2_md_path.read_text(encoding="utf-8"))
 return 0
if __name__=="__main__": raise SystemExit(main())
