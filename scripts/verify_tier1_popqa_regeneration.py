#!/usr/bin/env python3
"""CPU verification and v4-to-v4.2 transition table for Stage 2 PopQA regeneration."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from src.abstention_classifier import is_final_classifier_record
from src.config import CFG
from src.tokenization_contract import CONTRACT_VERSION
from src.eval_common import read_jsonl

def main():
    p=argparse.ArgumentParser();p.add_argument("--popqa-path",type=Path,default=CFG.data_dir/"popqa_sample.jsonl");a=p.parse_args()
    rows=read_jsonl(a.popqa_path)
    archived=read_jsonl(CFG.results_dir/"archive_v4"/"data"/"popqa_sample.jsonl")
    old_ids={r["popqa_id"] for r in archived};new_ids={r["popqa_id"] for r in rows}
    if old_ids!=new_ids:raise AssertionError(f"PopQA question IDs changed; missing={sorted(old_ids-new_ids)} added={sorted(new_ids-old_ids)}")
    missing=[]
    for r in rows:
        if not r.get("input_ids_sha256") or r.get("tokenization_contract_version")!=CONTRACT_VERSION or r.get("_generation_batch_size")!=1 or not is_final_classifier_record(r.get("classifier",{})) or not r.get("_v4_baseline_response") or not r.get("_v4_baseline_partition"):
            missing.append(r.get("popqa_id"))
    if missing:raise AssertionError(f"{len(missing)} rows fail canonical regeneration requirements: {missing[:20]}")
    counts=defaultdict(int)
    for r in rows:counts[(r["_v4_baseline_partition"],r["baseline_partition"])]+=1
    print("v4 partition -> v4.2 partition: count")
    for (old,new),n in sorted(counts.items()):print(f"{old:>9} -> {new:<9}: {n}")
    moved=sum(r["_v4_is_popularity_matched"] and r["_v4_baseline_partition"]=="wrong" and r["baseline_partition"]=="abstain" for r in rows)
    print(f"Previously matched wrong items now abstaining: {moved}")
    if moved < 20: raise RuntimeError("Fewer than 20 prior matched wrong items moved to abstain; stop and investigate the registered discrepancy.")
    print(f"Verified {len(rows)} rows; all hashes, v4 backups, batch sizes, and final judge records present.")
if __name__=="__main__":main()
