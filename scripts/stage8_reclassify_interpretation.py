"""Reclassify committed Stage 8 interpretation without a model rerun."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.config import CFG
from src.eval_common import atomic_write_json,read_json
from src.popqa_uncertainty import classify_interpretation

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--check",action="store_true"); args=ap.parse_args()
    old=read_json(CFG.stage8_out_path); new=copy.deepcopy(old); s6=read_json(CFG.stage6_out_path)
    new["interpretation"]=classify_interpretation(new["holm_bonferroni_table"],s6,new["h4"],new["h5"],new["layerwise_auroc_exploratory"],new["l_star"],new["auroc_abstain_vs_correct"])
    assert json.dumps({k:v for k,v in old.items() if k!="interpretation"},sort_keys=True)==json.dumps({k:v for k,v in new.items() if k!="interpretation"},sort_keys=True)
    print("old overall:",old.get("interpretation",{}).get("overall")); print("new overall:",new["interpretation"]["overall"]); print("overall_as_registered_v4:",new["interpretation"]["overall_as_registered_v4"])
    if args.check:
        if old.get("interpretation")!=new["interpretation"]: raise SystemExit(1)
    else: atomic_write_json(CFG.stage8_out_path,new)
if __name__=="__main__": main()
