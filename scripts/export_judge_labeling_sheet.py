#!/usr/bin/env python3
"""Create frozen, blinded dev/test sheets for response-judge validation."""
import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.config import CFG

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def pool_rows():
    pool={}
    pairs={r["pair_id"]:r for r in read_jsonl(CFG.data_dir/"pairs.jsonl")}
    bctrl={r["item_id"]:r for r in read_jsonl(CFG.data_dir/"b_ctrl.jsonl")}
    for r in (read_jsonl(CFG.stage6_gen_path) if CFG.stage6_gen_path.exists() else []):
        uid=f"stage6:{r.get('condition')}:{r.get('item_id')}"
        prompt=(pairs.get(r.get("pair_id"),{}).get("unans_prompt" if r.get("dataset_part")=="unans" else "ans_prompt")
                or bctrl.get(r.get("item_id"),{}).get("prompt"))
        if not prompt: raise ValueError(f"Cannot join Stage 6 prompt for {uid}")
        pool[uid]={"uid":uid,"stratum":f"stage6:{r.get('condition')}","question":prompt,
                   "response":r.get("generated_text",""),"final_label":r.get("final_label"),
                   "regex_label":r.get("regex_label"),"judge_margin":r.get("judge_margin"),"judge_quantization":r.get("judge_quantization")}
    if CFG.stage10_out_path.exists():
        d=json.loads(CFG.stage10_out_path.read_text(encoding="utf-8"))
        pop={r["popqa_id"]:r for r in read_jsonl(CFG.data_dir/"popqa_sample.jsonl")}
        for r in d.get("items",[]):
            if r.get("alpha") is None: continue
            src=pop.get(r.get("popqa_id"),{})
            uid=f"stage10:{r.get('direction','learned')}:{r.get('alpha')}:{r.get('popqa_id')}"
            pool[uid]={"uid":uid,"stratum":f"stage10:{r.get('alpha')}:{r.get('direction','learned')}","question":src.get("question",""),
                       "response":r.get("generated_text",""),"final_label":r.get("final_label"),"regex_label":r.get("regex_label"),"judge_margin":r.get("judge_margin"),"judge_quantization":r.get("judge_quantization")}
    for r in read_jsonl(CFG.data_dir/"popqa_sample.jsonl"):
        uid=f"popqa:{r['popqa_id']}"
        rec=r.get("classifier",{})
        pool[uid]={"uid":uid,"stratum":"popqa:baseline","question":r["question"],
                   "response":r.get("baseline_response",""),"final_label":rec.get("final_label"),
                   "regex_label":rec.get("regex_label"),"judge_margin":rec.get("judge_margin")}
    return pool

def allocate(groups, n, rng, floor):
    if len(groups)*floor>n: raise ValueError("Sample size is too small for the per-stratum floor")
    total=sum(map(len,groups.values())); quotas={k:min(len(v),floor) for k,v in groups.items()}
    left=n-sum(quotas.values())
    while left:
        eligible=[k for k,v in groups.items() if quotas[k]<len(v)]
        if not eligible: break
        weights={k:len(groups[k])-quotas[k] for k in eligible}; den=sum(weights.values())
        exact={k:left*weights[k]/den for k in eligible}
        add={k:min(len(groups[k])-quotas[k],int(exact[k])) for k in eligible}
        for k,v in add.items(): quotas[k]+=v
        used=sum(add.values()); left-=used
        if left:
            order=sorted((k for k in eligible if quotas[k]<len(groups[k])),key=lambda k:(-(exact[k]-int(exact[k])),k))
            for k in order[:left]: quotas[k]+=1
            left-=min(left,len(order))
    out=[]
    for k,rows in groups.items(): out.extend(rng.sample(rows,quotas[k]))
    return out

def main():
    from src.abstention_classifier import classify_texts
    a=argparse.ArgumentParser(); a.add_argument("--output-dir",type=Path,default=CFG.results_dir); x=a.parse_args()
    pool=pool_rows()
    pending=[r for r in pool.values() if not r.get("final_label")]
    if pending:
        classified=classify_texts([{"uid":r["uid"],"question":r["question"],"response":r["response"]} for r in pending])
        for row,rec in zip(pending,classified):
            row["final_label"],row["regex_label"]=rec["final_label"],rec["regex_label"]
            row["judge_margin"],row["judge_quantization"]=rec.get("judge_margin"),rec.get("judge_quantization")
    groups=defaultdict(list)
    for r in pool.values(): groups[r["stratum"]].append(r)
    bad={k:sum(row["final_label"]=="UNPARSEABLE" for row in v) for k,v in groups.items()}
    over={k:f"{n}/{len(groups[k])}" for k,n in bad.items() if n/len(groups[k])>.01}
    if over: raise RuntimeError(f"Judge validation pool exceeds the 1% UNPARSEABLE limit: {over}")
    rng=random.Random(CFG.judge_validation_seed)
    test=allocate(groups,200,rng,4); selected={r["uid"] for r in test}
    rem={k:[r for r in v if r["uid"] not in selected] for k,v in groups.items()}
    # Dev selection oversamples extreme-alpha rows and model disagreements.
    ranked=[]
    for rows in rem.values():
        for r in rows:
            alpha_extreme="stage10:" in r["stratum"] and (":-2.0:" in r["stratum"] or ":2.0:" in r["stratum"])
            ranked.append((0 if r["final_label"]!=r["regex_label"] else 1,0 if alpha_extreme else 1,r))
    ranked.sort(key=lambda x:(x[0],x[1],x[2]["uid"]))
    dev=[x[2] for x in ranked[:100]]
    x.output_dir.mkdir(parents=True,exist_ok=True)
    import hashlib
    blinded={r["uid"]:"J"+hashlib.sha256(r["uid"].encode()).hexdigest()[:18] for r in test+dev}
    for split,rows in (("test",test),("dev",dev)):
        shuffled=list(rows); rng.shuffle(shuffled)
        with (x.output_dir/f"judge_validation_{split}.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["uid","question","response","human_label","notes"]); w.writeheader()
            for r in shuffled: w.writerow({"uid":blinded[r["uid"]],"question":r["question"],"response":r["response"],"human_label":"","notes":""})
    key={blinded[r["uid"]]:{"final_label":r["final_label"],"regex_label":r["regex_label"],"judge_margin":r.get("judge_margin"),"stratum":r["stratum"]} for r in test+dev}
    (x.output_dir/"judge_validation_key.json").write_text(json.dumps(key,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    from src.abstention_classifier import JUDGE_PROMPT_SHA256, JUDGE_PROMPT_VERSION, _revision
    manifest={"seed":CFG.judge_validation_seed,"judge_model_id":CFG.judge_model_id,"judge_model_revision":_revision(),
              "judge_quantization":next((r.get("judge_quantization") for r in test+dev if r.get("judge_quantization")),CFG.judge_quantization),"judge_prompt_version":JUDGE_PROMPT_VERSION,
              "judge_prompt_sha256":JUDGE_PROMPT_SHA256,"test_uids":sorted(blinded[r["uid"]] for r in test),
              "dev_uids":sorted(blinded[r["uid"]] for r in dev),"strata":{k:len(v) for k,v in groups.items()}}
    (x.output_dir/"judge_validation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Wrote blinded sheets: {len(test)} held-out, {len(dev)} dev")
if __name__=="__main__": main()
