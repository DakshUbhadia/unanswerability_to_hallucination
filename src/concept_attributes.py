"""Derive registered prompt attributes and label-balanced contrast directions."""
from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np

WH_WORDS=("Who","Which","What","Where","When","How","other")
ANSWER_TYPES=("year","place","person","vessel-or-artifact","other")
DOMAINS=("physical science","life science","engineering/infrastructure","humanities/archival","other")

def _context(prompt): return prompt.split("Question:",1)[0].strip()
def _wh(prompt):
    q=prompt.split("Question:",1)[-1].strip()
    m=re.search(r"\b(who|which|what|where|when|how)\b",q,re.I)
    return m.group(1).capitalize() if m else "other"
def _answer_type(text):
    t=text.strip().lower()
    if re.search(r"\b(1[5-9]\d\d|20\d\d|21\d\d)\b",t): return "year"
    if re.search(r"\b(ship|vessel|aircraft|artifact|artefact|instrument|device)\b",t): return "vessel-or-artifact"
    if re.search(r"\b(city|river|mountain|country|region|valley|island|lake|place)\b",t): return "place"
    if len(t.split())>=2 and all(w[:1].isupper() for w in text.split()[:2]): return "person"
    return "other"
def derive_attribute_rows(pairs_path="data/pairs.jsonl", templates_path="data/templates.jsonl", domain_path="data/domain_categories.json", tokenizer=None):
    # Load the locked template table as an integrity check. Pair rows carry rendered
    # prompts, but template metadata is the authoritative source for family/domain.
    templates=[json.loads(s) for s in Path(templates_path).read_text(encoding="utf-8").splitlines() if s]
    template_by_id={r["template_id"]:r for r in templates}
    domains=json.loads(Path(domain_path).read_text(encoding="utf-8")); domains=domains.get("mapping",domains); pairs=[json.loads(s) for s in Path(pairs_path).read_text(encoding="utf-8").splitlines() if s]
    # The registered table contains one row per distinct answerable/unanswerable prompt.
    rows=[]
    for p in pairs:
        if p["split"]!="train": continue
        if p["template_id"] not in template_by_id:
            raise ValueError(f"training pair references unknown template {p['template_id']}")
        if template_by_id[p["template_id"]].get("family") != p["family"]:
            raise ValueError(f"family mismatch for template {p['template_id']}")
        for label,key in (("answerable","ans_prompt"),("unanswerable","unans_prompt")):
            prompt=p[key];ctx=_context(prompt);q=prompt.split("Question:",1)[-1]
            token_len=len(tokenizer.encode(prompt,add_special_tokens=False)) if tokenizer is not None else int(p.get("ans_token_len" if label=="answerable" else "unans_token_len",len(prompt.split())))
            answer_tokens=len(tokenizer.encode(p["ground_truth"],add_special_tokens=False)) if tokenizer is not None else len(p["ground_truth"].split())
            rows.append({"pair_id":p["pair_id"],"template_id":p["template_id"],"family":p["family"],"instance_idx":int(p["instance_idx"]),"label":label,
                "wh_word":_wh(prompt),"answer_type":_answer_type(p["ground_truth"]) if p["family"]=="B" else None,
                "prompt_token_length":token_len,"context_sentence_count":len(re.findall(r"[.!?](?:\s|$)",ctx)),
                "coarse_domain":domains[p["template_id"]],"digits_in_context":bool(re.search(r"\d",ctx)),"answer_token_length":"1" if answer_tokens==1 else "2+"})
    return contrast_labels(rows)

def save_attribute_rows(rows, path="data/concept_attributes.jsonl"):
    """Persist the deterministic attribute table as registered JSONL."""
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix(target.suffix+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    tmp.replace(target)
    return target

def contrast_labels(rows):
    out={}
    for r in rows:
        for name,level in (("family",r["family"]),("wh_word",r["wh_word"]),("answer_type",r["answer_type"]),
                           ("context_sentence_count","2" if r["context_sentence_count"]==2 else "3+"),
                           ("instance_idx",str(r["instance_idx"])),("coarse_domain",r["coarse_domain"]),
                           ("digits_in_context","present" if r["digits_in_context"] else "absent"),("answer_token_length",r["answer_token_length"])):
            if level is not None: out.setdefault(name,{})[str(level)]=True
    # Prompt length terciles are computed once over the registered training prompt table.
    lengths=np.array([r["prompt_token_length"] for r in rows]); q1,q2=np.quantile(lengths,[1/3,2/3])
    for r in rows:r["prompt_length_tercile"]="bottom" if r["prompt_token_length"]<=q1 else ("top" if r["prompt_token_length"]>q2 else "middle")
    return rows

def make_contrast_directions(acts, rows, r_hat, r_norm, min_per_side=32):
    """Return eligible label-balanced contrast directions and auditable group counts."""
    x=np.asarray(acts,float); rh=np.asarray(r_hat,float);rh/=np.linalg.norm(rh)
    if x.ndim!=2 or len(x)!=len(rows): raise ValueError("one activation row per attribute record required")
    attrs={"family":["A","B"],"wh_word":list(WH_WORDS),"answer_type":list(ANSWER_TYPES),"prompt_length_tercile":["top","bottom"],
      "context_sentence_count":["2","3+"],"instance_idx":["0","1","2","3"],"coarse_domain":list(DOMAINS),
      "digits_in_context":["present","absent"],"answer_token_length":["1","2+"]}
    directions=[]
    for attr,levels in attrs.items():
        for level in levels:
            # answer type is defined only on family B; all other attributes use the available training rows.
            selected=[i for i,r in enumerate(rows) if attr!="answer_type" or r["family"]=="B"]
            pos=np.array([i for i in selected if str(rows[i].get(attr))==str(level)])
            neg=np.array([i for i in selected if str(rows[i].get(attr))!=str(level)])
            # Require >=32 prompts on each side within each answerability label.
            ok=True; diffs=[]; counts={}
            for lab in ("answerable","unanswerable"):
                pp=np.array([i for i in pos if rows[i]["label"]==lab]); nn=np.array([i for i in neg if rows[i]["label"]==lab])
                counts[lab]={"level":len(pp),"other":len(nn)}
                if len(pp)<min_per_side or len(nn)<min_per_side:ok=False;break
                diffs.append(x[pp].mean(0)-x[nn].mean(0))
            if not ok:continue
            raw=np.mean(diffs,axis=0);cos=float(raw@rh/(np.linalg.norm(raw)+1e-12));orth=raw-(raw@rh)*rh
            norm=np.linalg.norm(orth)
            if norm<1e-12:continue
            directions.append({"attribute":attr,"level":str(level),"group_sizes":counts,"pre_orthogonalization_cosine":cos,
                "unit":(orth/norm).astype(np.float32),"steer":(orth/norm*float(r_norm)).astype(np.float32)})
    return directions
