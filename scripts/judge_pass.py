"""Score unique generated responses into the offline judge cache."""
import argparse,hashlib,importlib.metadata,json,sys,time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.abstention_classifier import classify_texts
from src.config import CFG
from src.eval_common import read_jsonl

def _records(path):
    path=Path(path)
    if path.suffix.lower()=='.jsonl':return read_jsonl(path)
    data=json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data,list):return data
    if isinstance(data,dict):
        for key in ('items','rows','generations','matched_items'):
            if isinstance(data.get(key),list):return data[key]
    raise ValueError(f'No response rows found in {path}')

def collect(paths):
    pairs={r['pair_id']:r for r in read_jsonl(CFG.data_dir/'pairs.jsonl')}
    bctrl={r['pair_id']:r for r in read_jsonl(CFG.data_dir/'b_ctrl.jsonl')}
    popqa={r['popqa_id']:r for r in read_jsonl(CFG.data_dir/'popqa_sample.jsonl')}
    unique={};raw_by_path={};condition_keys=defaultdict(set)
    for path in paths:
        count=0
        for i,row in enumerate(_records(path)):
            response=row.get('response',row.get('generated_text',row.get('baseline_response')))
            if response is None:continue
            question=row.get('question')
            if not question:
                pid=row.get('pair_id')
                if row.get('dataset_part')=='bctrl' or str(row.get('condition','')).endswith('bctrl'):
                    src=bctrl.get(pid,{})
                    question=src.get('prompt')
                else:
                    src=pairs.get(pid,{})
                    part=row.get('dataset_part')
                    question=src.get('unans_prompt' if part=='unans' else 'ans_prompt')
                if not question and row.get('popqa_id'):
                    question=popqa.get(row['popqa_id'],{}).get('question')
            if not question:raise ValueError(f"No question for {path} row {i}; row keys={sorted(row)}")
            key=(str(question),str(response));count+=1
            uid=str(row.get('uid',row.get('popqa_id',row.get('item_id',f'{Path(path).name}:{i}'))))
            condition=str(row.get('condition',row.get('direction','baseline')))
            if key not in unique:unique[key]={'uid':uid,'question':str(question),'response':str(response),'conditions':set(),'source_paths':set()}
            unique[key]['conditions'].add(condition);unique[key]['source_paths'].add(str(path))
            condition_keys[condition].add(key)
        raw_by_path[str(path)]=count
    items=[]
    for row in unique.values():
        items.append({**row,'conditions':sorted(row['conditions']),'source_paths':sorted(row['source_paths'])})
    return items,raw_by_path,condition_keys

def _version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',action='append',type=Path)
    p.add_argument('--batch-size',type=int,default=CFG.judge_batch_size);p.add_argument('--offline',action='store_true');p.add_argument('--limit-unique',type=int,help='Score only the first N deduplicated items for a throughput pilot')
    p.add_argument('--output',type=Path,default=CFG.judge_pass_summary_path);a=p.parse_args()
    paths=a.input or [Path(x) for x in CFG.judge_inputs if Path(x).exists()]
    missing=[str(x) for x in (a.input or []) if not Path(x).exists()]
    if missing:raise FileNotFoundError(f'Response inputs missing: {missing}')
    if not paths:raise FileNotFoundError('No response inputs found; generate and save responses first')
    items,raw_by_path,condition_keys=collect(paths);n_unique_available=len(items)
    if a.limit_unique is not None:
        if a.limit_unique<=0:raise ValueError('--limit-unique must be positive')
        items=items[:a.limit_unique];selected={(r['question'],r['response']) for r in items}
        condition_keys={k:v & selected for k,v in condition_keys.items()}
    t=time.perf_counter()
    records=classify_texts(items,offline=a.offline,batch_size=a.batch_size)
    by_key={(r['question'],r['response']):r for r in records}
    margins={}
    for cond,keys in condition_keys.items():
        vals=[by_key[k].get('judge_margin') for k in keys if by_key[k].get('judge_margin') is not None]
        margins[cond]={'unique_items':len(keys),'scored_items':len(vals),'low_margin_lt_1_nat':sum(float(v)<1 for v in vals)}
    cache_path=Path(CFG.judge_cache_path)
    sha=hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path.exists() else None
    try:
        import torch
        cuda={'available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}
    except Exception:cuda={'available':False,'device_count':0}
    summary={'backend':CFG.judge_backend,'model_id':CFG.judge_model_id,'revision':records[0].get('judge_model_revision') if records else None,
      'quantization':records[0].get('judge_quantization') if records else None,'prompt_version':records[0].get('judge_prompt_version') if records else None,
      'prompt_sha256':records[0].get('judge_prompt_sha256') if records else None,'raw_counts_by_file':raw_by_path,
      'n_raw':sum(raw_by_path.values()),'n_unique_available':n_unique_available,'n_unique_scored':len(items),'n_unique_question_response':len(items),'deduplicated':sum(raw_by_path.values())-n_unique_available,
      'margin_summary_by_condition':margins,'judge_cache_path':str(cache_path),'judge_cache_sha256':sha,
      'batch_size_items':a.batch_size,'elapsed_seconds':time.perf_counter()-t,'items_per_second':len(records)/max(time.perf_counter()-t,1e-9),'torch':_version('torch'),
      'transformers':_version('transformers'),'bitsandbytes':_version('bitsandbytes'),'cuda':cuda}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
