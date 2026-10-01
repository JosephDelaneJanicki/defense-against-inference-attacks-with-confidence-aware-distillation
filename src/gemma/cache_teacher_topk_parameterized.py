import argparse, json
from pathlib import Path
import numpy as np, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="google/gemma-3-4b-pt"; MAX_LENGTH=256; TOP_K=64; BATCH_SIZE=1
THRESHOLDS=[0.85,0.95]

def load_jsonl(path):
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for i,line in enumerate(f):
            r=json.loads(line)
            rows.append({"index":i,"sha256":r.get("sha256",f"row_{i}"),"text":r["text"]})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",type=Path,required=True)
    ap.add_argument("--corpus",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    cache_path=a.output_dir/"teacher_topk_cache.pt"
    summary_path=a.output_dir/"teacher_topk_summary.json"
    rows=load_jsonl(a.corpus)
    if len(rows)!=5000: raise RuntimeError(f"Expected 5000 corpus rows, got {len(rows)}")
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(base,a.teacher).cuda().eval()
    cached=[]; confs=[]; masses=[]; total=0; above={t:0 for t in THRESHOLDS}
    for start in range(0,len(rows),BATCH_SIZE):
        br=rows[start:start+BATCH_SIZE]
        enc=tok([r["text"] for r in br],return_tensors="pt",padding=True,
                truncation=True,max_length=MAX_LENGTH,add_special_tokens=True)
        ids=enc["input_ids"].cuda(); am=enc["attention_mask"].cuda()
        with torch.inference_mode():
            out=model(input_ids=ids,attention_mask=am)
            sl=out.logits[:,:-1,:].float(); labels=ids[:,1:]; mask=am[:,1:].bool()
            kl,ki=torch.topk(sl,k=TOP_K,dim=-1)
            fullz=torch.logsumexp(sl,dim=-1)
            conf=torch.exp(kl[...,0]-fullz)
            kmass=torch.exp(torch.logsumexp(kl,dim=-1)-fullz)
        for i,s in enumerate(br):
            m=mask[i]
            ci=conf[i][m].float().cpu(); mi=kmass[i][m].float().cpu()
            cached.append({
                "index":s["index"],"sha256":s["sha256"],
                "input_ids":ids[i][am[i].bool()].to(torch.int32).cpu(),
                "target_ids":labels[i][m].to(torch.int32).cpu(),
                "topk_ids":ki[i][m].to(torch.int32).cpu(),
                "topk_logits":kl[i][m].to(torch.float16).cpu(),
                "top1_confidence":ci,"topk_mass":mi,
            })
            cn=ci.numpy(); mn=mi.numpy(); confs.append(cn); masses.append(mn); total+=len(cn)
            for t in THRESHOLDS: above[t]+=int(np.sum(cn>t))
        done=min(start+BATCH_SIZE,len(rows))
        if done==1 or done%500==0: print(f"Cached {done}/{len(rows)}")
    c=np.concatenate(confs); m=np.concatenate(masses); ps=[50,75,90,95,99]
    summary={"experiment":"gemma_shadow_teacher_topk_cache","base_model":MODEL_ID,
      "teacher_adapter":str(a.teacher),"corpus":str(a.corpus),"num_sequences":len(rows),
      "max_length":MAX_LENGTH,"top_k":TOP_K,"total_prediction_tokens":total,
      "teacher_confidence":{"mean":float(c.mean()),"std":float(c.std()),
        "percentiles":{f"p{x}":float(np.percentile(c,x)) for x in ps},
        "fraction_above_threshold":{str(t):above[t]/total for t in THRESHOLDS}},
      "topk_probability_mass":{"mean":float(m.mean()),"std":float(m.std()),
        "percentiles":{f"p{x}":float(np.percentile(m,x)) for x in ps}}}
    torch.save({"metadata":summary,"sequences":cached},cache_path)
    summary_path.write_text(json.dumps(summary,indent=2))
    print("[CACHE COMPLETE]",cache_path)

if __name__=="__main__": main()
