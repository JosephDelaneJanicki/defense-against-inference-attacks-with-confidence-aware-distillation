import argparse,json,math,random,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model

MODEL_ID="google/gemma-3-4b-pt"; MAX_LENGTH=256
R=8; ALPHA=16; DROPOUT=.05; EPOCHS=2; LR=2e-4; GRAD_ACCUM=8

class CacheDS(Dataset):
    def __init__(self,p):
        x=torch.load(p,map_location="cpu",weights_only=False); self.meta=x["metadata"]; self.rows=x["sequences"]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]
class ValDS(Dataset):
    def __init__(self,p,tok):
        self.rows=[]
        with p.open("r",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line); e=tok(r["text"],truncation=True,max_length=MAX_LENGTH,
                    padding=False,add_special_tokens=True)
                self.rows.append({"input_ids":torch.tensor(e["input_ids"]),
                                  "attention_mask":torch.tensor(e["attention_mask"])})
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]
def collate(b):
    r=b[0]
    return {k:r[k].long().unsqueeze(0) if k in ("input_ids","topk_ids")
            else r[k].float().unsqueeze(0) for k in ("input_ids","topk_ids","topk_logits","topk_mass")}
def kd(student,ids,tl,mass):
    s=student[:,:-1,:].float()
    tp=F.softmax(tl,dim=-1)*mass.clamp(0,1).unsqueeze(-1); other=(1-mass).clamp(0,1)
    slp=F.log_softmax(s,dim=-1); gathered=torch.gather(slp,-1,ids)
    sother=(1-gathered.exp().sum(-1)).clamp(1e-8,1)
    return (-(tp*gathered).sum(-1)-other*torch.log(sother)).mean()
def eval_nll(model,ds):
    model.eval(); total=0.; n=0
    with torch.inference_mode():
        for r in ds:
            ids=r["input_ids"].unsqueeze(0).cuda(); am=r["attention_mask"].unsqueeze(0).cuda()
            o=model(input_ids=ids,attention_mask=am); logits=o.logits[:,:-1,:].float()
            losses=F.cross_entropy(logits.transpose(1,2),ids[:,1:],reduction="none")
            v=losses[am[:,1:].bool()]; total+=float(v.sum()); n+=v.numel()
    model.train(); return total/n
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--validation",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--seed",type=int,required=True); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    a.output_dir.mkdir(parents=True,exist_ok=True); tok=AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tr=CacheDS(a.cache); va=ValDS(a.validation,tok)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(); model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=R,lora_alpha=ALPHA,lora_dropout=DROPOUT,bias="none",
        task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj"])).cuda()
    dl=DataLoader(tr,batch_size=1,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(a.seed))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR)
    total_updates=math.ceil(len(dl)/GRAD_ACCUM)*EPOCHS
    sch=torch.optim.lr_scheduler.LinearLR(opt,start_factor=1,end_factor=1e-3,total_iters=total_updates)
    initial=eval_nll(model,va); print("Initial validation NLL:",initial)
    opt.zero_grad(set_to_none=True); hist=[]; update=0; start=time.time()
    for epoch in range(EPOCHS):
        eh=[]
        for step,b in enumerate(dl):
            ids=b["input_ids"].cuda(); ti=b["topk_ids"].cuda(); tl=b["topk_logits"].cuda(); tm=b["topk_mass"].cuda()
            o=model(input_ids=ids,attention_mask=torch.ones_like(ids)); loss=kd(o.logits,ti,tl,tm)
            val=float(loss.detach()); hist.append(val); eh.append(val); (loss/GRAD_ACCUM).backward()
            if (step+1)%GRAD_ACCUM==0 or step+1==len(dl):
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); update+=1
                if update%25==0: print(f"epoch {epoch+1}/{EPOCHS} update {update}/{total_updates} KD={np.mean(hist[-25:]):.6f}")
        print(f"Epoch {epoch+1} KD={np.mean(eh):.6f} validation NLL={eval_nll(model,va):.6f}")
    final=eval_nll(model,va); elapsed=time.time()-start
    ad=a.output_dir/"adapter"; model.save_pretrained(ad); tok.save_pretrained(ad)
    summary={"experiment":"gemma_raw_soft_shadow_student","seed":a.seed,"base_model":MODEL_ID,
      "teacher_cache":str(a.cache),"train_sequences":len(tr),"validation_sequences":len(va),
      "epochs":EPOCHS,"learning_rate":LR,"gradient_accumulation":GRAD_ACCUM,
      "initial_validation_nll":initial,"final_validation_nll":final,
      "mean_kd_loss":float(np.mean(hist)),"training_seconds":elapsed,
      "peak_allocated_gb":torch.cuda.max_memory_allocated()/1024**3}
    (a.output_dir/"training_summary.json").write_text(json.dumps(summary,indent=2))
    print("[TRAIN COMPLETE]",ad); print("Validation NLL:",initial,"->",final)
if __name__=="__main__": main()
