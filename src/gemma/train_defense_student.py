#!/usr/bin/env python3
"""Train one Gemma 3 4B PT LoRA student using a fixed Top-64 teacher cache.

Definitions:
  raw_soft: original Top-64 + residual-bucket cross entropy.
  temp_soft: T=2 softmax over cached Top-64 logits, with ORIGINAL residual mass.
             This is a Top-64 conditional-temperature approximation, not full-vocab T.
  clip_soft: cap teacher top-1 probability at 0.85; redistribute excess to
             other Top-64 tokens proportionally; preserve residual mass.
  confidence_filtered: exclude top 20% of sequences by mean cached teacher
             top-1 confidence, then use original raw-soft KD on remaining 80%.
"""
import argparse, json, math, random, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = 'google/gemma-3-4b-pt'
MAX_LENGTH = 256
R, ALPHA, DROPOUT, EPOCHS, LR, GRAD_ACCUM = 8, 16, 0.05, 2, 2e-4, 8
DEFENSES = ('raw_soft', 'temp_soft', 'clip_soft', 'confidence_filtered')

class CacheDS(Dataset):
    def __init__(self, path, defense, filter_fraction):
        x = torch.load(path, map_location='cpu', weights_only=False)
        self.meta = x['metadata']
        rows = x['sequences']
        if not rows:
            raise ValueError('Teacher cache contains no sequences')
        self.original_count = len(rows)
        self.excluded = []
        if defense == 'confidence_filtered':
            n_drop = int(round(len(rows) * filter_fraction))
            ranked = sorted(range(len(rows)), key=lambda i: (
                -float(torch.as_tensor(rows[i]['top1_confidence']).float().mean()),
                str(rows[i].get('sha256', i))))
            remove = set(ranked[:n_drop])
            self.excluded = [str(rows[i].get('sha256', i)) for i in ranked[:n_drop]]
            rows = [row for i, row in enumerate(rows) if i not in remove]
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def collate(batch):
    if len(batch) != 1:
        raise ValueError('Expected batch_size=1')
    row = batch[0]
    ids = torch.as_tensor(row['input_ids'], dtype=torch.long)
    top_ids = torch.as_tensor(row['topk_ids'], dtype=torch.long)
    logits = torch.as_tensor(row['topk_logits'], dtype=torch.float32)
    mass = torch.as_tensor(row['topk_mass'], dtype=torch.float32)
    if ids.numel() - 1 != top_ids.shape[0] or top_ids.shape != logits.shape or mass.numel() != top_ids.shape[0]:
        raise ValueError(f'Cache token alignment mismatch: input={ids.shape}, top_ids={top_ids.shape}, logits={logits.shape}, mass={mass.shape}')
    if not torch.isfinite(logits).all() or not torch.isfinite(mass).all():
        raise ValueError('Nonfinite cached teacher values')
    if (mass < 0).any() or (mass > 1.0001).any():
        raise ValueError('Invalid cached topk_mass')
    return {'input_ids': ids.unsqueeze(0), 'topk_ids': top_ids.unsqueeze(0),
            'topk_logits': logits.unsqueeze(0), 'topk_mass': mass.unsqueeze(0)}

def teacher_probs(top_logits, mass, defense, temperature, clip_max):
    conditional = F.softmax(top_logits / (temperature if defense == 'temp_soft' else 1.0), dim=-1)
    probs = conditional * mass.clamp(0, 1).unsqueeze(-1)
    if defense == 'clip_soft':
        top = probs[..., :1]
        excess = (top - clip_max).clamp(min=0)
        others = probs[..., 1:]
        denom = others.sum(-1, keepdim=True)
        # If the other Top-K tokens have zero mass, put excess in residual bucket.
        # This case is extremely unlikely with a 64-token cache.
        redistributed = others + excess * (others / denom.clamp_min(1e-12))
        probs = torch.cat([top - excess, redistributed], dim=-1)
        if bool((excess > 0).any() and (denom <= 0).any()):
            raise ValueError('Cannot redistribute clipped mass: zero other Top-K probability')
    return probs

def kd_loss(student_logits, top_ids, top_logits, mass, defense, temperature, clip_max):
    s = student_logits[:, :-1, :].float()
    p = teacher_probs(top_logits, mass, defense, temperature, clip_max)
    slp = F.log_softmax(s, dim=-1)
    gathered = torch.gather(slp, -1, top_ids)
    student_other = (1.0 - gathered.exp().sum(-1)).clamp(1e-8, 1)
    other = (1.0 - mass.clamp(0, 1))
    return (-(p * gathered).sum(-1) - other * torch.log(student_other)).mean()

class ValDS(Dataset):
    def __init__(self, path, tokenizer):
        self.rows = []
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                enc = tokenizer(obj['text'], truncation=True, max_length=MAX_LENGTH,
                                padding=False, add_special_tokens=True)
                self.rows.append({'input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
                                  'attention_mask': torch.tensor(enc['attention_mask'], dtype=torch.long)})
        if not self.rows: raise ValueError('Empty validation set')
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def eval_nll(model, ds, device):
    model.eval(); total = 0.; n = 0
    with torch.inference_mode():
        for row in ds:
            ids = row['input_ids'].unsqueeze(0).to(device)
            mask = row['attention_mask'].unsqueeze(0).to(device)
            out = model(input_ids=ids, attention_mask=mask)
            loss = F.cross_entropy(out.logits[:, :-1, :].float().transpose(1, 2),
                                   ids[:, 1:], reduction='none')
            valid = loss[mask[:, 1:].bool()]
            total += float(valid.sum()); n += valid.numel()
    model.train()
    return total / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--validation', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--defense', choices=DEFENSES, required=True)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--clip-max', type=float, default=0.85)
    p.add_argument('--filter-fraction', type=float, default=0.20)
    a = p.parse_args()
    if a.temperature <= 0 or not (0 < a.clip_max < 1) or not (0 <= a.filter_fraction < 1):
        p.error('Invalid defense hyperparameters')
    if not a.cache.is_file() or not a.validation.is_file():
        p.error('Cache and validation paths must exist')
    if (a.output_dir / 'adapter' / 'adapter_config.json').exists():
        raise RuntimeError(f'Adapter already exists, refusing overwrite: {a.output_dir}')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda': raise RuntimeError('CUDA required for Gemma 4B training')
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    train = CacheDS(a.cache, a.defense, a.filter_fraction)
    val = ValDS(a.validation, tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT,
        bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj'])).to(device)
    loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=LR)
    updates = math.ceil(len(loader) / GRAD_ACCUM) * EPOCHS
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=1e-3, total_iters=updates)
    initial = eval_nll(model, val, device)
    print(f'[START] defense={a.defense} seed={a.seed} rows={len(train)} initial_nll={initial:.6f}', flush=True)
    opt.zero_grad(set_to_none=True); history = []; started = time.time(); update = 0
    for epoch in range(EPOCHS):
        epoch_losses = []
        for step, batch in enumerate(loader):
            ids = batch['input_ids'].to(device)
            top_ids = batch['topk_ids'].to(device)
            top_logits = batch['topk_logits'].to(device)
            mass = batch['topk_mass'].to(device)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
            loss = kd_loss(out.logits, top_ids, top_logits, mass,
                           a.defense, a.temperature, a.clip_max)
            value = float(loss.detach()); history.append(value); epoch_losses.append(value)
            (loss / GRAD_ACCUM).backward()
            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); update += 1
                if update % 25 == 0:
                    print(f'epoch {epoch+1}/{EPOCHS} update {update}/{updates} KD={np.mean(history[-25:]):.6f}', flush=True)
        print(f'epoch {epoch+1} KD={np.mean(epoch_losses):.6f} val_nll={eval_nll(model,val,device):.6f}', flush=True)
    final = eval_nll(model, val, device)
    adapter = a.output_dir / 'adapter'
    model.save_pretrained(adapter); tok.save_pretrained(adapter)
    if train.excluded:
        (a.output_dir / 'excluded_sha256.json').write_text(json.dumps(train.excluded, indent=2))
    summary = {'experiment':'gemma_defense_student', 'seed':a.seed, 'defense':a.defense,
       'base_model':MODEL_ID, 'teacher_cache':str(a.cache), 'train_sequences':len(train),
       'original_train_sequences':train.original_count, 'excluded_sequences':len(train.excluded),
       'validation_sequences':len(val), 'epochs':EPOCHS, 'learning_rate':LR,
       'gradient_accumulation':GRAD_ACCUM, 'temperature':a.temperature,
       'clip_max':a.clip_max, 'filter_fraction':a.filter_fraction,
       'temperature_method':'conditional_top64_fixed_residual_mass',
       'clip_method':'top1_cap_redistribute_within_top64_fixed_residual_mass',
       'filter_method':'exclude_top_fraction_by_sequence_mean_top1_confidence',
       'initial_validation_nll':initial, 'final_validation_nll':final,
       'mean_kd_loss':float(np.mean(history)), 'training_seconds':time.time()-started,
       'peak_allocated_gb':torch.cuda.max_memory_allocated()/1024**3}
    (a.output_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[TRAIN COMPLETE] {adapter} NLL {initial:.6f} -> {final:.6f}', flush=True)

if __name__ == '__main__': main()
