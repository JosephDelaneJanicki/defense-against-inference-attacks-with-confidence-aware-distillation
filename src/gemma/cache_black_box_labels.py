import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_ID = "google/gemma-3-4b-pt"
MAX_LENGTH = 256


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--student-adapter", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    student_adapter = Path(args.student_adapter)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BLACK-BOX HARD-LABEL CACHE")
    print("=" * 72)

    print("Released student:", student_adapter)
    print("Query corpus:", corpus_path)
    print("Output:", output_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(
        base,
        student_adapter,
    )

    model.cuda()
    model.eval()

    rows = []

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print("Examples:", len(rows))

    cached = []

    with torch.inference_mode():

        for i, row in enumerate(rows):

            encoded = tokenizer(
                row["text"],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

            # logits[:, t] predicts token t+1
            pseudo_targets = (
                logits[:, :-1, :]
                .argmax(dim=-1)
                .squeeze(0)
                .cpu()
            )

            source_ids = input_ids.squeeze(0).cpu()

            cached.append({
                "input_ids": source_ids,
                "pseudo_targets": pseudo_targets,
            })

            if (i + 1) % 250 == 0:
                print(f"Cached {i + 1}/{len(rows)}")

    torch.save(
        {
            "model_id": MODEL_ID,
            "student_adapter": str(student_adapter),
            "corpus": str(corpus_path),
            "max_length": MAX_LENGTH,
            "label_type": "hard_argmax_black_box",
            "examples": cached,
        },
        output_path,
    )

    print("\nSaved:", output_path)
    print("BLACK-BOX LABEL CACHE COMPLETE")


if __name__ == "__main__":
    main()