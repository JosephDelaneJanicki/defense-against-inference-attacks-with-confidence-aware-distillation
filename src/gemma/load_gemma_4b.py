import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "google/gemma-3-4b-pt"
DEVICE = "cuda"


def gpu_memory(label):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n[{label}]")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Peak:      {peak:.2f} GB")


def main():
    print(f"Loading: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Requested dtype: BF16")

    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(DEVICE)

    model.eval()

    print("\nModel loaded.")
    print("Parameter dtype:", next(model.parameters()).dtype)
    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f} B",
    )

    gpu_memory("After model load")

    text = (
        "Membership inference attacks attempt to determine whether "
        "a particular example was used to train a machine learning model."
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    print("\nTokens:", inputs["input_ids"].shape[1])

    with torch.inference_mode():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"],
        )

    print("Loss:", outputs.loss.item())
    print("Logits shape:", tuple(outputs.logits.shape))
    print("Logits dtype:", outputs.logits.dtype)

    gpu_memory("After forward pass")

    print("\nSUCCESS: Gemma 3 4B PT is running in BF16.")


if __name__ == "__main__":
    main()