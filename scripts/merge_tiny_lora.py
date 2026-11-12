"""Step 0.2 helper: train a 20-step LoRA on 0.8B (same tiny corpus as arch_check), merge it,
save a full checkpoint for the vLLM merged-LoRA serving test. Also exercises common.models.merge_and_save."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.arch_check import load  # noqa: E402

OUT = Path("/workspace/artifacts/0.2/tiny_lora_merged")


def main():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model, tok, proc, *_ = load("Qwen/Qwen3.5-0.8B")
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n" * 8
    batch = tok([text] * 4, return_tensors="pt").to("cuda")
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-4)
    for _ in range(20):
        loss = pm(**batch, labels=batch["input_ids"]).loss
        loss.backward(); opt.step(); opt.zero_grad()
    merged = pm.merge_and_unload()
    OUT.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    if proc is not None:
        proc.save_pretrained(OUT)
    print("saved", OUT, "final loss", loss.item())


if __name__ == "__main__":
    main()
