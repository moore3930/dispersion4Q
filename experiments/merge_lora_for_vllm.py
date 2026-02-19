import os
import fire
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
):
    if os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"Merged model already exists at {output_dir}, skipping merge.", flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    print(f"Loading base model: {base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"Loading adapter: {adapter_dir}", flush=True)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)

    print("Merging adapter into base model...", flush=True)
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}", flush=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print("Merge completed.", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
