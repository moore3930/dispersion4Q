from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import numpy as np

model_name_or_path = "MODEL_NAME_OR_PATH"
cache_dir = "CACHE_DIR"
save_dir = "SAVE_DIR"
lora_dir = "LORA_DIR"

config = AutoConfig.from_pretrained(model_name_or_path,
                                    cache_dir=cache_dir,
                                    )

config.use_cache = False


model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
)

# Load the lora module
print(f"Loading Lora: {lora_dir}")
model = PeftModel.from_pretrained(model, lora_dir)
model = model.merge_and_unload()    

model.save_pretrained(save_dir)