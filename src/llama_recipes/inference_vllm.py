from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "haoranxu/ALMA-7B-Pretrain"  # Example: "meta-llama/Llama-2-7b-hf"
lora_model_path = "haoranxu/ALMA-7B-Pretrain-LoRA"
merged_model_path = "/gpfs/work4/0/gus20642/dwu18/calibration/ALMA-1"

# Load base model and merge with LoRA
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.merge_and_unload()

# Save the merged model and tokenizer
merged_model_path = "/gpfs/work4/0/gus20642/dwu18/calibration/ALMA-1"
model.save_pretrained(merged_model_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)

# Load the merged model in vLLM
llm = LLM(model=merged_model_path, tensor_parallel_size=1)

# Define sampling parameters
# sampling_params = SamplingParams(temperature=1.0, max_tokens=256, top_p=0.9)
# sampling_params = SamplingParams(temperature=1.0, max_tokens=256, top_p=0.9, n=10)
# output = llm.generate("Translate this from English to Chinese:\nEnglish: I'm sorry that your order is running late.\nChinese:", sampling_params)

# Define beam search parameters
params = BeamSearchParams(beam_width=5, max_tokens=50)
output = llm.beam_search(["Translate this from English to Chinese:\nEnglish: I'm sorry that your order is running late.\nChinese:"], params)

print("==== DEBUG ====")
print(output)
print(output[0].outputs[0].text, flush=True)