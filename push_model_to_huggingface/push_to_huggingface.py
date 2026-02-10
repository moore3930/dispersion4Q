from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi

local_dir = "MODEL_PATH"
repo_id = "REPO_ID"

# Load local files
model = AutoModel.from_pretrained(local_dir)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

# Push to hub
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
