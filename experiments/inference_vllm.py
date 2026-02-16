import os
import re
import shutil
import time
import random

from transformers import AutoTokenizer

LANG_NAME = {
    "en": "English",
    "zh": "Chinese",
    "ar": "Arabic",
    "de": "German",
    "cs": "Czech",
    "ru": "Russian",
    "is": "Icelandic",
    "es": "Spanish",
    "hi": "Hindi",
    "ja": "Japanese",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ko": "Korean",
}


def create_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def load_bitext(dataset_name, split, lang_pairs):
    assert split in ["train", "valid", "test", "ref"], f"Unknown split: {split}"
    random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    dir_name = os.path.join(repo_root, "src", "llama_recipes", "customer_data", dataset_name)

    output_dataset = []
    for lp in lang_pairs:
        src, tgt = lp.split("-")
        pair_name = f"{src}-{tgt}"
        src_name = f"{split}.{src}-{tgt}.{src}"
        tgt_name = f"{split}.{src}-{tgt}.{tgt}"

        with open(os.path.join(dir_name, split, pair_name, src_name)) as src_fin, open(
            os.path.join(dir_name, split, pair_name, tgt_name)
        ) as tgt_fin:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                row = {
                    "id": str(random.randint(-2 ** 31, 2 ** 31 - 1)),
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "src": src_sent.strip(),
                    "tgt": tgt_sent.strip(),
                }
                output_dataset.append(row)
    return output_dataset


def _build_prompts(tokenizer, dataset_name, lang_pair):
    dataset = load_bitext(dataset_name, "test", [lang_pair])
    prompt_template = (
        "Translate the following text from {src_lang} into {tgt_lang}.\n"
        "{src_lang}: {src}\n"
        "{tgt_lang}:"
    )

    prompts = []
    for sample in dataset:
        src = sample["src_lang"]
        tgt = sample["tgt_lang"]
        base_prompt = prompt_template.format(
            src_lang=LANG_NAME[src],
            tgt_lang=LANG_NAME[tgt],
            src=sample["src"],
        )
        messages = [{"role": "user", "content": base_prompt}]
        if getattr(tokenizer, "chat_template", None) is not None:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = base_prompt
        if tokenizer.bos_token:
            prompt = tokenizer.bos_token + prompt
        prompts.append(prompt)
    return prompts


def _generate_texts(llm, prompts, sampling_params, lora_request=None):
    if lora_request is None:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    else:
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_request, use_tqdm=True
        )

    texts = []
    for output in outputs:
        text = output.outputs[0].text if output.outputs else ""
        texts.append(re.split(r"[\t\n]", text)[0])
    return texts


def run_vllm_inference(
    model_name,
    test_config,
    peft_model=None,
    max_new_tokens=1000,
    seed=42,
    do_sample=True,
    top_p=1.0,
    temperature=1.0,
    top_k=50,
    repetition_penalty=1.0,
    lang_pairs=None,
    output_dir=None,
):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ModuleNotFoundError as exc:
        if exc.name == "pyairports":
            raise ModuleNotFoundError(
                "vLLM dependency missing: 'pyairports'. Install it in the active env, "
                "for example: `uv pip install pyairports` (or `pip install pyairports`)."
            ) from exc
        raise

    beam_size = getattr(test_config, "beam_size", 1)
    if beam_size != 1:
        raise ValueError(
            f"vLLM backend only supports beam_size=1 in this pipeline, got beam_size={beam_size}."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id >= 0:
        terminators.append(eot_id)

    # Avoid torch.compile/Triton JIT path that may require system Python headers (Python.h)
    # on cluster nodes where python-devel is unavailable.
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "enforce_eager": True,
    }
    if peft_model:
        raise ValueError(
            "vLLM inference in this pipeline does not support runtime --peft_model. "
            "Merge LoRA into the base model first (e.g. via "
            "`experiments/merge_lora_for_vllm.py`) and run vLLM on the merged model."
        )
    llm = LLM(**llm_kwargs)
    lora_request = None

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else -1,
        repetition_penalty=repetition_penalty,
        stop_token_ids=terminators,
        seed=seed,
    )

    if lang_pairs is not None:
        lang_pairs = lang_pairs.split(",")
    else:
        lang_pairs = test_config.lang_pairs

    for lang_pair in lang_pairs:
        print(f"Processing {lang_pair} ...", flush=True)
        prompts = _build_prompts(tokenizer, test_config.dataset, lang_pair)
        print(f"--> Test Set Length = {len(prompts)}", flush=True)

        start = time.perf_counter()
        results = _generate_texts(llm, prompts, sampling_params, lora_request=lora_request)
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms", flush=True)

        src, tgt = lang_pair.split("-")
        create_clean_dir(os.path.join(output_dir, lang_pair))
        output_file = os.path.join(output_dir, lang_pair, f"hyp.{src}-{tgt}.{tgt}")
        with open(output_file, "w") as fout:
            for line in results:
                fout.write(line.strip() + "\n")
