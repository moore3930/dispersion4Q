import os
import re
import shutil
import time
import random

from transformers import AutoTokenizer
from llama_recipes.inference.translation_prompt_utils import (
    build_translation_prompt,
    load_model_prompt_config,
)


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


def _build_prompts(tokenizer, dataset_name, lang_pair, model_name):
    dataset = load_bitext(dataset_name, "test", [lang_pair])
    prompt_config = load_model_prompt_config(model_name)

    prompts = []
    for sample in dataset:
        prompt = build_translation_prompt(
            tokenizer=tokenizer,
            src_lang_code=sample["src_lang"],
            tgt_lang_code=sample["tgt_lang"],
            src_text=sample["src"],
            model_name=model_name,
            prompt_config=prompt_config,
            add_generation_prompt=True,
        )
        if tokenizer.bos_token:
            prompt = tokenizer.bos_token + prompt
        prompts.append(prompt)
    return prompts


def _clean_generated_text(text):
    return re.split(r"[\t\n]", text)[0]


def _generate_texts_sampling(llm, prompts, sampling_params, lora_request=None):
    if lora_request is None:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    else:
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_request, use_tqdm=True
        )

    texts = []
    for output in outputs:
        text = output.outputs[0].text if output.outputs else ""
        texts.append(_clean_generated_text(text))
    return texts


def _generate_texts_beam(llm, prompts, beam_search_params, lora_request=None):
    beam_prompts = []
    for prompt in prompts:
        if isinstance(prompt, str):
            beam_prompts.append({"prompt": prompt})
        else:
            beam_prompts.append(prompt)

    if lora_request is None:
        outputs = llm.beam_search(beam_prompts, beam_search_params, use_tqdm=True)
    else:
        outputs = llm.beam_search(
            beam_prompts,
            beam_search_params,
            lora_request=lora_request,
            use_tqdm=True,
        )

    texts = []
    for output in outputs:
        if output.sequences:
            text = output.sequences[0].text or ""
        else:
            text = ""
        texts.append(_clean_generated_text(text))
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
    max_model_len=None,
    max_num_batched_tokens=None,
    max_num_seqs=None,
    gpu_memory_utilization=None,
    tensor_parallel_size=1,
    swap_space=4.0,
    cpu_offload_gb=0.0,
    enforce_eager=True,
):
    try:
        from vllm import LLM, SamplingParams
        try:
            from vllm import BeamSearchParams
        except ImportError:
            from vllm.sampling_params import BeamSearchParams
    except ModuleNotFoundError as exc:
        if exc.name == "pyairports":
            raise ModuleNotFoundError(
                "vLLM dependency missing: 'pyairports'. Install it in the active env, "
                "for example: `uv pip install pyairports` (or `pip install pyairports`)."
            ) from exc
        raise

    beam_size = int(getattr(test_config, "beam_size", 1))
    if beam_size < 1:
        raise ValueError(
            f"Invalid beam_size={beam_size}. beam_size must be >= 1."
        )
    use_beam_search = beam_size > 1
    if use_beam_search:
        # Beam-search internals in this environment can trigger torch.compile/
        # Inductor JIT and require Python.h on compute nodes. Force eager path.
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch._dynamo

            torch._dynamo.config.disable = True
        except Exception:
            pass

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
        "enforce_eager": True if enforce_eager is None else bool(enforce_eager),
        "tensor_parallel_size": int(tensor_parallel_size),
        "swap_space": float(swap_space),
        "cpu_offload_gb": float(cpu_offload_gb),
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = int(max_num_seqs)
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
    if peft_model:
        raise ValueError(
            "vLLM inference in this pipeline does not support runtime --peft_model. "
            "Merge LoRA into the base model first (e.g. via "
            "`experiments/merge_lora_for_vllm.py`) and run vLLM on the merged model."
        )
    llm = LLM(**llm_kwargs)
    lora_request = None

    if use_beam_search:
        if not hasattr(llm, "beam_search"):
            raise RuntimeError(
                "beam_size > 1 was requested, but this vLLM version does not expose "
                "LLM.beam_search. Upgrade vLLM or set beam_size=1."
            )
        beam_search_params = BeamSearchParams(
            beam_width=beam_size,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            length_penalty=float(getattr(test_config, "length_penalty", 1.0)),
        )
    else:
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
        prompts = _build_prompts(tokenizer, test_config.dataset, lang_pair, model_name)
        print(f"--> Test Set Length = {len(prompts)}", flush=True)
        if prompts:
            print("--> Resolved prompt for first sample:", flush=True)
            print(prompts[0], flush=True)

        start = time.perf_counter()
        if use_beam_search:
            results = _generate_texts_beam(
                llm,
                prompts,
                beam_search_params,
                lora_request=lora_request,
            )
        else:
            results = _generate_texts_sampling(
                llm,
                prompts,
                sampling_params,
                lora_request=lora_request,
            )
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms", flush=True)

        src, tgt = lang_pair.split("-")
        create_clean_dir(os.path.join(output_dir, lang_pair))
        output_file = os.path.join(output_dir, lang_pair, f"hyp.{src}-{tgt}.{tgt}")
        with open(output_file, "w") as fout:
            for line in results:
                fout.write(line.strip() + "\n")
