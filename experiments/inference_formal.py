# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import shutil
import sys
import re
import time
import subprocess
import json
from pathlib import Path
from types import SimpleNamespace
import fire


def _cli_flag_was_set(flag_name: str) -> bool:
    alt_flag = flag_name.replace("_", "-")
    for arg in sys.argv[1:]:
        if (
            arg == flag_name
            or arg.startswith(flag_name + "=")
            or arg == alt_flag
            or arg.startswith(alt_flag + "=")
        ):
            return True
    return False


def _load_vllm_model_profile(model_name: str):
    model_configs_dir = Path(__file__).resolve().parent / "model_configs"
    if not model_configs_dir.is_dir():
        return None

    model_name_l = str(model_name).lower()
    best_profile = None
    best_score = -1

    for config_file in sorted(model_configs_dir.glob("*.json")):
        with config_file.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        match_substrings = profile.get("match_substrings", [])
        for match_token in match_substrings:
            token = str(match_token).lower()
            if token and token in model_name_l and len(token) > best_score:
                best_profile = dict(profile)
                best_profile["_config_file"] = str(config_file)
                best_score = len(token)

    return best_profile


def create_clean_dir(path):
    """
    Create a clean directory. If the directory exists, remove it first.
    :param path: Path of the directory to create.
    """
    # Remove the directory if it exists
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create the directory
    os.makedirs(path)


def main(
    model_name,
    inference_backend: str = "hf",  # Options: hf, vllm
    vllm_env_path: str = None,  # Required when inference_backend=vllm
    peft_model: str = None,
    quantization: str = None,  # Options: 4bit, 8bit
    max_new_tokens=1000,  # The maximum numbers of tokens to generate
    prompt_file: str = None,
    seed: int = 42,  # seed value for reproducibility
    do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True,
    # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0,
    # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,
    # [optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool = False,
    max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False,
    # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    share_gradio: bool = False,  # Enable endpoint creation for gradio.live
    lang_pairs: str = None,
    output_dir: str = None,
    vllm_max_model_len: int = None,
    vllm_max_num_batched_tokens: int = None,
    vllm_max_num_seqs: int = None,
    vllm_gpu_memory_utilization: float = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_swap_space: float = 4.0,
    vllm_cpu_offload_gb: float = 0.0,
    vllm_enforce_eager: bool = None,
    **kwargs,
):
    if inference_backend not in {"hf", "vllm"}:
        raise ValueError(
            f"Unknown inference_backend={inference_backend}. Expected one of: hf, vllm."
        )

    if inference_backend == "vllm":
        if not vllm_env_path:
            raise ValueError(
                "vLLM backend requires --vllm_env_path <path-to-venv>."
            )
        profile = _load_vllm_model_profile(model_name)
        if profile:
            vllm_profile = profile.get("vllm", {})
            if not _cli_flag_was_set("--vllm_max_model_len") and "max_model_len" in vllm_profile:
                vllm_max_model_len = vllm_profile["max_model_len"]
            if (
                not _cli_flag_was_set("--vllm_max_num_batched_tokens")
                and "max_num_batched_tokens" in vllm_profile
            ):
                vllm_max_num_batched_tokens = vllm_profile["max_num_batched_tokens"]
            if not _cli_flag_was_set("--vllm_max_num_seqs") and "max_num_seqs" in vllm_profile:
                vllm_max_num_seqs = vllm_profile["max_num_seqs"]
            if (
                not _cli_flag_was_set("--vllm_gpu_memory_utilization")
                and "gpu_memory_utilization" in vllm_profile
            ):
                vllm_gpu_memory_utilization = vllm_profile["gpu_memory_utilization"]
            if (
                not _cli_flag_was_set("--vllm_tensor_parallel_size")
                and "tensor_parallel_size" in vllm_profile
            ):
                vllm_tensor_parallel_size = vllm_profile["tensor_parallel_size"]
            if not _cli_flag_was_set("--vllm_swap_space") and "swap_space" in vllm_profile:
                vllm_swap_space = vllm_profile["swap_space"]
            if (
                not _cli_flag_was_set("--vllm_cpu_offload_gb")
                and "cpu_offload_gb" in vllm_profile
            ):
                vllm_cpu_offload_gb = vllm_profile["cpu_offload_gb"]
            if (
                not _cli_flag_was_set("--vllm_enforce_eager")
                and "enforce_eager" in vllm_profile
            ):
                vllm_enforce_eager = vllm_profile["enforce_eager"]
            if (
                not _cli_flag_was_set("--max_new_tokens")
                and "max_new_tokens" in vllm_profile
            ):
                max_new_tokens = int(vllm_profile["max_new_tokens"])

            for env_key, env_value in profile.get("env", {}).items():
                os.environ.setdefault(str(env_key), str(env_value))
            print(f"Using vLLM model profile: {profile.get('_config_file')}", flush=True)

        # Torch compile is enabled by default. Set DISPERSION4Q_DISABLE_TORCH_COMPILE=1
        # to force eager mode for environments that cannot compile kernels.
        disable_torch_compile = os.environ.get("DISPERSION4Q_DISABLE_TORCH_COMPILE", "0") == "1"
        if disable_torch_compile:
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
        else:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
            os.environ.pop("TORCHDYNAMO_DISABLE", None)
        vllm_python = os.path.abspath(os.path.join(vllm_env_path, "bin", "python"))
        if not os.path.exists(vllm_python):
            raise FileNotFoundError(
                f"vLLM Python not found at '{vllm_python}'. "
                f"Create the env first or pass --vllm_env_path <path>."
            )
        if os.environ.get("DISPERSION4Q_VLLM_REEXEC") != "1":
            env = os.environ.copy()
            env["DISPERSION4Q_VLLM_REEXEC"] = "1"
            if disable_torch_compile:
                env["TORCH_COMPILE_DISABLE"] = "1"
                env["TORCHDYNAMO_DISABLE"] = "1"
            else:
                env.pop("TORCH_COMPILE_DISABLE", None)
                env.pop("TORCHDYNAMO_DISABLE", None)
            subprocess.run([vllm_python] + sys.argv, check=True, env=env)
            return

        dataset_name = kwargs.get("dataset")
        if not dataset_name:
            raise ValueError("vLLM backend requires --dataset <dataset_name>.")
        beam_size = int(kwargs.get("beam_size", 1))
        configured_lang_pairs = (
            lang_pairs.split(",") if isinstance(lang_pairs, str) and lang_pairs else []
        )
        test_config = SimpleNamespace(
            dataset=dataset_name,
            beam_size=beam_size,
            length_penalty=float(length_penalty),
            lang_pairs=configured_lang_pairs,
        )

        from inference_vllm import run_vllm_inference
        return run_vllm_inference(
            model_name=model_name,
            test_config=test_config,
            peft_model=peft_model,
            max_new_tokens=max_new_tokens,
            seed=seed,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            lang_pairs=lang_pairs,
            output_dir=output_dir,
            max_model_len=vllm_max_model_len,
            max_num_batched_tokens=vllm_max_num_batched_tokens,
            max_num_seqs=vllm_max_num_seqs,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            tensor_parallel_size=vllm_tensor_parallel_size,
            swap_space=vllm_swap_space,
            cpu_offload_gb=vllm_cpu_offload_gb,
            enforce_eager=vllm_enforce_eager,
        )

    import torch
    from tqdm import tqdm
    from accelerate.utils import is_xpu_available
    from llama_recipes.inference.model_utils import load_model, load_peft_model
    from transformers import AutoTokenizer
    from llama_recipes.utils.dataset_utils import (
        get_translation_dataset,
        get_prefernce_dataset,
    )
    from llama_recipes.configs import (
        fsdp_config as FSDP_CONFIG,
        train_config as TRAIN_CONFIG,
    )
    from llama_recipes.utils.config_utils import (
        get_dataloader_kwargs,
        update_config,
    )

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # Update the configuration for the training and sharding process
    test_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((test_config, fsdp_config), **kwargs)
    # dataset_config = generate_dataset_config(test_config, kwargs)

    model = load_model(model_name, quantization, use_fast_kernels, **kwargs)

    if test_config.preload_peft_dir is not None:
        # merge peft into backbone, may not 100% aligned
        print("Load and merge peft...")
        model = load_peft_model(model, test_config.preload_peft_dir)
        model = model.merge_and_unload()

    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # TODO, batch inference
    def inference_new(
            dataloader,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            config,
            pbar,
            **kwargs,
    ):
        output = []
        for step, batch in enumerate(dataloader):
            # TODO, dirty
            batch.pop('labels')
            if is_xpu_available():
                batch = {k: v.to("xpu") for k, v in batch.items()}
            else:
                batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.no_grad():
                batch_output = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    min_length=min_length,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_beams=config.beam_size,
                    eos_token_id=terminators,
                    **kwargs,
                )

                # prompt_len = batch['attention_mask'].sum(-1)

                batch_len = batch['input_ids'].shape[-1]
                batch_output = batch_output[:, batch_len:]
                batch_output = [tokenizer.decode(output, skip_special_tokens=True) for output in batch_output]

                # replace \n with \t when hallucinating
                # batch_output = [sent.replace("\n", "\t").strip() for sent in batch_output]

                # remove \n with \t when hallucinating
                batch_output = [re.split(r'[\t\n]', sent)[0] for sent in batch_output]

                output += batch_output
            pbar.update(1)
        return output

    # TODO, inference for each dataset
    output = {}

    if lang_pairs is not None:
        lang_pairs = lang_pairs.split(',')
    else:
        lang_pairs = test_config.lang_pairs

    for lang_pair in lang_pairs:
        # Get test data
        print("Processing {} ...".format(lang_pair), flush=True)
        if test_config.dataset == "haoranxu/ALMA-R-Preference":
            dataset_test = get_prefernce_dataset(
                tokenizer,
                test_config,
                test_config.dataset,
                mode="infer",
                subset_name=test_config.subset_name,
                split="train",
                lang_pairs=[lang_pair],
                filter="reference"
            )
        else:
            dataset_test = get_translation_dataset(
                tokenizer,
                test_config.dataset,
                mode="infer",
                split="test",
                lang_pairs=[lang_pair]
            )
        print(f"--> Test Set Length = {len(dataset_test)}", flush=True)

        test_dl_kwargs = get_dataloader_kwargs(
            test_config, dataset_test, tokenizer, "infer"
        )

        # Create DataLoaders for inference
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            **test_dl_kwargs,
        )
        print(f"--> Num of Testing Set Batches loaded = {len(test_dataloader)}", flush=True)

        start = time.perf_counter()
        total_length = len(test_dataloader)
        pbar = tqdm(colour="blue", desc=f"Inference", total=total_length, dynamic_ncols=True)
        results = inference_new(test_dataloader, temperature, top_p, top_k, max_new_tokens, test_config, pbar=pbar)
        pbar.close()
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms", flush=True)
        output[lang_pair] = results

        # dump results
        src, tgt = lang_pair.split("-")
        create_clean_dir(os.path.join(output_dir, lang_pair))
        output_file = os.path.join(output_dir, lang_pair, "hyp.{}-{}.{}".format(src, tgt, tgt))
        with open(output_file, 'w') as fout:
            for line in results:
                fout.write(line.strip() + "\n")


if __name__ == "__main__":
    fire.Fire(main)
