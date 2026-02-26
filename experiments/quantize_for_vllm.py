import json
import os
import shutil
from pathlib import Path

import fire
from llmcompressor import oneshot


def _load_quantization_mode(quantization_config_path: str) -> str:
    with open(quantization_config_path) as fin:
        cfg = json.load(fin)
    quant = str(cfg.get("quantization", "")).strip().lower()
    if quant not in {"awq", "gptq"}:
        raise ValueError(
            f"Unsupported quantization mode '{quant}' in {quantization_config_path}. "
            "Expected one of: awq, gptq."
        )
    return quant


def _load_model_profile(model_name: str | None, model_configs_dir: str | None = None) -> dict | None:
    if not model_name:
        return None

    if model_configs_dir:
        configs_dir = Path(model_configs_dir)
    else:
        configs_dir = Path(__file__).resolve().parent / "model_configs"
    if not configs_dir.is_dir():
        return None

    model_name_l = str(model_name).lower()
    best_profile = None
    best_score = -1

    for config_file in sorted(configs_dir.glob("*.json")):
        with config_file.open("r", encoding="utf-8") as fin:
            profile = json.load(fin)
        for token in profile.get("match_substrings", []):
            token_l = str(token).lower()
            if token_l and token_l in model_name_l and len(token_l) > best_score:
                best_profile = dict(profile)
                best_profile["_config_file"] = str(config_file)
                best_score = len(token_l)
    return best_profile


def _infer_model_name_from_dir(model_dir: str) -> str | None:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as fin:
            cfg = json.load(fin)
    except Exception:
        return None
    for key in ("_name_or_path", "name_or_path", "model_name"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    archs = cfg.get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        return archs[0]
    return None


def _is_gemma3_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    name = str(model_name).strip().lower()
    return "gemma-3" in name or "gemma3" in name


def _normalize_awq_mappings(raw_mappings: list[dict]) -> list[dict]:
    normalized = []
    for entry in raw_mappings:
        if not isinstance(entry, dict):
            continue
        smooth = entry.get("smooth_layer")
        balance = entry.get("balance_layers")
        if not isinstance(smooth, str) or not smooth.strip():
            continue
        if not isinstance(balance, list) or not all(isinstance(x, str) for x in balance):
            continue
        normalized.append(
            {
                "smooth_layer": smooth,
                "balance_layers": balance,
            }
        )
    return normalized


def _get_awq_overrides(
    model_dir: str,
    model_name_for_profile: str | None = None,
    model_configs_dir: str | None = None,
) -> tuple[dict | None, str | None]:
    model_hint = model_name_for_profile or _infer_model_name_from_dir(model_dir)
    profile = _load_model_profile(model_hint, model_configs_dir=model_configs_dir)
    if not profile:
        return None, None

    quant_profile = profile.get("quantization", {})
    if not isinstance(quant_profile, dict):
        return None, profile.get("_config_file")
    awq = quant_profile.get("awq")
    if not isinstance(awq, dict):
        return None, profile.get("_config_file")

    overrides = {}
    ignore = awq.get("ignore")
    if isinstance(ignore, list) and all(isinstance(x, str) for x in ignore):
        overrides["ignore"] = ignore

    mappings = awq.get("mappings")
    if isinstance(mappings, list):
        normalized = _normalize_awq_mappings(mappings)
        if normalized:
            overrides["mappings"] = normalized

    if not overrides:
        return None, profile.get("_config_file")
    return overrides, profile.get("_config_file")


def _build_recipe(quantization_mode: str, awq_overrides: dict | None = None) -> dict:
    group_4bit = {
        "targets": ["Linear"],
        "input_activations": None,
        "output_activations": None,
        "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": False if quantization_mode == "awq" else True,
            "strategy": "group",
            "group_size": 128,
        },
    }

    if quantization_mode == "awq":
        awq_modifier = {
            "ignore": ["lm_head"],
            "config_groups": {"group_0": group_4bit},
        }
        if awq_overrides:
            if "ignore" in awq_overrides:
                awq_modifier["ignore"] = awq_overrides["ignore"]
            if "mappings" in awq_overrides:
                awq_modifier["mappings"] = awq_overrides["mappings"]
        return {
            "quant_stage": {
                "quant_modifiers": {
                    "AWQModifier": awq_modifier
                }
            }
        }

    # GPTQ
    return {
        "quant_stage": {
            "quant_modifiers": {
                "GPTQModifier": {
                    "ignore": ["lm_head"],
                    "block_size": 128,
                    "dampening_frac": 0.01,
                    "actorder": "static",
                    "config_groups": {"group_0": group_4bit},
                }
            }
        }
    }


def _is_quantized_model_ready(model_dir: str) -> bool:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    if not any(model_path.glob("*.safetensors")):
        return False
    try:
        with open(config_path, "r") as fin:
            cfg = json.load(fin)
    except Exception:
        return False
    return "quantization_config" in cfg


def _build_calibration_json(
    dataset_name: str,
    lang_pairs: list[str],
    output_json: str,
    max_samples: int,
) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "src" / "llama_recipes" / "customer_data" / dataset_name / "test"
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with open(output_path, "w") as fout:
        for lp in lang_pairs:
            src, tgt = lp.split("-")
            src_file = data_root / lp / f"test.{lp}.{src}"
            with open(src_file) as fin:
                for line in fin:
                    text = line.strip()
                    if not text:
                        continue
                    fout.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
                    num_written += 1
                    if num_written >= max_samples:
                        return num_written
    return num_written


def main(
    model_dir: str,
    quantization_config: str,
    output_dir: str,
    calibration_data_path: str = None,
    dataset_name: str = "wmt24_testset",
    lang_pairs: str = "en-ru,en-zh",
    num_calibration_samples: int = 512,
    max_seq_length: int = 384,
    model_name_for_profile: str = None,
    model_configs_dir: str = None,
):
    quantization_mode = _load_quantization_mode(quantization_config)
    model_hint = model_name_for_profile or _infer_model_name_from_dir(model_dir)

    if quantization_mode == "awq" and _is_gemma3_model(model_hint):
        raise RuntimeError(
            "AWQ quantization for Gemma 3 is disabled in this pipeline. "
            "Please check this script/model instead: "
            "https://huggingface.co/pytorch/gemma-3-12b-it-AWQ-INT4"
        )

    if _is_quantized_model_ready(output_dir):
        print(f"Quantized model already exists at {output_dir}, skipping quantization.", flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "_quant_work")
    os.makedirs(work_dir, exist_ok=True)

    awq_overrides = None
    profile_path = None
    if quantization_mode == "awq":
        awq_overrides, profile_path = _get_awq_overrides(
            model_dir=model_dir,
            model_name_for_profile=model_hint,
            model_configs_dir=model_configs_dir,
        )
        if awq_overrides:
            print(
                f"Using model-specific AWQ overrides from: {profile_path}",
                flush=True,
            )
    recipe = _build_recipe(quantization_mode, awq_overrides=awq_overrides)
    recipe_path = os.path.join(work_dir, f"{quantization_mode}_recipe.json")
    with open(recipe_path, "w") as fout:
        json.dump(recipe, fout, indent=2)

    if calibration_data_path is None:
        calib_path = os.path.join("data", "calibration_data", f"{dataset_name}.json")
    else:
        calib_path = calibration_data_path
    calib_path = os.path.abspath(calib_path)
    calib_dir = os.path.dirname(calib_path)
    os.makedirs(calib_dir, exist_ok=True)
    pairs = [p.strip() for p in lang_pairs.split(",") if p.strip()]
    if os.path.exists(calib_path) and os.path.getsize(calib_path) > 0:
        print(f"Using existing calibration data file: {calib_path}", flush=True)
        with open(calib_path, "r") as fin:
            written = sum(1 for _ in fin)
    else:
        written = _build_calibration_json(
            dataset_name=dataset_name,
            lang_pairs=pairs,
            output_json=calib_path,
            max_samples=num_calibration_samples,
        )
    if written == 0:
        raise ValueError("No calibration samples were generated.")

    # Force a deterministic split name for llmcompressor (it expects train/calibration key).
    # We load the JSONL content as split "test", then alias that split to "calibration".
    calib_loader_dir = os.path.join(work_dir, "calibration_dataset")
    os.makedirs(calib_loader_dir, exist_ok=True)
    calib_loader_path = os.path.join(calib_loader_dir, "test.json")
    if os.path.abspath(calib_path) != os.path.abspath(calib_loader_path):
        shutil.copyfile(calib_path, calib_loader_path)

    print(f"Quantizing model with {quantization_mode}...", flush=True)
    print(f"Calibration samples: {written}", flush=True)
    print(f"Calibration data file: {calib_path}", flush=True)
    oneshot(
        model=model_dir,
        tokenizer=model_dir,
        trust_remote_code_model=True,
        recipe=recipe_path,
        dataset="json",
        dataset_path=calib_loader_dir,
        splits={"calibration": "test"},
        text_column="text",
        num_calibration_samples=written,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
    )
    print(f"Quantized model saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
