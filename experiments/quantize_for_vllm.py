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


def _build_recipe(quantization_mode: str) -> dict:
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
        return {
            "quant_stage": {
                "quant_modifiers": {
                    "AWQModifier": {
                        "ignore": ["lm_head"],
                        "config_groups": {"group_0": group_4bit},
                    }
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
):
    quantization_mode = _load_quantization_mode(quantization_config)
    quant_cfg_candidates = (
        "quantize_config.json",
        "quant_config.json",
        "gptq_config.json",
        "awq_config.json",
    )
    if any(os.path.exists(os.path.join(output_dir, name)) for name in quant_cfg_candidates):
        print(f"Quantized model already exists at {output_dir}, skipping quantization.", flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "_quant_work")
    os.makedirs(work_dir, exist_ok=True)

    recipe = _build_recipe(quantization_mode)
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
