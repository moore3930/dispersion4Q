from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

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
    "et": "Estonian",
    "lv": "Latvian",
    "sl": "Slovenian",
    "fy": "Frisian",
    "ug": "Uyghur",
}


def _model_configs_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "experiments" / "model_configs"


def load_model_prompt_config(model_name: str) -> Dict[str, Any]:
    model_configs_dir = _model_configs_dir()
    if not model_configs_dir.is_dir():
        return {}

    model_name_l = str(model_name or "").lower()
    if not model_name_l:
        return {}

    best_profile = None
    best_score = -1

    for config_file in sorted(model_configs_dir.glob("*.json")):
        with config_file.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        for match_token in profile.get("match_substrings", []):
            token = str(match_token).lower()
            if token and token in model_name_l and len(token) > best_score:
                best_profile = profile
                best_score = len(token)

    if not best_profile:
        return {}
    return best_profile.get("prompt", {}) or {}


def resolve_model_instruction(model_name: str, prompt_config: Dict[str, Any]) -> str:
    configured_instruction = prompt_config.get("model_instruction")
    if configured_instruction is not None:
        return str(configured_instruction).strip()

    model_name_l = str(model_name or "").lower()
    if "gemma" in model_name_l:
        return "Provide only one translation and do not output anything else after that."
    return ""


def resolve_add_target_lang_prompt(model_name: str, prompt_config: Dict[str, Any]) -> bool:
    configured_value = prompt_config.get("add_target_lang_prompt")
    if configured_value is not None:
        return bool(configured_value)

    model_name_l = str(model_name or "").lower()
    return "mistral" in model_name_l


def build_translation_user_prompt(
    src_lang_code: str,
    tgt_lang_code: str,
    src_text: str,
    model_name: str,
    prompt_config: Dict[str, Any],
) -> str:
    src_lang_name = LANG_NAME.get(src_lang_code, src_lang_code)
    tgt_lang_name = LANG_NAME.get(tgt_lang_code, tgt_lang_code)

    model_instruction = resolve_model_instruction(model_name, prompt_config)
    instruction_suffix = f" {model_instruction}" if model_instruction else ""
    add_target_lang_prompt = resolve_add_target_lang_prompt(model_name, prompt_config)
    target_lang_prompt = f"{tgt_lang_name}:" if add_target_lang_prompt else ""

    return (
        f"Translate the following text from {src_lang_name} into {tgt_lang_name}.{instruction_suffix}\n"
        f"{src_lang_name}: {src_text}\n"
        f"{target_lang_prompt}"
    )


def build_translation_prompt(
    tokenizer,
    src_lang_code: str,
    tgt_lang_code: str,
    src_text: str,
    model_name: str,
    prompt_config: Dict[str, Any],
    add_generation_prompt: bool = True,
) -> str:
    user_prompt = build_translation_user_prompt(
        src_lang_code=src_lang_code,
        tgt_lang_code=tgt_lang_code,
        src_text=src_text,
        model_name=model_name,
        prompt_config=prompt_config,
    )
    messages = [{"role": "user", "content": user_prompt}]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return user_prompt
