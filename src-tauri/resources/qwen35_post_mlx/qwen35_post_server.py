#!/usr/bin/env python3
"""
Qwen3.5 post-processing server using mlx-lm.
A persistent process keeps model loaded in memory for low-latency text cleanup.
"""

import json
import os
import re
import sys
import time

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

_MODEL_PATH = os.environ.get("HANDY_QWEN35_MODEL_PATH", "").strip()
_model = None
_tokenizer = None


def load_model_once():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not _MODEL_PATH:
        raise RuntimeError("HANDY_QWEN35_MODEL_PATH is empty")

    from mlx_lm import load

    start = time.time()
    _model, _tokenizer = load(_MODEL_PATH)
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s from {_MODEL_PATH}", file=sys.stderr, flush=True)
    return _model, _tokenizer


def build_prompt(system_prompt: str, user_content: str, tokenizer) -> str:
    safety_rules = (
        "You are a strict transcription post-processor.\n"
        "Output rules:\n"
        "1. Output only the final processed text.\n"
        "2. Never output reasoning, analysis, or chain-of-thought.\n"
        "3. Never output <think> tags.\n"
        "4. Never output explanations, bullet points, examples, or wrappers.\n"
        "5. Never repeat the instruction text.\n"
    )

    base_system = (system_prompt or "").strip()
    merged_system = (
        f"{safety_rules}\n\n{base_system}" if base_system else safety_rules
    ).strip()

    task = (user_content or "").strip()
    if not task:
        task = "Return an empty string."

    task = f"{task}\n\nReturn only the final text."
    messages = [
        {"role": "system", "content": merged_system},
        {"role": "user", "content": task},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                pass
        except Exception:
            pass

    return (
        f"{merged_system}\n\n"
        f"Task:\n{task}"
    )


def clean_output(text: str) -> str:
    output = str(text or "").strip()
    output = re.sub(r"(?is)<think>.*?</think>", "", output)
    output = output.replace("<think>", "").replace("</think>", "")
    output = output.strip()

    # Collapse duplicated consecutive non-empty lines.
    lines = []
    prev = None
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line
    if lines:
        output = "\n".join(lines)

    # Remove common no-think markers echoed by some chat templates.
    output = re.sub(r"(?im)^\s*/?no[_\s-]?think\s*$", "", output)

    # Strip common prefixed wrappers while keeping the actual final text.
    output = re.sub(r"(?im)^(final answer|answer|result|输出|结果)[:：]\s*", "", output)
    output = output.strip()

    # Re-collapse lines after marker stripping.
    lines = []
    prev = None
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line
    if lines:
        output = "\n".join(lines)
    return output.strip()


def process_text(
    text: str,
    system_prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.16,
    repetition_context_size: int = 96,
) -> str:
    model, tokenizer = load_model_once()

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    prompt = build_prompt(system_prompt, text, tokenizer)
    sampler = make_sampler(max(0.0, float(temperature)), top_p=max(0.0, float(top_p)))
    logits_processors = make_logits_processors(
        repetition_penalty=float(repetition_penalty),
        repetition_context_size=max(16, int(repetition_context_size)),
    )
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )

    output = str(result or "").strip()
    if output.startswith(prompt):
        output = output[len(prompt) :].strip()

    cleaned = clean_output(output)
    return cleaned


def main():
    print("Qwen3.5 post-process server starting...", file=sys.stderr, flush=True)

    try:
        load_model_once()
        print("READY", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr, flush=True)
        raise

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            text = str(req.get("text", ""))
            system_prompt = str(req.get("system_prompt", ""))
            max_tokens = int(req.get("max_tokens", 256) or 256)
            temperature = float(req.get("temperature", 0.0) or 0.0)
            top_p = float(req.get("top_p", 0.0) or 0.0)
            repetition_penalty = float(req.get("repetition_penalty", 1.16) or 1.16)
            repetition_context_size = int(req.get("repetition_context_size", 96) or 96)

            processed = process_text(
                text=text,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )

            print(json.dumps({"text": processed}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
