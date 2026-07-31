import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

from model_code.steering_extraction import generateSteering, generateBatchedSteering

# Progress bar 
class _NoOpProgress:
    def update(self, n=1):
        return None

    def close(self):
        return None
def _make_progress(total, desc, show_progress):
    if not show_progress:
        return _NoOpProgress()
    try:
        tqdm_fn = __import__("tqdm").tqdm
        return tqdm_fn(total=total, desc=desc)
    except Exception:
        return _NoOpProgress()

# Test different prompts and strengths
def generateTextsList(
    prompts: list[str],
    system_text: str,
    model,
    steering_vector,
    tokenizer,
    steering_strengths=None,
    max_new_tokens=100,
    target_layers=None,
    show_progress=True,
    progress_desc="Generating"
):
    if steering_strengths is None:
        steering_strengths = [1.0] 

    generated_texts = {}
    total_steps = len(prompts) * len(steering_strengths)
    pbar = _make_progress(total_steps, progress_desc, show_progress)
    try:
        for prompt_text in prompts:
            generated_texts[prompt_text] = []
            for steer_strength in steering_strengths:
                generated_text = generateSteering(
                    user_text=prompt_text,
                    system_text=system_text,
                    model=model,
                    steering_vector=steering_vector,
                    tokenizer=tokenizer,
                    target_layers=target_layers,
                    steering_strength=steer_strength,
                    max_new_tokens=max_new_tokens,
                )
                generated_texts[prompt_text].append({
                    "steering_strength": steer_strength,
                    "generated_text": generated_text
                })
                pbar.update(1)
    finally:
        pbar.close()
    return generated_texts








# Test different layers and strengths
def generateTextsLayers(
    prompt_text: str,
    system_text: str,
    model,
    steering_vector,
    tokenizer,
    target_layers=None,
    steering_strengths=None,
    max_new_tokens=100,
    show_progress=True,
    progress_desc="Generating by layer"
):
    if target_layers is None:
        target_layers = [20]
    if steering_strengths is None:
        steering_strengths = [1.0]

    generated_texts = {}
    total_steps = len(steering_strengths) * len(target_layers)
    pbar = _make_progress(total_steps, progress_desc, show_progress)
    try:
        for steer_strength in steering_strengths:
            generated_texts[steer_strength] = []
            for layer in target_layers:
                generated_text = generateSteering(
                    user_text=prompt_text,
                    system_text=system_text,
                    model=model,
                    steering_vector=steering_vector,
                    tokenizer=tokenizer,
                    target_layers=[layer],
                    steering_strength=steer_strength,
                    max_new_tokens=max_new_tokens,
                )
                generated_texts[steer_strength].append({
                    "target_layers": [layer],
                    "steering_strength": steer_strength,
                    "generated_text": generated_text
                })
                pbar.update(1)
    finally:
        pbar.close()
    return generated_texts


def generateTextsBatched(
    user_texts: list[str],
    system_text: str,
    model,
    steering_vector,
    tokenizer,
    target_layers=None,
    steering_strengths=None,
    max_new_tokens=100,
    show_progress=True,
    progress_desc="Generating batched steering"
):
    if target_layers is None:
        target_layers = [20]
    if steering_strengths is None:
        steering_strengths = [1.0]
    if steering_vector is None:
        steering_strengths = [1.0]  # If no steering vector, set strength to 0
        steering_vector = None

    generated_texts = []
    total_steps = len(user_texts) * len(steering_strengths)
    pbar = _make_progress(total_steps, progress_desc, show_progress)
    try:
        for steer_strength in steering_strengths:
            all_generated_texts = generateBatchedSteering(
                user_texts=user_texts,
                system_text=system_text,
                model=model,
                steering_vector=steering_vector,
                tokenizer=tokenizer,
                target_layers=target_layers,
                steering_strength=steer_strength,
                max_new_tokens=max_new_tokens,
            )
            all_texts = [
                {
                    "user_text": user_text,
                    "steering_strength": steer_strength,
                    "generated_text": generated_text
                } for user_text, generated_text in zip(user_texts, all_generated_texts)
            ]
            generated_texts.extend(all_texts)
            pbar.update(len(all_texts))
    finally:
        pbar.close()
    return generated_texts




def save_generated_outputs(
    data,
    output_path: str | None = None,
    output_dir: str = "outputs",
    file_prefix: str = "texts_generated",
    include_timestamp: bool = True,
    indent: int = 2,
    ensure_ascii: bool = False,
):
    """Save generated outputs to a JSON file and return the saved path."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = f"{file_prefix}_{timestamp}.json" if timestamp else f"{file_prefix}.json"
        output_file = Path(output_dir) / filename
    else:
        output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)

    return str(output_file)


def save_emotion_responses_csv(
    emotion_responses: dict,
    output_path: str,
    derived_from: str,
    model_name: str = "LLAMA Indonesian",
):
    """Save generated emotion responses as one CSV row per response."""
    rows = [
        {
            "id": str(uuid.uuid4()),
            "prompt": response.get("user_text", ""),
            "response": response.get("generated_text", ""),
            "steering_condition": emotion,
            "strength": response.get("steering_strength", "NA"),
            "derived_from": derived_from,
            "model": model_name,
        }
        for emotion, responses in emotion_responses.items()
        for response in responses
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "prompt", "response", "steering_condition",
        "strength", "derived_from", "model",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(output_file), len(rows)
