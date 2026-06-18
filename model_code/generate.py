from model_code.steering_extraction import generateSteering

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
