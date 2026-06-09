from model_code.steering_extraction import generateSteering

# Test different prompts and strengths
def generateTextsList(
    prompts: list[str],
    system_text: str,
    model,
    steering_vector,
    tokenizer,
    steering_strengths=None,
    max_new_tokens=100
):
    if steering_strengths is None:
        steering_strengths = [1.0]

    generated_texts = {}
    for prompt_text in prompts:
        generated_texts[prompt_text] = []
        for steer_strength in steering_strengths:
            generated_text = generateSteering(
                user_text=prompt_text,
                system_text=system_text,
                model=model,
                steering_vector=steering_vector,
                tokenizer=tokenizer,
                steering_strength=steer_strength,
                max_new_tokens=max_new_tokens,
            )
            generated_texts[prompt_text].append({
                "steering_strength": steer_strength,
                "generated_text": generated_text
            })
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
    max_new_tokens=100
):
    if target_layers is None:
        target_layers = [20]
    if steering_strengths is None:
        steering_strengths = [1.0]

    generated_texts = {}
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
    return generated_texts