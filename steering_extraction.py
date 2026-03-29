# This is for extracting vectors
# We mostly follow this method: https://elib.dlr.de/218629/1/The_Effectiveness_of_Style_Vectors_for_Steering_Large_Language_Models_A_Human_Evaluation.pdf
# Extracting the input representation and averaging them to determine layer representation
import torch

import torch
import torch.nn.functional as F

def extractAllLayer(user_text: str, model, tokenizer):
    """
    Extract mean hidden-state vectors from all transformer layers.

    Returns:
        torch.Tensor of shape [num_layers, hidden_size]
    """

    # With chat template
    messages = [
        {"role": "user", "content": user_text},
    ]

    text_for_model = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text_for_model,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )
    # input_length = input_ids.shape[1]
    # generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    # print(generated_text)
    all_layer_means = [
        layer_hidden.mean(dim=1).squeeze(0).detach().cpu()
        for layer_hidden in outputs.hidden_states[1:]
    ]


    return torch.stack(all_layer_means)



def _extractVectorSteer(module, input, output, steering_vector, strength, layer_idx):
    if isinstance(output, tuple):
        hidden_state = output[0]
        is_tuple = True
    else:
        hidden_state = output
        is_tuple = False

    layer_vector = steering_vector[layer_idx]  # pick corresponding [hidden_size] for this layer

    if layer_vector.device != hidden_state.device:
        layer_vector = layer_vector.to(hidden_state.device)
    if layer_vector.dtype != hidden_state.dtype:
        layer_vector = layer_vector.to(hidden_state.dtype)

    # Normalize the steering vector to unit norm before applying
    # layer_vector_normalized = F.normalize(layer_vector, p=2, dim=0)
    
    # Clone and apply steering to all token positions
    steered_hidden = hidden_state.clone()
    steered_hidden = steered_hidden + (layer_vector * strength)
    
    # Soemtimes the output can be a tuple, containing the previous keys and those likes
    # So we only want to modify the first part
    if is_tuple:
        return (steered_hidden,) + output[1:]
        #           [0]               [1] 
    else:
        return steered_hidden



def generateSteering(
    user_text: str,
    system_text: str,
    model,
    tokenizer,
    steering_vector=None,   # expected shape [num_layers, hidden_size]
    steering_strength=1.0,
    target_layers=None,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}
    ]

    # Apply chat template without tokenizing, then tokenize separately
    text_for_model = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text_for_model,
        return_tensors="pt"
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    hook_handles = []

    if steering_vector is not None:
        steering_vector = steering_vector.to(model.device)

        num_layers = len(model.model.layers)

        if target_layers is None:
            target_layers = list(range(num_layers))
        if isinstance(target_layers, int):
            target_layers = [target_layers]

        for layer_idx in target_layers:
            def hook_fn(module, input, output, layer_idx=layer_idx, steering_vector=steering_vector):
                return _extractVectorSteer(
                    module, input, output, steering_vector, steering_strength, layer_idx
                )

            handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            hook_handles.append(handle)

        print(f"Applied steering to layers {target_layers} with strength {steering_strength}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for handle in hook_handles:
        handle.remove()

    input_length = input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return generated_text