# This is for extracting vectors
# We mostly follow this method: https://elib.dlr.de/218629/1/The_Effectiveness_of_Style_Vectors_for_Steering_Large_Language_Models_A_Human_Evaluation.pdf
# Extracting the input representation and averaging them to determine layer representation
import os

import torch

import torch
import torch.nn.functional as F


# ======================================================
# EXTRACTING STEERING VECTORS
# ======================================================

def _extractAllLayer(user_text: str, model, tokenizer):
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

    # Extract the mean hidden state for each layer (excluding the input embedding layer)
    all_layer_means = [
        layer_hidden.mean(dim=1).squeeze(0).detach().cpu()
        for layer_hidden in outputs.hidden_states[1:]
    ]


    return torch.stack(all_layer_means)


def retrieve_steering_vector(model, tokenizer, emotion_sets: dict, name_folder: str, layer_id=None):
    """
    Retrieves a steering vector for a given emotion by passing the emotion as inference
    Then receiving the hidden state and subtracting the mean of the other emotion hidden states to get a contrastive vector
    """
    if not isinstance(emotion_sets, dict) or not emotion_sets:
        raise ValueError("emotion_sets must be a non-empty dict of {emotion: prompts}.")

    if len(emotion_sets) < 2:
        raise ValueError("emotion_sets must contain at least 2 emotions to compute contrastive steering vectors.")

    hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    if layer_id is None:
        layer_id = [21]
    if isinstance(layer_id, int):
        layer_id = [layer_id]
    if not layer_id:
        raise ValueError("layer_id must contain at least one layer index.")
    if any((lid < 0 or lid >= hidden_layers) for lid in layer_id):
        raise ValueError(f"layer_id values must be in [0, {hidden_layers - 1}].")

    emotion_vectors = {emotion: [] for emotion in emotion_sets}
    vectors = {emotion: torch.zeros(hidden_layers, hidden_size) for emotion in emotion_sets}
    steering_vectors = {}


    # Extracting total vectors for each emotion
    for emotion, prompts in emotion_sets.items():
        if not prompts:
            raise ValueError(f"Emotion '{emotion}' has no prompts.")
        for prompt in prompts:
            vector = _extractAllLayer(prompt, model, tokenizer) # [32,4096]
            # For analysis purposes
            for lid in layer_id:
                emotion_vectors[emotion].append(vector[lid].cpu().numpy())
            vectors[emotion] += vector

    # Normalize vectors to get means
    for emotion in vectors:
        vectors[emotion] = vectors[emotion] / len(emotion_sets[emotion])
    
    for target in vectors:
    # Take the mean of all the other emotion vectors to create a contrastive vector
        contrastive = torch.stack(
            [vectors[e] for e in vectors if e != target], dim=0
        ).mean(dim=0)   # [layers, hidden_size]
        steering_vectors[target] = vectors[target] - contrastive # Both [32,4096]
    
    # Save the steering vectors and emotion vectors for analysis
    save_dir = f"resources/saved_vectors/{name_folder}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(steering_vectors, os.path.join(save_dir, "steering_vectors.pt"))
    torch.save(emotion_vectors, os.path.join(save_dir, "emotion_vectors.pt"))
    
    
    return steering_vectors, emotion_vectors
            

# ======================================================
# GENERATION WITH STEERING 
# ======================================================


# Apply steering and generate text with steer 
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


    # Clone and apply steering to all token positions
    steered_hidden = hidden_state.clone()
    steered_hidden = steered_hidden + (layer_vector * strength)
    
    # Soemtimes the output can be a tuple, containing the previous keys and those likes
    # So we only want to modify the first part
    if is_tuple:
        return (steered_hidden,) + output[1:]
    else:
        return steered_hidden



# Generate steering through class wrapper that has __exit__ and __enter__ to apply and remove hooks, 
class SteeringApplier:
    def __init__(self, model, steering_vector, strength=1.0, target_layers=None):
        self.model = model
        self.steering_vector = steering_vector.to(model.device)
        self.strength = strength
        self.target_layers = target_layers if target_layers is not None else list(range(model.config.num_hidden_layers))
        self.hook_handles = []

    def _hook_fn(self, module, input, output, layer_idx):
        return _extractVectorSteer(
            module, input, output, self.steering_vector, self.strength, layer_idx
        )

    def __enter__(self):
        for layer_idx in self.target_layers:
            handle = self.model.model.layers[layer_idx].register_forward_hook(
                lambda module, input, output, idx=layer_idx: self._hook_fn(module, input, output, idx)
            )
            self.hook_handles.append(handle)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


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
    """Generate text with optional steering applied to specified layers."""
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

    # we want to make sure the steering vector is on the same device 
    # AND that the steering vector is the correct shape and type
    if steering_vector is not None:
        if steering_vector.shape[0] != model.config.num_hidden_layers or steering_vector.shape[1] != model.config.hidden_size:
            raise ValueError(f"steering_vector must have shape [{model.config.num_hidden_layers}, {model.config.hidden_size}]")
        if steering_vector.device != model.device:
            steering_vector = steering_vector.to(model.device)
        if steering_vector.dtype != next(model.parameters()).dtype:
            steering_vector = steering_vector.to(next(model.parameters()).dtype)


    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Only wrap wrapper if steering_vector is provided, otherwise just generate normally
    if steering_vector is None:
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
    else:
        with SteeringApplier(model, steering_vector, steering_strength, target_layers):
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
    
    input_length = input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


    return generated_text











def norm_vectors(vectors):
    """
    Receive the full [32,4096] steering vectors 
    Normalise each vector to have unit norm, so that we can apply them with a consistent strength across layers without some layers dominating due to larger magnitudes.
    """
    normed_vectors = torch.zeros_like(vectors)
    for i, vector in enumerate(vectors):
        normed_vectors[i] = vector / torch.norm(vector)
    return normed_vectors