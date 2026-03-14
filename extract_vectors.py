
# Method to extract vectors from model 
def hook_extract_vector(module, input, output):
    # The output from model.model (LlamaModel) during generation is typically a BaseModelOutputWithPast object.
    # We want to extract the last_hidden_state tensor from it.
    # try another method 
    # hidden_state = output.last_hidden_state

    if isinstance(output, tuple):
        hidden_state = output[0]
    else:
        hidden_state = output.last_hidden_state

    # Detach the tensor from the graph and move to CPU to save memory if not needed for backprop
    extracted_activations.append(hidden_state.detach().cpu())

    # IMPORTANT: Hooks should return the original output or a modified one. If only observing, return original output.
    return output

# model generates with the given prompt and extracts the vectors
def generate_extract(text: str, model, tokenizer,):
    # inputs = tokenizer(text, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # return outputs

