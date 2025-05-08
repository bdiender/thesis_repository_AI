import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt


def get_weight_changes(model1, model2):
    num_layers = len(model1.text_field_embedder._token_embedders['tokens'].transformer_model.encoder.layer)
    components = ['query', 'key', 'value', 'attn_output', 'ffn_intermediate', 'ffn_output']
    changes = {comp: np.zeros(num_layers) for comp in components}

    for layer_idx in range(num_layers):
        layer1 = model1.text_field_embedder._token_embedders['tokens'].transformer_model.encoder.layer[layer_idx]
        layer2 = model2.text_field_embedder._token_embedders['tokens'].transformer_model.encoder.layer[layer_idx]

        attn1 = layer1.attention.self
        attn2 = layer2.attention.self

        weights1 = attn1.query.weight
        weights2 = attn2.query.weight
        changes['query'][layer_idx] = torch.norm(weights2 - weights1, p='fro').item() / torch.norm(weights1, p='fro').item()

        weights1 = attn1.key.weight
        weights2 = attn2.key.weight
        changes['key'][layer_idx] = torch.norm(weights2 - weights1, p='fro').item() / torch.norm(weights1, p='fro').item()

        weights1 = attn1.value.weight
        weights2 = attn2.value.weight
        changes['value'][layer_idx] = torch.norm(weights2 - weights1, p='fro').item() / torch.norm(weights1, p='fro').item()

        weights1 = layer1.attention.output.dense.weight
        weights2 = layer2.attention.output.dense.weight
        changes['attn_output'][layer_idx] = torch.norm(weights2 - weights1, p='fro').item() / torch.norm(weights1, p='fro').item()

        weights1 = layer1.intermediate.dense.weight
        weights2 = layer2.intermediate.dense.weight
        changes['ffn_intermediate'][layer_idx] = torch.norm(weights2 - weights1, p='fro').item() / torch.norm(weights1, p='fro').item()

        weights1 = layer1.output.dense.weight
        weights2 = layer2.output.dense.weight
        changes['ffn_output'][layer_idx] = torch.norm(weights2 - weights1, p='fro').item() / torch.norm(weights1, p='fro').item()
    
    return changes


def visualize_weight_changes(model1, model2, title="Weight Changes After Fine-tuning"):
    """
    Visualize weight changes between two BERT models as a heatmap.
    """
    changes = get_weight_changes(model1, model2)

    # Convert to a 2D array for heatmap
    num_layers = len(model1.encoder.layer)
    components = ["query", "key", "value", "attn_output", "ffn_intermediate", "ffn_output"]

    data = np.zeros((len(components), num_layers))
    for i, comp in enumerate(components):
        data[i] = changes[comp]

    # Create nice labels
    nice_labels = {
        "query": "Query",
        "key": "Key",
        "value": "Value",
        "attn_output": "Attention Output",
        "ffn_intermediate": "FFN Intermediate",
        "ffn_output": "FFN Output"
    }

    # Create heatmap
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=[f"Layer {i}" for i in range(num_layers)],
        yticklabels=[nice_labels[comp] for comp in components]
    )

    plt.title(title, fontsize=16)
    plt.tight_layout()

    return plt.gcf()


def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


deu_model = load_model('../../from_das/1_deu/model.pkl')
vep_model = load_model('outputs/vep_test/model.pkl')

deu_model.cpu()
vep_model.cpu()

print(get_weight_changes(vep_model, deu_model))
