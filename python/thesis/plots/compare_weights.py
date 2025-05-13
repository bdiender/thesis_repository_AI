import argparse
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sns

from .plot_utils import load_model

MODELS = (
    ('nld', '#DE8900', 'Dutch'),
    ('nob', '#0072B2', 'Norwegian'),
    ('ces', '#B30000', 'Czech'),
    ('fin', '#008E67', 'Finnish')
)

COMPONENTS = (
    'query', 'key', 'value',
    'attn_output', 'ffn_intermediate', 'ffn_output'
)

NICE_LABELS = {
    "query": "Query",
    "key": "Key",
    "value": "Value",
    "attn_output": "Attention Output",
    "ffn_intermediate": "FFN Intermediate",
    "ffn_output": "FFN Output"
}

def rel_change(w1, w2):
    norm1 = torch.norm(w1, p='fro').item()

    if norm1 == 0:
        return 0.0

    return torch.norm(w2 - w1, p='fro').item() / norm1

def get_weight_changes(model1, model2):
    enc1 = model1.text_field_embedder._token_embedders['tokens'].transformer_model.encoder
    enc2 = model2.text_field_embedder._token_embedders['tokens'].transformer_model.encoder

    num_layers = len(enc1.layer)
    changes = {c: np.zeros(num_layers) for c in COMPONENTS}

    getters = {
        'query': lambda l1, l2: (l1.attention.self.query.weight, l2.attention.self.query.weight),
        'key': lambda l1, l2: (l1.attention.self.key.weight, l2.attention.self.key.weight),
        'value': lambda l1, l2: (l1.attention.self.value.weight, l2.attention.self.value.weight),
        'attn_output': lambda l1, l2: (l1.attention.output.dense.weight, l2.attention.output.dense.weight),
        'ffn_intermediate':lambda l1, l2: (l1.intermediate.dense.weight, l2.intermediate.dense.weight),
        'ffn_output': lambda l1, l2: (l1.output.dense.weight, l2.output.dense.weight),
    }

    for i, (lay1, lay2) in enumerate(zip(enc1.layer, enc2.layer)):
        for comp in COMPONENTS:
            w1, w2 = getters[comp](lay1, lay2)
            changes[comp][i] = rel_change(w1, w2)

    return changes

def visualize_weight_changes(model1, model2, color='#FF0000', title=None):
    title = title or 'Weight Changes After Fine-tuning'
    changes = get_weight_changes(model1, model2)

    num_layers = len(model1.text_field_embedder._token_embedders['tokens'].transformer_model.encoder.layer)
    data = np.array([changes[c] for c in COMPONENTS])
    cmap = sns.light_palette(color, as_cmap=True)
    plt.figure(figsize=(12, 7))

    ax = sns.heatmap(
        data,
        annot=True,
        fmt='.4f',
        cmap=cmap,
        xticklabels=[f'Layer {i}' for i in range(num_layers)],
        yticklabels=[NICE_LABELS[c] for c in COMPONENTS]
    )

    plt.title(title, fontsize=16)
    plt.tight_layout()

    return plt.gcf()

def visualize_all_weight_changes(models, lang_dict, title=None):
    title = title or 'Relative BERT-Layer Changes by Language'
    ref_model, _ = models[0]
    others = models[1:]

    n_comp = len(COMPONENTS)
    n_layers = len(ref_model.text_field_embedder._token_embedders['tokens'].transformer_model.encoder.layer)
    n_lang = len(others)

    data = np.zeros((n_comp, n_layers, n_lang))

    for l_idx, (_, mdl) in enumerate(others):
        ch = get_weight_changes(ref_model, mdl)
        for c_idx, comp in enumerate(COMPONENTS):
            data[c_idx, :, l_idx] = ch[comp]

    fig, axs = plt.subplots(n_comp, n_layers, figsize=(n_layers * 1.2, n_comp * 1.2), sharex=True, sharey=True)

    for i in range(n_comp):
        for j in range(n_layers):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            xs = np.arange(n_lang)
            ax.bar(xs, data[i, j, :], width=0.8, color=[lang_dict[mn] for mn, _ in others])
            ax.axhline(0, linewidth=0.3)
            ax.set_ylim(0, data.max() * 1.05)
            if j == 0:
                ax.set_ylabel(COMPONENTS[i], rotation=0, labelpad=20, va='center')
            if i == n_comp - 1:
                ax.set_xlabel(f'L{i}', labelpad=5)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--visualize',
        required=True,
        help='One of {"all", "nld", "nob", "ces", "fin"}'
    )
    parser.add_argument(
        '--model_dir',
        default=None,
        help='Directory that contains output directories'
    )
    parser.add_argument(
        '--output-dir',
        default=None
    )
    parser.add_argument(
        '--plot_title',
        default=None
    )
    parser.add_argument(
        '--verbose',
        default=False
    )
    args = parser.parse_args()
    
    now = lambda: dt.now().strftime('%H:%M:%S')
    output_dir = args.output_dir or 'outputs/plots'
    model_dir = args.model_dir or '../../from_das'
    
    deu = load_model(os.path.join(model_dir, '1_deu', 'model.pkl'))
    deu.cpu()
    if args.verbose:
        print(f'[{now()}] German model loaded.')
    
    fig = None
    if args.visualize == 'all':
        local = [('deu', deu)]
    
        for mn, _, ma in MODELS:
            path = os.path.join(model_dir, f'2_{mn}', 'model.pkl')
            if os.path.isfile(path):
                m = load_model(path)
                m.cpu()
    
                if args.verbose:
                    print(f'[{now()}] {ma} model loaded.')
    
                local.append((mn, m))
    
            elif args.verbose:
                print(f'Could not find {ma} model at {path}')
    
        fig = visualize_all_weight_changes(local, {ma: mc for _, mc, ma in MODELS}, title=args.plot_title)
    
    else:
        mn = args.visualize
        path = os.path.join(model_dir, f'2_{mn}', 'model.pkl')
        color = {mn: mc for mn, mc, _ in MODELS}[mn]
    
        if os.path.isfile(path):
            model = load_model(path)
            model.cpu()
    
            if args.verbose:
                print(f'[{now()}] {mn} model loaded.')
    
        elif args.verbose:
            print(f'Could not find {mn} model at {path}')
    
        fig = visualize_weight_changes(deu, model, color=color, title=args.plot_title)
    
    if fig:
        fig.savefig(os.path.join(output_dir, f'weights_heatmap_{args.visualize}.pdf'))
