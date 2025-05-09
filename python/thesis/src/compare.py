import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt

now = lambda: dt.now().strftime('%H:%M:%S')

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


def get_decoder_weight_changes(model1, model2):
    """
    Compute relative Frobenius‐norm changes for each submodule in DependencyDecoder.
    Returns a dict mapping component names to scalar change values.
    """
    dec1 = model1.decoder
    dec2 = model2.decoder

    changes = {}

    # BiLSTM layers inside decoder.encoder: it's a PytorchSeq2SeqWrapper around an LSTM
    # weight_ih_l{k}, weight_hh_l{k} for each layer k and direction
    lstm = dec1.encoder._module  # the actual torch.nn.LSTM
    num_layers = lstm.num_layers
    directions = 2 if lstm.bidirectional else 1
    for layer in range(num_layers):
        for dir_idx in range(directions):
            suffix = f'_l{layer}{"_reverse" if dir_idx else ""}'
            w1 = getattr(lstm, f'weight_ih{suffix}')
            w2 = getattr(model2.decoder.encoder._module, f'weight_ih{suffix}')
            key = f'lstm_ih{suffix}'
            changes[key] = torch.norm(w2 - w1, p='fro').item() / torch.norm(w1, p='fro').item()

            w1 = getattr(lstm, f'weight_hh{suffix}')
            w2 = getattr(model2.decoder.encoder._module, f'weight_hh{suffix}')
            key = f'lstm_hh{suffix}'
            changes[key] = torch.norm(w2 - w1, p='fro').item() / torch.norm(w1, p='fro').item()

    # head_arc_mlp and child_arc_mlp (linear weight)
    for name in ['head_arc_mlp', 'child_arc_mlp', 'head_tag_mlp', 'child_tag_mlp']:
        lin1 = getattr(dec1, name)._linear_layers[-1]
        lin2 = getattr(dec2, name)._linear_layers[-1]

        changes[name + '_weight'] = torch.norm(lin2.weight - lin1.weight, p='fro').item() / torch.norm(lin1.weight, p='fro').item()
        changes[name + '_bias']   = torch.norm(lin2.bias   - lin1.bias,   p=2).item()        / torch.norm(lin1.bias,   p=2).item()

    # arc_attention (BilinearMatrixAttention has weight matrix W and biases)
    # its parameter is .weight, ._bias1, .bias2
    att1 = dec1.arc_attention
    att2 = dec2.arc_attention
    changes['arc_attention_weight'] = torch.norm(att2._weight_matrix - att1._weight_matrix, p='fro').item() / torch.norm(att1._weight_matrix, p='fro').item()
    changes['arc_attention_bias'] = torch.norm(att2._bias   - att1._bias,   p=2).item()        / torch.norm(att1._bias,   p=2).item()

    # tag_bilinear
    tb1 = dec1.tag_bilinear
    tb2 = dec2.tag_bilinear
    changes['tag_bilinear_weight'] = torch.norm(tb2.weight - tb1.weight, p='fro').item() / torch.norm(tb1.weight, p='fro').item()
    changes['tag_bilinear_bias']   = torch.norm(tb2.bias   - tb1.bias,   p=2).item()        / torch.norm(tb1.bias,   p=2).item()

    # head_sentinel
    hs1 = dec1._head_sentinel
    hs2 = dec2._head_sentinel
    changes['head_sentinel'] = torch.norm(hs2 - hs1, p='fro').item() / torch.norm(hs1, p='fro').item()

    return changes


def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def visualize_decoder_weight_changes(model1, model2, hex_color, title="Decoder Weight Changes"):
    """
    heatmap of decoder‐weight changes. hex_color should be a CSS‐style hex string, e.g. "#4A90E2".
    """
    changes = get_decoder_weight_changes(model1, model2)
    components = [c for c in changes.keys() if not c.startswith('lstm')]
    data = np.array([changes[c] for c in components]).reshape(len(components), 1)

    nice_labels = {
        'head_arc_mlp_weight': 'W_arc-head',
        'head_arc_mlp_bias': 'b_arc-head',
        'child_arc_mlp_weight': 'W_arc-dep',
        'child_arc_mlp_bias': 'b_arc-dep',
        'head_tag_mlp_weight': 'W_tag-head',
        'head_tag_mlp_bias': 'b_tag-head',
        'child_tag_mlp_weight': 'W_tag-dep',
        'child_tag_mlp_bias': 'b_tag-dep',
        'arc_attention_weight': 'S_arc weight',
        'arc_attention_bias': 'S_arc bias',
        'tag_bilinear_weight': 'S_tag weight',
        'tag_bilinear_bias': 'S_tag bias',
        'head_sentinel': 'Head sentinel'
    }

    plt.figure(figsize=(4, len(components) * 0.5 + 1))
    cmap = sns.light_palette(hex_color, as_cmap=True)
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        yticklabels=[nice_labels[c] for c in components],
        cbar_kws={'label': 'Relative change'}
    )
    ax.set_xticks([])
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def visualize_weight_changes(model1, model2, hex_color="#FF0000", title="Weight Changes After Fine‑tuning"):
    """
    Visualize weight changes between two BERT models as a heatmap.
    
    Args:
      model1, model2: two BERT models
      hex_color (str): base color in hex (e.g. "#1f77b4") for the palette
      title (str): plot title
    """
    changes = get_weight_changes(model1, model2)

    # prepare data
    num_layers = len(model1.text_field_embedder._token_embedders['tokens'].transformer_model.encoder.layer)
    components = ["query", "key", "value", "attn_output", "ffn_intermediate", "ffn_output"]
    data = np.zeros((len(components), num_layers))
    for i, comp in enumerate(components):
        data[i] = changes[comp]

    nice_labels = {
        "query": "Query",
        "key": "Key",
        "value": "Value",
        "attn_output": "Attention Output",
        "ffn_intermediate": "FFN Intermediate",
        "ffn_output": "FFN Output"
    }

    # build a single‑hue palette from white → hex_color
    cmap = sns.light_palette(hex_color, as_cmap=True)

    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        xticklabels=[f"Layer {i}" for i in range(num_layers)],
        yticklabels=[nice_labels[c] for c in components]
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    return plt.gcf()


deu_model = load_model('../../from_das/1_deu/model.pkl')
deu_model.cpu()

print(f'[{now()}] German model loaded.')


second_stage_models = (
    ('2_nld', '#DE8900', 'Dutch'),
    ('2_swe', '#0072B2', 'Swedish'),
    ('2_ces', '#B30000', 'Czech'),
    ('2_hun', '#008E67', 'Hungarian')
)

for name, color, adj in second_stage_models:
    model = load_model(f'../../from_das/{name}/model.pkl')
    model.cpu()
    print(f'[{now()}] {adj} model loaded.')
    plot = visualize_decoder_weight_changes(deu_model, model, hex_color=color,
                                    title=f'Decoder Weight Changes after Fine-Tuning on {adj} Dataset')
    plot.savefig(f'outputs/comparison_{name[2:]}2.pdf', format='pdf', bbox_inches='tight')
    print(f'[{now()}] {adj} plot saved.')


def visualize_weight_changes_multilang(models, lang_colors, title="Multilingual Weight Changes"):
    """
    Draw a grid of components × layers where each cell contains a tiny barplot
    of relative changes across languages.
    
    models: list of (name, model) pairs, first one is the “reference” (German), 
            the rest are the second-stage models in the order you want bars.
    lang_colors: dict mapping model‑name → hex color
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) collect data: shape (n_components, n_layers, n_langs)
    ref_name, ref_model = models[0]
    other = models[1:]
    comp_names = ["query","key","value","attn_output","ffn_intermediate","ffn_output"]
    n_comp = len(comp_names)
    n_layers = len(ref_model.text_field_embedder._token_embedders['tokens'].transformer_model.encoder.layer)
    n_langs = len(other)

    data = np.zeros((n_comp, n_layers, n_langs))
    for l_idx, (lang_name, lang_model) in enumerate(other):
        changes = get_weight_changes(ref_model, lang_model)
        for c_idx, comp in enumerate(comp_names):
            data[c_idx, :, l_idx] = changes[comp]

    # 2) set up figure
    fig, axes = plt.subplots(n_comp, n_layers, figsize=(n_layers*1.2, n_comp*1.2), 
                             sharex=True, sharey=True)
    for i in range(n_comp):
        for j in range(n_layers):
            ax = axes[i,j]
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # draw no background; we'll draw bars only
            ax.set_frame_on(False)
            # 3) inset barplot
            vals = data[i,j,:]
            xs = np.arange(n_langs)
            ax.bar(xs, vals, width=0.8, color=[lang_colors[name] for name,_ in other])
            # optionally: small horizontal line at zero
            ax.axhline(0, color='k', linewidth=0.3)
            # limit to [0, max] so bars fill cell
            ax.set_ylim(0, data.max()*1.05)

            # label left edge and bottom edge
            if j==0:
                ax.set_ylabel(comp_names[i], rotation=0, labelpad=20, va='center')
            if i==n_comp-1:
                ax.set_xlabel(f"L{j}", labelpad=5)

    fig.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.96])
    return fig


def visualize_decoder_changes_multilang(models, lang_colors, title="Decoder Changes by Language"):
    """
    For each decoder component (row), draw a tiny barplot of relative change
    across languages (columns), colored by lang_colors.
    
    models: list of (lang_name, model) pairs. First is the German reference.
    lang_colors: dict mapping lang_name → hex color
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # reference + others
    ref_name, ref_model = models[0]
    others = models[1:]
    comp_names = list(get_decoder_weight_changes(ref_model, others[0][1]).keys())
    n_comp = len(comp_names)
    n_lang = len(others)

    # collect data: shape (n_comp, n_lang)
    data = np.zeros((n_comp, n_lang))
    for j, (lang, m) in enumerate(others):
        changes = get_decoder_weight_changes(ref_model, m)
        for i, comp in enumerate(comp_names):
            data[i, j] = changes[comp]

    # plot
    fig, axes = plt.subplots(n_comp, 1, figsize=(4, n_comp*0.6), sharex=True)
    for i, ax in enumerate(axes):
        vals = data[i]
        xs = np.arange(n_lang)
        ax.bar(xs, vals, color=[lang_colors[lang] for lang,_ in others], width=0.8)
        ax.set_ylabel(comp_names[i], rotation=0, labelpad=10, va='center')
        ax.set_xticks([])

    # bottom axis: language labels
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([lang for lang,_ in others], rotation=45, ha='right')
    fig.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.95])
    return fig


# nld_model = load_model('../../from_das/2_nld/model.pkl')
# nld_model.cpu()

# print(f'[{now()}] Dutch model loaded.')

# swe_model = load_model('../../from_das/2_swe/model.pkl')
# swe_model.cpu()

# print(f'[{now()}] Swedish model loaded.')

# ces_model = load_model('../../from_das/2_ces/model.pkl')
# ces_model.cpu()

# print(f'[{now()}] Czech model loaded.')

# hun_model = load_model('../../from_das/2_hun/model.pkl')
# hun_model.cpu()

# print(f'[{now()}] Hungarian model loaded.')

# models = [
#     ("German", deu_model),
#     ("Dutch", nld_model),
#     ("Swedish", swe_model),
#     ("Czech", ces_model),
#     ("Hungarian", hun_model),
# ]
# colors = {"Dutch":"#DE8900","Swedish":"#0072B2","Czech":"#B30000","Hungarian":"#008E67"}

# fig = visualize_decoder_changes_multilang(models, colors,
#         title="Relative BERT‑Layer Changes by Language")
# fig.savefig("outputs/multilang_decoder_heatmap.pdf", bbox_inches="tight")
