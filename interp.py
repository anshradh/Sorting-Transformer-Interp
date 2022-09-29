# %%
import torch
from torch.utils.data import DataLoader
from SortingTransformer.training_sorting_gpt2 import (
    sort_gpt_attn_only_config,
    test_dataset,
)

import w2d4_attn_only_transformer

from einops import rearrange, reduce

import torch.nn.functional as F

import numpy as np
import plotly.express as px

device = torch.device("cpu")
MAIN = __name__ == "__main__"

# %%

# Load the previously trained model
if MAIN:
    trained_model = w2d4_attn_only_transformer.AttnOnlySortingTransformer(
        sort_gpt_attn_only_config
    )
    trained_model.load_state_dict(
        torch.load("one_head_easy_trained_attn_only_sort_gpt2.pt")
    )
    trained_model.eval().to(device)
# %%
def run_model_on_data(model, dataset, batch_size=32):
    """
    Run the model on the dataset and return the logits and labels.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits = []
    labels = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits.append(model(x)[:, 9:])
        labels.append(y[:, 9:])
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    return logits, labels


if MAIN:
    logits, labels = run_model_on_data(trained_model, test_dataset)
    n_correct = (
        (
            rearrange(logits.argmax(dim=-1), "b s -> (b s)")
            == rearrange(labels, "b s -> (b s)")
        )
        .double()
        .sum()
    )
    print(f"Accuracy: {n_correct / (logits.shape[0] * logits.shape[1]):.5f}")
    incorrect_indices = (logits.argmax(dim=-1) != labels).any(dim=-1)
    print(
        "Sample of incorrect examples: ",
        "\n",
        logits.argmax(dim=-1)[incorrect_indices][:5],
        "\n",
        labels[incorrect_indices][:5],
    )
## From this, it looks like the model struggles when digits are repeated 3 or more times. It gets confused and then outputs an adjacent number an extra time instead of the correct number the right number of times.
# %%
cache = {}
if MAIN:
    tokens = test_dataset[0][0].unsqueeze(0)
    trained_model.cache_all(cache)
    logits = trained_model(tokens)
    trained_model.reset_hooks()

# %%
def to_numpy(tensor):
    """Helper function to convert things to numpy before plotting with Plotly."""
    return tensor.detach().cpu().numpy()


def plot_logit_attribution(logit_attr, tokens):
    y_labels = tokens[:-1]
    x_labels = ["Direct"] + [
        f"L{l}H{h}"
        for l in range(sort_gpt_attn_only_config["n_layers"])
        for h in range(sort_gpt_attn_only_config["n_heads"])
    ]
    px.imshow(
        to_numpy(logit_attr),
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
    ).show()


def logit_attribution(
    embed,
    l1_results,
    W_U,
    tokens,
):
    """
    W_U_to_logits' is a (position, d_model) tensor where each row is the unembed for the correct NEXT token at the current position.

     Inputs:
         embed: (position, d_model)
         l1_results: (position, head_index, d_model)
         W_U: (d_vocab, d_model)
     Returns:
         Tensor representing the concatenation (along dim=-1) of logit attributions from the direct path (position-1,1) and layer 0 logits (position-1, n_heads)
    """
    W_U_to_logits = W_U[tokens[1:], :]
    direct_path_logits = torch.einsum("pm,pm->p", W_U_to_logits, embed[:-1, :])
    l1_logits = torch.einsum("pm,pim->pi", W_U_to_logits, l1_results[:-1])
    logit_attribution = torch.concat([direct_path_logits[:, None], l1_logits], dim=-1)
    return logit_attribution


# %%
if MAIN:
    batch_index = 0
    embed = cache["hook_embed"][batch_index]
    l0_results = cache["blocks.0.attn.hook_result"][batch_index]
    logit_attr = logit_attribution(
        embed.cpu(),
        l0_results.cpu(),
        trained_model.unembed.W_U,
        tokens[batch_index],
    )
    plot_logit_attribution(
        logit_attr, [f"{v}_{i}" for i, v in enumerate(list(tokens[0].numpy()))]
    )

# %%
def plot_attn_pattern(patterns, tokens, title=None):
    # Patterns has shape [head_index, query_pos, key_pos] or [query_pos, key_pos]
    if len(patterns.shape) == 3:
        px.imshow(
            to_numpy(patterns),
            animation_frame=0,
            y=tokens,
            x=tokens,
            labels={"x": "Key", "y": "Query"},
            color_continuous_scale="Blues",
            title=title,
        ).update_yaxes(nticks=len(tokens)).show()
    else:
        px.imshow(
            to_numpy(patterns),
            y=tokens,
            x=tokens,
            labels={"x": "Key", "y": "Query"},
            color_continuous_scale="Blues",
            title=title,
        ).update_yaxes(nticks=len(tokens)).show()


if MAIN:
    for layer in range(sort_gpt_attn_only_config["n_layers"]):
        plot_attn_pattern(
            cache[f"blocks.{layer}.attn.hook_attn"][0],
            [f"{v}_{i}" for i, v in enumerate(list(tokens[0].numpy()))],
            f"Layer {layer} attention patterns",
        )
# The model basically looks for the digit it's currently querying as the key, or the preceding one(s) in terms of attention relevance - we'll see if we cna verify this with the QK circuit later on.
# %%
cfg = sort_gpt_attn_only_config


def current_attn_detector(cache):
    current_attn_score = torch.zeros(cfg["n_layers"], cfg["n_heads"])
    for layer in range(cfg["n_layers"]):
        attn = cache[f"blocks.{layer}.attn.hook_attn"]
        current_attn_score[layer] = reduce(
            attn.diagonal(dim1=-2, dim2=-1),
            "batch head_index pos -> head_index",
            "mean",
        )
    return current_attn_score


def prev_attn_detector(cache):
    prev_attn_score = torch.zeros(cfg["n_layers"], cfg["n_heads"])
    for layer in range(cfg["n_layers"]):
        attn = cache[f"blocks.{layer}.attn.hook_attn"]
        prev_attn_score[layer] = reduce(
            attn.diagonal(dim1=-2, dim2=-1, offset=-1),
            "batch head_index pos -> head_index",
            "mean",
        )
    return prev_attn_score


def first_attn_detector(cache):
    first_attn_score = torch.zeros(cfg["n_layers"], cfg["n_heads"])
    for layer in range(cfg["n_layers"]):
        attn = cache[f"blocks.{layer}.attn.hook_attn"]
        first_attn_score[layer] = reduce(
            attn[:, :, :, 0], "batch head_index pos -> head_index", "mean"
        )
    return first_attn_score


def plot_head_scores(scores_tensor, title=""):
    px.imshow(
        to_numpy(scores_tensor),
        labels={"y": "Layer", "x": "Head"},
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    ).show()


if MAIN:
    current_attn_scores = current_attn_detector(cache)
    plot_head_scores(current_attn_scores, "Current Token Heads")
    prev_attn_scores = prev_attn_detector(cache)
    plot_head_scores(prev_attn_scores, "Prev Token Heads")
    first_attn_scores = first_attn_detector(cache)
    plot_head_scores(first_attn_scores, "First Token Heads")

##
# The head doesn't seem to be doing anything particularly close to any of these patterns.

# %%
def ablated_head_run(
    model: w2d4_attn_only_transformer.AttnOnlySortingTransformer,
    tokens: torch.Tensor,
    layer: int,
    head_index: int,
):
    def ablate_head_hook(value, hook):
        value[:, :, head_index, :] = 0.0
        return value

    logits = model.run_with_hooks(
        tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_v", ablate_head_hook)]
    )
    return logits


def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = torch.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -(pred_log_probs.mean())


if MAIN:
    original_loss = cross_entropy_loss(logits, tokens)
    ablation_scores = torch.zeros((cfg["n_layers"], cfg["n_heads"]))
    for layer in range(cfg["n_layers"]):
        for head_index in range(cfg["n_heads"]):
            ablation_scores[layer, head_index] = (
                cross_entropy_loss(
                    ablated_head_run(trained_model, tokens, layer, head_index), tokens
                )
                - original_loss
            )
    plot_head_scores(ablation_scores)
# Nothing super revelaing here.
# %%
## OV Circuit
if MAIN:
    W_O_0 = trained_model.blocks[0].attn.W_O[0]
    W_V_0 = trained_model.blocks[0].attn.W_V[0]
    W_E = trained_model.embed.W_E
    W_U = trained_model.unembed.W_U
    W_pos = trained_model.pos_embed.W_pos
    OV_0_vocab_circuit = W_U @ W_O_0 @ W_V_0 @ W_E
    OV_0_pos_circuit = W_U @ W_O_0 @ W_V_0 @ W_pos
    px.imshow(to_numpy(OV_0_vocab_circuit)).show()
    px.imshow(to_numpy(OV_0_pos_circuit)).show()
    # The OV vocab circuit basically shows that the model has learned ordering (if 0 is attended to, that negatively impacts the likelihood of 9 and vice-versa).
# %%
# QK circuit
def mask_scores(attn_scores):
    """Mask the attention scores so that tokens don't attend to previous tokens."""
    mask = torch.tril(torch.ones_like(attn_scores)).bool()
    neg_inf = torch.tensor(-1e4).to(attn_scores.device)
    masked_attn_scores = torch.where(mask, attn_scores, neg_inf)
    return masked_attn_scores


if MAIN:
    W_Q_0 = trained_model.blocks[0].attn.W_Q[0]
    W_K_0 = trained_model.blocks[0].attn.W_K[0]
    W_pos = trained_model.pos_embed.W_pos
    pos_by_pos_0 = (
        torch.cat((W_E, W_pos), -1).T
        @ W_Q_0.T
        @ W_K_0
        @ torch.cat((W_E, W_pos), -1)
        / np.sqrt(cfg["d_head"])
    )
    pos_by_pos_0[10:, 10:] = mask_scores(pos_by_pos_0[10:, 10:])
    pos_by_pos_pattern_0 = F.softmax(pos_by_pos_0, dim=-1)
    labels = [f"tok_{i}" for i in range(cfg["d_vocab"])] + [
        f"pos_{i}" for i in range(cfg["n_ctx"])
    ]
    px.imshow(
        to_numpy(pos_by_pos_pattern_0),
        labels={"y": "Query", "x": "Key"},
        y=labels,
        x=labels,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    ).update_yaxes(nticks=29).update_xaxes(nticks=29).show()
# The QK circuit confirms that the model learns the ordering of digits and generally pays attention to the digit itself the most, then to digits close in proximity. Also very neatly, the model learns to place a lot of attention on the 0 token at the first position of the output sequence and a lot of attention on the 9 token at the last position of the output sequence.
# %%
# one kind of adversarial attack is to see how the model does with big skips - it seems to rely a lot on "closeness" of digits to figure out ordering, so perhaps it will suck when there are large jumps.

adversarial_input = torch.tensor(
    [[9, 7, 9, 0, 8, 7, 9, 7, 0, 0, 0, 0, 0, 7, 7, 7, 8, 9, 9]]
)
adversarial_pred = trained_model(adversarial_input).argmax(-1)
print(adversarial_input[0, :10].sort(-1).values)
print(adversarial_pred[0, 9:])

# Yup, big jumps do indeed kind of suck - hard to tell if this is because big jumps imply repeats or not, though.
# %%
