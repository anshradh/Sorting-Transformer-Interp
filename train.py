# %%
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import wandb

import w2d4_attn_only_transformer

MAIN = __name__ == "__main__"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
class SortDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    From: https://github.com/karpathy/minGPT/blob/master/generate.ipynb
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {"train", "test"}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000  # ...

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):

        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rare
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unique digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = (
                "test" if h % 4 == 0 else "train"
            )  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok

        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[: self.length - 1] = -1
        return x, y


# %%
train_dataset = SortDataset("train", length=10, num_digits=10)
test_dataset = SortDataset("test", length=10, num_digits=10)
if MAIN:
    x, y = train_dataset[0]
    for a, b in zip(x, y):
        print(int(a), int(b))
# %%
def train(model, config_dict, train_dataset, test_dataset):
    """
    Main training loop, nothing particularly special here
    """
    wandb.init(project="sorting-gpt2", config=config_dict)
    config = wandb.config
    print(f"Training with config: {config}")
    device = config.device
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    examples_seen = 0
    for epoch in range(config.num_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            train_loss = nn.CrossEntropyLoss(ignore_index=-1)(
                rearrange(logits, "b l v -> (b l) v"), rearrange(y, "b l -> (b l)")
            )
            preds = rearrange(logits, "b l v -> (b l) v").argmax(dim=-1)
            reshaped_targets = rearrange(y, "b l -> (b l)")
            actual_targets = reshaped_targets[reshaped_targets != -1]
            actual_preds = preds[reshaped_targets != -1]
            train_acc = (actual_preds == actual_targets).float().mean()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            examples_seen += x.shape[0]
        print(
            f"Epoch {epoch + 1} / {config.num_epochs} loss {train_loss.item():.3f} acc {train_acc.item():.3f}"
        )
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_loss = nn.CrossEntropyLoss(ignore_index=-1)(
                    rearrange(logits, "b l v -> (b l) v"), rearrange(y, "b l -> (b l)")
                )
                preds = rearrange(logits, "b l v -> (b l) v").argmax(dim=-1)
                reshaped_targets = rearrange(y, "b l -> (b l)")
                actual_targets = reshaped_targets[reshaped_targets != -1]
                actual_preds = preds[reshaped_targets != -1]
                test_acc = (actual_preds == actual_targets).float().mean()
                print(f"Test loss {test_loss.item():.3f} acc {test_acc.item():.3f}")
                break
        model.train()
        wandb.log(
            dict(
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                step=examples_seen,
            )
        )
    wandb.finish()
    return model


# %%
sort_gpt_attn_only_config = {
    "d_model": 128,
    "d_head": 128,
    "n_heads": 1,
    "d_mlp": 512,
    "n_layers": 1,
    "n_ctx": 19,
    "eps": 1e-5,
    "d_vocab": 10,
    "act_fn": "GeLU",
    "use_attn_scale": True,
    "use_local_attn": False,
    "use_attn_result": True,
}
if MAIN:
    train_config = dict(
        learning_rate=8e-4,
        batch_size=128,
        num_epochs=30,
        device=device,
    )
    model = w2d4_attn_only_transformer.AttnOnlySortingTransformer(
        sort_gpt_attn_only_config
    )
    model = train(model, train_config, train_dataset, test_dataset)
    torch.save(model.state_dict(), "one_head_easy_trained_attn_only_sort_gpt2.pt")

# %%
