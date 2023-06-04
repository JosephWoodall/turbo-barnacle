import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel
block_size = 256  # what is the maximum context length for predictions
max_iters = 5000  # how many training iterations
eval_interval = 500  # how often to evaluate our model on the validation set
learning_rate = 3e-4  # how fast should our model learn?
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # is gpu available?
eval_iters = 200  # how many iterations to use when evaluating on the validation set
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# --- data ---
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("src/gpt_from_scratch/raw_data/input.txt", "r") as f:
    text = f.read()

# --- vocabulary ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers (character level tokenizer)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encode a string s to a list of integers
def encode(s): return [stoi[c] for c in s]
# decode a list of integers l to a string
def decode(l): return ''.join([itos[i] for i in l])


# --- data preparation ---
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9*len(data))  # first 90% of the data will be train, rest will be val
train_data = data[:n]
val_data = data[n:]

# --- model ---


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, T, C) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Simple Bigram Model


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # the final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T, = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # (B*T, C); this will conform to what torch expects
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # negative loss likelihood; which is cross entropy loss; measures the quality of the logits with respect to the targets
            loss = F.cross_entropy(logits, targets)

        # (B, T, C); which is Batch (4), Time (8), and Channel (65) tensors of integers, and is the score, or next character in a sequence
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C) tensor
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C) tensor
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) tensor
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1) tensor
        return idx


model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()), 'M parameters')

# creating the pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while we will evaluate our model on the validation set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the lsos
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- generation ---
context = torch.zeros((1, 1), dtype=torch.long,
                      device=device)  # (B=1, T=1) tensor
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('decoder_transformer_output.txt', 'w').write(decode(m.generate(context(max_new_tokens=10000)[0].tolist())))
