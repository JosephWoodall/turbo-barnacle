import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel
block_size = 8 # what is the maximum context length for predictions
max_iters = 1000 # how many training iterations
eval_interval = 100 # how often to evaluate our model on the validation set
learning_rate = 1e-2 # how fast should our model learn?
device = 'cuda' if torch.cuda.is_available() else 'cpu' # is gpu available?
eval_iters = 200 # how many iterations to use when evaluating on the validation set

torch.manual_seed(1337)

# --- data ---
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("src/gpt_from_scratch/raw_data/input.txt", "r") as f:
    text = f.read()

# --- vocabulary ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers (character level tokenizer)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encode a string s to a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decode a list of integers l to a string

# --- data preparation ---
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9*len(data)) # first 90% of the data will be train, rest will be val
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

# Simple Bigram Model

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)
        
        if targets is None: 
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C); this will conform to what torch expects
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # negative loss likelihood; which is cross entropy loss; measures the quality of the logits with respect to the targets
                
        return logits, loss # (B, T, C); which is Batch (4), Time (8), and Channel (65) tensors of integers, and is the score, or next character in a sequence
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction 
            logits, loss = self(idx)
            # focus only on the last time step 
            logits = logits[:, -1, :] # becomes (B, C) tensor
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) tensor
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) tensor
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) tensor
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# creating the pytorch optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while we will evaluate our model on the validation set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data 
    xb, yb = get_batch('train')
    
    # evaluate the lsos
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# --- generation ---
context = torch.zeros((1, 1), dtype=torch.long, device=device) # (B=1, T=1) tensor
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))