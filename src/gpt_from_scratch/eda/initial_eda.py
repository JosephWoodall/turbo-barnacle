import os 

#os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
with open("src/gpt_from_scratch/raw_data/input.txt", "r") as f:
    text = f.read()
print("Length of text: {} characters".format(len(text)))
print(text[:1000])

'''
This will help us determine the vocabulary size, and convert the characters into a numerical representation for the transformer model to understand.
'''
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

'''
Creates a mapping from characters to integers (character level tokenizer); this will give us long sequences, although, it's a simple tokenizer
'''
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encode a string s to a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decode a list of integers l to a string
print(encode("hii there"))
print(decode(encode("hii there")))

'''
The below will encode the entire text dataset and store it into a torch.tensor
'''
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

'''
This will separate the data into a training and validation set
'''
n = int(0.9*len(data)) # first 90% of the data will be train, rest will be val
train_data = data[:n]
val_data = data[n:]

block_size = 8 
print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is: {context} the target is: {target}")
    

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel
block_size=8 # what is the maximum context length for predictions

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs: ', xb.shape)
print(xb) # the input to the transformer
print(xb.dtype)
print('targets: ', yb.shape)
print(yb)
print(yb.dtype)

print("-------")

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is: {context.tolist()} the target is: {target}")

print("-------")
        
print(xb) # the input to the transformer

'''
Now it's time to feed the xb object into a simple language model, the Bigram Language Model, using torch
'''
print("-------Bigram Language Model-------")
import torch.nn as nn 
from torch.nn import functional as F

torch.manual_seed(1337)

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
            
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1, 1), dtype=torch.long) # (B=1, T=1) tensor
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

'''
The bigram model was totally random, and so that's why you get this output of Sr?qP-QWktXoL&jLDJgOLVz'RIoDqHdhsV&vLLxatjscMpwLERSPyao.qfzs$Ys$zF-w,;eEkzxjgCKFChs!iWW.ObzDnxA Ms$3
But, now we can optimize it
'''
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # you can even set the learning rate much higher for this bigram model
batch_size=32
for steps in range(10000):
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward() 
    optimizer.step()
    
print(loss.item())
print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()))

'''
So even though this was a pretty rough optimization process, we can see that the model is improving, the loss is down quite dramatically, and the output is relatively better than earlier. Good.
'''

print("There is a mathematical trick in self-attention: ")
print("The simplist way for tokens to talk to estimate an average of all the proceeding elements. This is called the query, key, value mechanism.")
# here is a toy example 
B, T, C = 4, 8, 2 # batch size, time, and channel
x = torch.randn(B, T, C) # (B, T, C) tensor
print(x.shape)

print("We want x[b, t] to be a weighted average of all the proceeding elements, x[b, :t]")
xbow = torch.zeros((B, T, C)) # (B, T, C) tensor
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C) tensor
        xbow[b, t] = torch.mean(xprev, 0)
        print(x[0])
        print(xbow[b, t])
        
print("The above is a very inefficient way to do this, and so we can use the query, key, value mechanism to do this more efficiently.")
print("----")
print("Version 1")
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a/torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print('a=')
print(a)
print('---')
print('b=')
print(b)
print('---')
print('c=')
print(c)

print("----")
print("Version 2")
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1,keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) -> (B, T, C)
print(torch.allclose(xbow, xbow2))
print(xbow[0])
print(xbow2[0])

print("----")
print("Version 3")
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril==0, float("-inf"))
wei = F.softmax(wei, dim=1)
xbow3=wei @ x 
print(torch.allclose(xbow, xbow3))