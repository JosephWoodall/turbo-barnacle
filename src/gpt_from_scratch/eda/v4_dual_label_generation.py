import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import defaultdict
import re
import random
from faker import Faker

import json

class WordTokenizer:
    """
    WordTokenizer class is responsible for tokenizing input text into individual words.
    """
    def __init__(self):
        self.word_regex = re.compile(r'\w+')

    def tokenize(self, text):
        tokens = self.word_regex.findall(text)
        return tokens

class KeyValueDataset(Dataset):
    """
    KeyValueDataset class represents a PyTorch dataset. It takes a list of key-value pairs and a tokenizer as inputs.
    It tokenizes the input and output text using the tokenizer.
    """
    def __init__(self, key_value_pairs, tokenizer):
        self.key_value_pairs = key_value_pairs
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.key_value_pairs)
    
    def __getitem__(self, index):
        key, value = self.key_value_pairs[index]
        input_tokens = self.tokenizer.tokenize(key)
        output_tokens = self.tokenizer.tokenize(value)
        return input_tokens, output_tokens

def collate_fn(batch):
    """
    collate_fn is a helper function used by the DataLoader to collate and preprocess the batch data.
    """
    input_seqs, output_seqs = zip(*batch)
    input_lengths = [len(seq) for seq in input_seqs]
    output_lengths = [len(seq) for seq in output_seqs]
    max_input_length = max(input_lengths)
    max_output_length = max(output_lengths)

    padded_input_seqs = []
    padded_output_seqs = []

    for input_seq, output_seq in zip(input_seqs, output_seqs):
        input_padding = [0] * (max_input_length - len(input_seq))
        padded_input_seqs.append(input_seq + input_padding)

        output_padding = [0] * (max_output_length - len(output_seq))
        padded_output_seqs.append(output_seq + output_padding)

    input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in tokens] for tokens in padded_input_seqs]).transpose(0, 1)  # Transpose for transformer input
    output_tensor = torch.tensor([[output_vocab.get(token, 0) for token in tokens] for tokens in padded_output_seqs]).transpose(0, 1)  # Transpose for transformer input

    return input_tensor, output_tensor


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model class represents the main model architecture.
    It takes input vocabulary size, output size, number of layers, hidden size, number of attention heads, and dropout as inputs.
    The model consists of an embedding layer, transformer encoder layers, and a linear layer for prediction.

    """
    def __init__(self, input_vocab_size, output_size, num_layers, hidden_size, num_heads, dropout):
        super(GPT, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if input_vocab_size > 0:
            self.embedding = nn.Embedding(input_vocab_size, hidden_size, padding_idx=0)
        else:
            self.embedding = None

        
        self.encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

        
    def forward(self, x):
        if self.embedding is not None:
            x = self.embedding(x)
        x = x.transpose(0, 1)  # Transpose for transformer input
        x = x.contiguous()  # Convert to contiguous tensor
        encoder_output = self.encoder(x)
        output = self.linear(encoder_output[-1])
        return output
    

fake = Faker()

class FakeDataGenerator:
    def __init__(self):
        self.fake = Faker()
    
    def generate_company(self):
        return self.fake.company()
    
    def generate_year(self):
        return str(self.fake.random_int(2020, 2025))
    
    def generate_document(self):
        return self.fake.text(max_nb_chars=50)
    
    def generate_group(self):
        return self.fake.word()
    
    def generate_topic(self):
        return self.fake.word()

fake_generator = FakeDataGenerator()

sample_dictionary = {
    'What is the revenue for Company A on 2023-01-02': 'Table',
    'What is the main point of this article?': 'Document',
    'What is the reporting date for all customers under the e-commerce group?': 'Table',
    'Can you summarize this article for me?': 'Document',
    'What is the revenue for Company B on 2023-01-02': 'Table',
}

_templates = [
    "What is the revenue for {company} in {year}?",
    "What is the main point of {document}?",
    "What is the reporting date for all customers under the {group} group?",
    "Can you summarize {document} for me?",
    "What is the revenue growth rate for {company}?",
    "How many articles are there about {topic}?",
    "Whats the revenue for {company} in {year}?",
    "Whats the main point of {document}?",
    "Whats the reporting date for all customers under the {group} group?",
    "Could you summarize {document} for me?",
]

# Create a word tokenizer
tokenizer = WordTokenizer()

# Prepare the dataset and vocabulary
key_value_pairs = [(key, value) for key, value in sample_dictionary.items()]
tokenized_dataset = KeyValueDataset(key_value_pairs, tokenizer)

# Populate the input and output vocabularies
input_vocab = defaultdict(lambda: len(input_vocab))  # Assign unique indices to each token
output_vocab = defaultdict(lambda: len(output_vocab))  # Assign unique indices to each token

for input_tokens, output_tokens in tokenized_dataset:
    if input_tokens:
        for token in input_tokens:
            input_vocab[token]
    if output_tokens:
        for token in output_tokens:
            output_vocab[token]

# Convert the vocabularies to regular dictionaries
input_vocab = dict(input_vocab)
output_vocab = dict(output_vocab)

if len(input_vocab) == 0 or len(output_vocab) == 0:
    print("Input or output vocabulary is empty. Please provide a dictionary with non-empty keys.")

# Define the hyperparameters
num_layers = 4
hidden_size = 128
num_heads = 32
dropout = 0.0
batch_size = 12
learning_rate = 1e-4
num_epochs = 40

# Create an instance of the GPT model
model = GPT(len(input_vocab), len(output_vocab), num_layers, hidden_size, num_heads, dropout)

total_params = sum(p.numel() for p in model.parameters())
print("-----------------------------------------------")
print(f"Total number of parameters: {total_params}")
print("-----------------------------------------------")

# Create a DataLoader for training
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

generated_data = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for input_tokens, output_tokens in dataloader:
        input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in tokens] for tokens in input_tokens]).transpose(0, 1)  # Transpose for transformer input
        output_tensor = torch.tensor([[output_vocab.get(token, 0) for token in tokens] for tokens in output_tokens]).transpose(0, 1)  # Transpose for transformer input

        optimizer.zero_grad()

        # Forward pass
        input_output = model(input_tensor)

        # Flatten the output and target tensors for computing the loss
        input_output = input_output.view(-1, len(output_vocab))
        output_tensor = output_tensor.view(-1)

        # Compute the loss
        loss = criterion(input_output, output_tensor)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for the epoch
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")

    random_template = random.choice(_templates)
    company = fake_generator.generate_company()
    year = fake_generator.generate_year()
    document = fake_generator.generate_document()
    group = fake_generator.generate_group()
    topic = fake_generator.generate_topic()

    prompt = random_template.format(
        company=company,
        year=year,
        document=document,
        group=group,
        topic=topic,
    )
    input_tokens = tokenizer.tokenize(prompt)
    input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in input_tokens]]).transpose(0, 1)  # Transpose for transformer input
    with torch.no_grad():
        output_tensor = model(input_tensor.transpose(0, 1))
    predicted_indices = output_tensor.argmax(dim=-1).tolist()[0]

    # Convert the predicted indices back to their original representations
    predicted_value = list(output_vocab.keys())[predicted_indices]

    # Print the generated prompt and the corresponding generated value
    print(f"Generated prompt: '{prompt}'")
    print(f"Generated value: '{predicted_value}'")
    print()
    
    generated_data.append({
        "question":prompt,
        "label":predicted_value
    })
    
# Save generated data to a JSON file
output_file = "src/gpt_from_scratch/eda/generated_data.json"
with open(output_file, "w") as f:
    json.dump(generated_data, f, indent=4)
    

"""
Inference for later on!
"""
# Generate a key-value pair using the trained model
prompt = "What is the revenue for Company A in 2023?"
input_tokens = tokenizer.tokenize(prompt)
input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in input_tokens]]).transpose(0, 1)  # Transpose for transformer input
with torch.no_grad():
    output_tensor = model(input_tensor.transpose(0, 1))
predicted_indices = output_tensor.argmax(dim=-1).tolist()[0]

# Convert the predicted indices back to their original representations
predicted_value = list(output_vocab.keys())[predicted_indices]

# Print the generated prompt and the corresponding generated value
print(f"Generated prompt: '{prompt}'")
print(f"Generated value: '{predicted_value}'")
