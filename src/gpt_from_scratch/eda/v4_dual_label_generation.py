import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import defaultdict
import re
import random
from faker import Faker
import yaml
import json


"""
The code below implements a Generative Pre-trained Transformer (GPT) model for generating text based on a given prompt. The code can be summarized as follows:
- Import necessary libraries and modules, including torch, torch.nn, TransformerEncoder, DataLoader, AdamW, defaultdict, re, random, faker, and json.
- Define two classes: "WordTokenizer" and "KeyValueDataset." The "WordTokenizer" class tokenizes input text into individual words using a regular expression. The "KeyValueDataset" class represents a PyTorch dataset that tokenizes input and output text using the provided tokenizer.
- Define a helper function "collate_fn" to preprocess and collate the batch data for the DataLoader.
- Implement the main GPT model in the "GPT" class, which consists of an embedding layer, transformer encoder layers, and a linear layer for prediction. The model takes input vocabulary size, output size, number of layers, hidden size, number of attention heads, and dropout as inputs.
- Create an instance of the Faker library to generate fake data for demonstration purposes.
- Define a sample dictionary to store input-output pairs for training.
- Provide a list of templates for generating prompts.
- Create a word tokenizer using the "WordTokenizer" class.
- Prepare the dataset and vocabulary based on the sample dictionary and tokenizer.
- Define hyperparameters such as the number of layers, hidden size, number of heads, dropout, batch size, learning rate, and number of epochs.
- Create an instance of the GPT model with the specified hyperparameters.
- Create a DataLoader to handle the training data, using the defined collate function.
- Define the optimizer (AdamW) and loss function (CrossEntropyLoss).
- Start the training loop, where the model is trained for the specified number of epochs. The average loss is printed for each epoch.
- During each epoch, a random template is selected, and fake data is generated based on the template using the Faker library. The prompt is tokenized and fed into the trained model, and the predicted value is obtained. The generated prompt and corresponding value are printed.
- The generated data is stored in a list.
- Finally, the generated data is saved to a JSON file.

The code also includes an inference section to demonstrate the generation of text based on a prompt using the trained model. The prompt is tokenized, passed through the model, and the predicted value is obtained and printed.
"""


"""
Class Definition
"""


class WordTokenizer:
    """
    WordTokenizer class is responsible for tokenizing input text into individual words using a regular expression.
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
     It pads the input and output sequences to the maximum length in the batch.
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

    input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in tokens]
                                for tokens in padded_input_seqs]).transpose(0, 1)  # Transpose for transformer input
    output_tensor = torch.tensor([[output_vocab.get(token, 0) for token in tokens]
                                 for tokens in padded_output_seqs]).transpose(0, 1)  # Transpose for transformer input

    return input_tensor, output_tensor


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model class represents the main model architecture.
    This is the main model class that implements the GPT architecture. 
    It inherits from the nn.Module class. The GPT model consists of an embedding layer, transformer encoder layers, and a linear layer for prediction. 
    The embedding layer converts input tokens into continuous vector representations. The transformer encoder layers perform multi-head self-attention and feed-forward operations to capture the contextual relationships between tokens. 
    The linear layer maps the final hidden state of the transformer encoder to the output size.
    
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
            self.embedding = nn.Embedding(
                input_vocab_size, hidden_size, padding_idx=0)
        else:
            self.embedding = None

        self.encoder_layer = TransformerEncoderLayer(
            hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.encoder = TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
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
    """
    This class uses the Faker library to generate fake data for demonstration purposes. Currently, it provides methods for generating various types of data such as company names, years, documents, groups, topics, and dates.
    """
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

    def generate_date(self):
        return self.fake.date(pattern="%Y-%m-%d")


fake_generator = FakeDataGenerator()


"""
Functional Code Below
"""

"""

Defining the hyperparameters for the model training...
Very generally, below are default hyperparameters for model training: 

--------------------
Small Model: (runs on macbook, google colab)
    
    Batch Size: 8-32
    Learning Rate: 0.001-0.01
    Model Size/Architecture: Fewer layers, smaller hidden units, smaller embedding dimensions (e.g., 128-256 dimensions)
    Sequence Length: 32-128 tokens
    Gradient Accumulation: 1-2 steps
    Regularization: Dropout rate of 0.1-0.3, weight decay of 0.0001-0.001
    Optimizer: Adam or RMSprop
    
    small_batch_size = 8
    small_learning_rate = 0.001
    small_num_layers = 2
    small_hidden_size = 128
    small_num_heads = 4
    small_dropout = 0.1
--------------------

--------------------
Medium Model: (macbook, google colab)

    Batch Size: 32-64
    Learning Rate: 0.001-0.01
    Model Size/Architecture: Moderate number of layers, hidden units, and embedding dimensions (e.g., 256-512 dimensions)
    Sequence Length: 64-256 tokens
    Gradient Accumulation: 2-4 steps
    Regularization: Dropout rate of 0.1-0.5, weight decay of 0.0001-0.01
    Optimizer: Adam or RMSprop
    
    medium_batch_size = 32
    medium_learning_rate = 0.001
    medium_num_layers = 4
    medium_hidden_size = 256
    medium_num_heads = 8
    medium_dropout = 0.3
--------------------

--------------------
Large Model: (google colab)
    Batch Size: 64-128
    Learning Rate: 0.001-0.01
    Model Size/Architecture: More layers, larger hidden units, and embedding dimensions (e.g., 512-1024 dimensions)
    Sequence Length: 128-512 tokens
    Gradient Accumulation: 4-8 steps
    Regularization: Dropout rate of 0.1-0.5, weight decay of 0.0001-0.01
    Optimizer: Adam or RMSprop

    large_batch_size = 64
    large_learning_rate = 0.001
    large_num_layers = 6
    large_hidden_size = 512
    large_num_heads = 16
    large_dropout = 0.5
--------------------

--------------------
Extra Large Model: (google colab)

    Batch Size: 128-256
    Learning Rate: 0.001-0.01
    Model Size/Architecture: Very deep architectures, large hidden units, and embedding dimensions (e.g., 1024-2048 dimensions)
    Sequence Length: 256-1024 tokens
    Gradient Accumulation: 8 or more steps
    Regularization: Dropout rate of 0.1-0.5, weight decay of 0.0001-0.01
    Optimizer: Adam or RMSprop

    extra_large_batch_size = 128
    extra_large_learning_rate = 0.001
    extra_large_num_layers = 8
    extra_large_hidden_size = 1024
    extra_large_num_heads = 32
    extra_large_dropout = 0.5
--------------------

"""

# Define the hyperparameters, you can copy and paste them using the recommended ones from above
batch_size = 12
learning_rate = 1e-4
num_layers = 4
hidden_size = 128
num_heads = 32
dropout = 0.0
num_epochs = 200
num_inferences = 5

# If you want more variety in the output data, then add more to sample_dctionary and _templates.
# Feel free to add slang, contractions and other variations to the templates.

"""
The sample_dictionary contains input-output pairs for model training.
"""
# For sample_dictionary, please only use "Table" and "Document" for your values in the K:V pairs
sample_dictionary = {
    "Whut is the revenu for Company A on 2023-01-02": "Table",
    "Whut is the main point of this articl": "Document",
    "Whut is the reportin date for all customers under the e-commerce group": "Table",
    "Can you summarize this articl for me": "Document",
    "Whut is the revenu for Company B on 2023-01-02": "Table",
    "Whut is the revenu for Company C on 2023-01-02": "Table",
    "Whut is the revenu for Company D on 2023-01-02": "Table",
    "Whut is the revenu for Company E on 2023-01-02": "Table",
    "What is the revenue for Company A in 2023?": "Table",
    "What is the main point of this article?": "Document",
    "What is the reporting date for all customers under the e-commerce group?": "Table",
    "Can you summarize this article for me?": "Document",
    "What is the revenue for Company B in 2023?": "Table",
    "What is the revenue for Company C in 2023?": "Table",
    "What is the revenue for Company D in 2023?": "Table",
    "What is the revenue for Company E in 2023?": "Table",
}

"""
The _templates list contains the questions that the model will be trained to answer (generating prompts)
"""
# For _templates, please only use the following variables: {company}, {year}, {document}, {group}, {topic}, {date}
_templates = [
    "What is the revenue for {company} in {year}?",
    "What is the main point of {document}?",
    "What is the reporting date for all customers under the {group} group?",
    "Can you summarize {document} for me?",
    "What is the revenue growth rate for {company}?",
    "How many articles are there about {topic}?",
    "What's the revenue for {company} in {year}?",
    "What's the main point of {document}?",
    "What's the reporting date for all customers under the {group} group?",
    "Could you summarize {document} for me?",
    "What is the revenue for {company} on {date}?",
    "What is the main topic of {document}?",
    "What is the reporting deadline for all customers under the {group} group?",
    "Could you summarize the key points of {document}?",
    "What is the revenue for {company} on {date}?",
    "What is the average revenue for {company} in {year}?",
    "What is the key message of {document}?",
    "What is the latest reporting date for all customers under the {group} group?",
    "What is the revenue growth percentage for {company}?",
    "How many articles are available on {topic}?",
    "What's the revenue of {company} in {year}?",
    "What's the primary point of {document}?",
    "What's the reporting deadline for all customers under the {group} group?",
    "Can you provide a summary of {document}?",
    "What is the revenue for {company} on {date}?",
    "What is the main subject discussed in {document}?",
    "What is the reporting due date for all customers under the {group} group?",
    "Could you summarize the main points of {document}?",
    "What is the revenue for {company} on {date}?",
    "What is the revenue growth rate for {company} in {year}?",
    "What is the main idea of {document}?",
    "What is the latest reporting deadline for all customers under the {group} group?",
    "What is the revenue increase for {company}?",
    "How many articles are related to {topic}?",
    "What's the revenue earned by {company} in {year}?",
    "What's the key takeaway from {document}?",
    "What's the last reporting date for all customers under the {group} group?",
    "Can you summarize the content of {document}?",
    "Whats the revenue for {company} in {year}?",
    "Whats the main point of {document}?",
    "Whats the reporting date for all customers in the {group} group",
    "Tell me what {document} is about",
    "Whats the revenue growth rate for {company}",
    "How many articles about {topic}",
    "Whats the revenue for {company} in {year}",
    "Whats the main point of {document}",
    "Whats the reporting date for all customers in the {group} group",
    "Tell me what {document} is about",
    "Whats the revenue for {company} on {date}",
    "Whats the main topic of {document}",
    "Whats the reporting deadline for all customers in the {group} group",
    "Tell me the key points of {document}",
    "Whats the revenue for {company} on {date}",
    "Whats the average revenue for {company} in {year}",
    "Whats the key message of {document}",
    "Whats the latest reporting date for all customers in the {group} group",
    "Whats the revenue growth percentage for {company}",
    "How many articles related to {topic}",
    "Whats the revenue of {company} in {year}",
    "Whats the primary point of {document}",
    "Whats the reporting deadline for all customers in the {group} group",
    "Tell me a summary of {document}",
    "Whats the revenue for {company} on {date}",
    "Whats the main subject discussed in {document}",
    "Whats the reporting due date for all customers in the {group} group",
    "Tell me the main points of {document}",
    "Whats the revenue for {company} on {date}",
    "Whats the revenue growth rate for {company} in {year}",
    "Whats the main idea of {document}",
    "Whats the latest reporting deadline for all customers in the {group} group",
    "Whats the revenue increase for {company}",
    "How many articles related to {topic}",
    "Whats the revenue earned by {company} in {year}",
    "Whats the key takeaway from {document}",
    "Whats the last reporting date for all customers in the {group} group",
    "Tell me the content of {document}",
    "Whuts the revenu for {company} in {year}?",
    "Whuts the main point of {document}?",
    "Whuts the reportin date for all customers in the {group} group",
    "Tell me what {document} is about",
    "Whuts the revenu growth rate for {company}",
    "How many articles about {topic}",
    "Whuts the revenu for {company} in {year}",
    "Whuts the main point of {document}",
    "Whuts the reportin date for all customers in the {group} group",
    "Tell me what {document} is about",
    "Whuts the revenu for {company} on {date}",
    "Whuts the main topic of {document}",
    "Whuts the reportin deadline for all customers in the {group} group",
    "Tell me the key points of {document}",
    "Whuts the revenu for {company} on {date}",
    "Whuts the average revenu for {company} in {year}",
    "Whuts the key message of {document}",
    "Whuts the latest reportin date for all customers in the {group} group",
    "Whuts the revenu growth percentage for {company}",
    "How many articles related to {topic}",
    "Whuts the revenu of {company} in {year}",
    "Whuts the primary point of {document}",
    "Whuts the reportin deadline for all customers in the {group} group",
    "Tell me a summary of {document}",
    "Whuts the revenu for {company} on {date}",
    "Whuts the main subject discussed in {document}",
    "Whuts the reportin due date for all customers in the {group} group",
    "Tell me the main points of {document}",
    "Whuts the revenu for {company} on {date}",
    "Whuts the revenu growth rate for {company} in {year}",
    "Whuts the main idea of {document}",
    "Whuts the latest reportin deadline for all customers in the {group} group",
    "Whuts the revenu increase for {company}",
    "How many articles related to {topic}",
    "Whuts the revenu earned by {company} in {year}",
    "Whuts the key takeaway from {document}",
    "Whuts the last reportin date for all customers in the {group} group",
    "Tell me the content of {document}",
]


torch.manual_seed(12345)

# Create a word tokenizer
tokenizer = WordTokenizer()

# Prepare the dataset and vocabulary
key_value_pairs = [(key, value) for key, value in sample_dictionary.items()]
tokenized_dataset = KeyValueDataset(key_value_pairs, tokenizer)

# Populate the input and output vocabularies
# Assign unique indices to each token
input_vocab = defaultdict(lambda: len(input_vocab))
# Assign unique indices to each token
output_vocab = defaultdict(lambda: len(output_vocab))

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


# Create an instance of the GPT model
model = GPT(len(input_vocab), len(output_vocab),
            num_layers, hidden_size, num_heads, dropout)


total_params = sum(p.numel() for p in model.parameters())
print("-----------------------------------------------")
print(f"Total number of parameters: {total_params}")
print("-----------------------------------------------")

# Create a DataLoader for training
dataloader = DataLoader(
    tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

generated_data = []

total_average_loss = 0

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for input_tokens, output_tokens in dataloader:
        input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in tokens]
                                    for tokens in input_tokens]).transpose(0, 1)  # Transpose for transformer input
        output_tensor = torch.tensor([[output_vocab.get(token, 0) for token in tokens]
                                     for tokens in output_tokens]).transpose(0, 1)  # Transpose for transformer input

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
    total_average_loss += total_loss  # accumulate loss across eopchs
    print(f"Epoch {epoch+1}/{num_epochs} :-----: Average Loss: {average_loss}")

    # Evaluation process starts here
    random_template = random.choice(_templates)
    company = fake_generator.generate_company()
    year = fake_generator.generate_year()
    date = fake_generator.generate_date()
    document = fake_generator.generate_document()
    group = fake_generator.generate_group()
    topic = fake_generator.generate_topic()

    prompt = random_template.format(
        company=company,
        year=year,
        document=document,
        group=group,
        topic=topic,
        date=date
    )
    input_tokens = tokenizer.tokenize(prompt)
    input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in input_tokens]]).transpose(
        0, 1)  # Transpose for transformer input
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
        "question": prompt,
        "label": predicted_value
    })

average_loss = total_average_loss / num_epochs
print("-----------------------------------------------")
print(f"Average loss across all epochs: {average_loss}")
print("-----------------------------------------------")
print("\n")
# Save the trained model
torch.save(model.state_dict(), "src/gpt_from_scratch/eda/trained_gpt_model.pt")

# Save generated data to YAML file
filename = "src/gpt_from_scratch/eda/generated_data.yaml"
with open(filename, "w") as file:
    yaml.dump(generated_data, file)

# Save generated data to a JSON file
output_file = "src/gpt_from_scratch/eda/generated_data.json"
with open(output_file, "w") as f:
    json.dump(generated_data, f, indent=4)


"""
Inference for later on!
"""
torch.manual_seed(1337)  # setting a different manual seed for inference
last_prompt = ""  # Variable to store the last used prompt

for _ in range(num_inferences):  # generate inferences using model with no_grad()
    while True:
        # Generate a random prompt
        prompt_template = random.choice(_templates)
        prompt = prompt_template.format(
            company=fake_generator.generate_company(),
            year=fake_generator.generate_year(),
            document=fake_generator.generate_document(),
            group=fake_generator.generate_group(),
            topic=fake_generator.generate_topic(),
            date=fake_generator.generate_date(),
        )
        # Check if the prompt is different from the last used prompt
        if prompt != last_prompt:
            last_prompt = prompt  # Update the last used prompt
            break  # Break out of the loop
    # Tokenize the prompt
    input_tokens = tokenizer.tokenize(prompt)
    input_tensor = torch.tensor([[input_vocab.get(token, 0) for token in input_tokens]]).transpose(
        0, 1)  # Transpose for transformer input

    # Pass the input tensor through the model
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor.transpose(0, 1))

    # Get the predicted indices
    predicted_indices = output_tensor.argmax(dim=-1).tolist()[0]

    # Convert the predicted indices back to their original representations
    predicted_value = list(output_vocab.keys())[predicted_indices]

    print("-----------------------------------------------")
    print("Inference")
    print("-----------------------------------------------")
    # Print the generated prompt and the corresponding generated value
    print(f"Generated prompt: '{prompt}'")
    print(f"Generated value: '{predicted_value}'")
    print("\n")
