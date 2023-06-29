# Assuming you have a dataset of conversations stored in a file
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        conversations = file.readlines()
    return conversations

def preprocess_data(conversations):
    # Implement your preprocessing logic here, including cleaning and tokenization
    # Split conversations into input/output pairs
    input_texts = []
    target_texts = []
    for conv in conversations:
        conv = conv.strip()
        for i in range(len(conv)-1):
            input_texts.append(conv[:i+1])
            target_texts.append(conv[i+1])
    return input_texts, target_texts

# Load and preprocess the dataset
conversations = load_dataset(r'src/gpt_from_scratch/raw_data/input.txt')
input_texts, target_texts = preprocess_data(conversations)

import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=2,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs = self.transformer(embeddings, embeddings)
        predictions = self.fc(outputs)
        return predictions

# Define hyperparameters
vocab_size = 10000  # Set based on your vocabulary size
embedding_dim = 256
hidden_dim = 512
num_layers = 2

# Create an instance of the GPT model
model = GPT(vocab_size, embedding_dim, hidden_dim, num_layers)

import torch.optim as optim
import torch.nn.functional as F

# Define hyperparameters for training
batch_size = 32
epochs = 10
learning_rate = 0.001

# Convert input and target texts to tensors
input_tensor = torch.tensor(input_texts)
target_tensor = torch.tensor(target_texts)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(0, len(input_tensor), batch_size):
        inputs = input_tensor[i:i+batch_size]
        targets = target_tensor[i:i+batch_size]

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch: {epoch+1} | Loss: {running_loss}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Load the trained model
model = GPT(vocab_size, embedding_dim, hidden_dim, num_layers)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# User input processing and response generation
def generate_response(input_text):
    input_tensor = torch.tensor([input_text])
    output_tensor = model(input_tensor)
    predicted_token_idx = torch.argmax(output_tensor, dim=2)
    predicted_token = predicted_token_idx.item()
    response = chr(predicted_token)  # Assuming your tokens are represented as characters
    return response

# Example usage
user_input = "Hello!"
response = generate_response(user_input)
print(response)
