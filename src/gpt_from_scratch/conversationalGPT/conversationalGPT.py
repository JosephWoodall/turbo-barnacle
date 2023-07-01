import torch
import torch.nn as nn

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
            target_texts.append(conv[i+1])  # Modify to create a target tensor with shape (batch_size,)
    return input_texts, target_texts

# Load and preprocess the dataset
conversations = load_dataset(r'src/gpt_from_scratch/raw_data/input.txt')
input_texts, target_texts = preprocess_data(conversations)

print(conversations[:5])
print(input_texts[:5])
print(target_texts[:5])

class GPT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(hidden_size, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded)
        output = self.fc(output)
        return output

# Define hyperparameters
vocab_size = 10000  # Set based on your vocabulary size
embedding_dim = 256
hidden_dim = 512
num_layers = 2

# Create an instance of the GPT model
model = GPT(vocab_size, embedding_dim, hidden_dim)

import torch.optim as optim
import torch.nn.functional as F

# Define hyperparameters for training
batch_size = 32
epochs = 10
learning_rate = 0.001

# Convert input and target texts to numerical representations
input_numerical = [[ord(c) for c in text] for text in input_texts]
target_numerical = [ord(c) for c in target_texts]  # Remove sublist iteration for target texts

# Get the maximum sequence length
max_seq_length = max([len(seq) for seq in input_numerical])

# Pad the sequences to have the same length
input_padded = [seq + [0] * (max_seq_length - len(seq)) for seq in input_numerical]
target_padded = target_numerical

# Convert padded sequences to tensors
input_tensor = torch.tensor(input_padded)
target_tensor = torch.tensor(target_padded)

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
        # Reshape the outputs to match the expected shape
        outputs = outputs.permute(0, 2, 1)  # (batch_size, vocab_size, sequence_length)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch: {epoch+1} | Loss: {running_loss}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Load the trained model
model = GPT(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# User input processing and response generation
def generate_response(input_text):
    input_numerical = [ord(c) for c in input_text]
    input_padded = input_numerical + [0] * (max_seq_length - len(input_numerical))
    input_tensor = torch.tensor([input_padded])
    output_tensor = model(input_tensor)
    predicted_token_idx = torch.argmax(output_tensor, dim=2)
    predicted_token = chr(predicted_token_idx.item())  # Assuming your tokens are represented as characters
    return predicted_token

# Example usage
user_input = "Hello!"
response = generate_response(user_input)
print(response)
