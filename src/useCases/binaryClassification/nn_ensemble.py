import torch
import torch.nn as nn
import torch.optim as optim

import yaml
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import random


"""
Class Definition for Neural Network Ensemble
"""


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class Ensemble(nn.Module):
    def __init__(self, num_models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList([NeuralNetwork()
                                    for _ in range(num_models)])

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        return torch.mean(torch.stack(predictions), dim=0)


"""
Preprocessing Data
"""
# Load the YAML file
with open('data.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# Extract the text and labels
texts = [entry['question'] for entry in data]
labels = [entry['label'] for entry in data]

# Tokenize the text
tokenizer = get_tokenizer('basic_english')
tokenized_texts = [tokenizer(text) for text in texts]

# Build the vocabulary
vocab = build_vocab_from_iterator(tokenized_texts)

# Convert the text to numerical values (tensors)
text_tensors = [torch.tensor([vocab[token] for token in tokens])
                for tokens in tokenized_texts]

# Convert the labels to numerical values (tensors)
label_tensors = torch.tensor(
    [1 if label == 'Document' else 0 for label in labels])

"""
Defining Hyperparameters
"""
num_models = 5
learning_rate = 0.001
num_epochs = 10

torch.manual_seed(1337)
# Shuffle and split the data into training and testing sets
data = list(zip(text_tensors, label_tensors))
random.shuffle(data)
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

train_texts, train_labels = zip(*train_data)
test_texts, test_labels = zip(*test_data)

# Create the ensemble model
ensemble_model = Ensemble(num_models)
total_params = sum(p.numel() for p in ensemble_model.parameters())
print("-----------------------------------------------")
print(f"Total number of parameters: {total_params}")
print("-----------------------------------------------")

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(ensemble_model.parameters(), lr=learning_rate)


# Custom F1 scoring function
def compute_f1_score(predictions, labels):
    predictions = torch.round(predictions).flatten()
    labels = labels.flatten()

    true_positives = torch.sum(predictions * labels).item()
    false_positives = torch.sum(predictions * (1 - labels)).item()
    false_negatives = torch.sum((1 - predictions) * labels).item()

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


# Training loop
for epoch in range(num_epochs):
    ensemble_model.train()
    optimizer.zero_grad()
    outputs = ensemble_model(train_texts)
    loss = criterion(outputs, torch.stack(train_labels))
    loss.backward()
    optimizer.step()

    # Evaluation
    ensemble_model.eval()
    with torch.no_grad():
        outputs = ensemble_model(test_texts)
        f1 = compute_f1_score(outputs, torch.stack(test_labels))

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, F1 Score: {f1:.4f}")

# Save the entire model
torch.save(ensemble_model.state_dict(), 'ensemble_model.pt')
