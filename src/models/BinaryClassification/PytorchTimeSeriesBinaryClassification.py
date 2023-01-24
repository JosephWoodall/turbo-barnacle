import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader

INPUT_SIZE = 7
HIDDEN_SIZE = 32
NUM_LAYERS = 1

BATCH_SIZE = 32
EPOCH_RANGE = 100


df = '' # Insert your data here

data = torch.tensor(df.values)
x = data[:, :-1] # Assuming the repsonse variable is last columnal feature
y = data[:, -1]
ds = TensorDataset(x, y)

class TimeSeriesModel(nn.Module):
    def __inti__(self): 
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS, batch_first = True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid =nn.Sigmoid()

    def forward(self, x): 
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = TimeSeriesModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

dl = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = True)

# Training the model
for epoch in range(EPOCH_RANGE):
    for x_batch, y_batch in dl: 
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred.squeeze(-1), y_batch)
        loss.backward()
        optimizer.step()

# Testing the model
with torch.no_grad():
    test_x = torch.tensor(test.values[:, :-1])
    test_y = torch.tensor(test.values[:, -1])
    test_y_pred = model(test_x)
    test_y_pred = (test_y_pred > 0.5).float()
    test_acc = (test_y_pred == test_y).float().mean()
    print("Test accuracy:", test_acc)