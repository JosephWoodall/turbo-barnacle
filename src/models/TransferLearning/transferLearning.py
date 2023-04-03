import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """ """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# Instantiate the dataset with your data
data = ... # your data
target = ... # your target
dataset = MyDataset(data, target)

# Define data preprocessing steps
data_transforms = transforms.Compose([transforms.ToTensor()])

# Apply the transform to the data
dataset = dataset.transform(data_transforms)

# Create a dataloader object
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Instantiate a pre-trained model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

# Put the model in evaluation mode
model.eval()

# Run inference on the data
for data, target in dataloader:
    output = model(data)
    print(output)