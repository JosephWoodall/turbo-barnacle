import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
In the above script, ImageSimilarity class is defined with two datasets as input. 
The class builds a ResNet-50 model pre-trained on ImageNet and removes the last fully connected 
layer, which is used to extract feature vectors from the images.

The extract_features method takes in a dataset and applies the pre-defined transform to 
the images and then pass them through the CNN to extract feature vectors.

The find_similarity method calls the extract_features method on both datasets 
and computes the dot product of feature vectors of the two datasets to get the similarity matrix.

It's important to note that the script uses Resnet50 model, this is just an example 
you can use any other model that you prefer or that you think will work better for your use case. 
Also, this script does not consider the case where images are different in size, you may have to 
pre-process the images before passing them to the model.
"""

class ImageSimilarity:
    """ """
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.model = self.build_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def build_model(self):
        """ """
        model = torchvision.models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def extract_features(self, dataset):
        """

        :param dataset: 

        """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        features = []
        for images, labels in dataloader:
            images = images.to(self.device)
            features.append(self.model(images))
        return torch.cat(features)

    def find_similarity(self):
        """ """
        dataset1_features = self.extract_features(self.dataset1)
        dataset2_features = self.extract_features(self.dataset2)
        dataset1_features = dataset1_features.view(dataset1_features.size(0), -1)
        dataset2_features = dataset2_features.view(dataset2_features.size(0), -1)
        similarity_matrix = torch.mm(dataset1_features, dataset2_features.t())
        return similarity_matrix