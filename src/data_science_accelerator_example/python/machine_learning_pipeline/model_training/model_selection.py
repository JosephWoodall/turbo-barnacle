import polars as pl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class ModelSelection:
    """
    The ModelSelection class is used to automatically select the best machine learning or deep learning model for a given response variable. 
    The class should take in a data set and the name(s) of the response variable(s) to use, split the data set into training and testing sets, and then train
    and evaluate a range of different models using hyperparameter tuning. The models to be tested should include the following: 
        - Linear Regression
        - Support Vector Regression
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - Multi-layer perceptron
        - Convolutional Neural Network
        - Recurrent Neural Network (LSTM)
        - Generatie Adversarial Network
    This class should return the best performing model along with its associated score. Additionally, this class should be able to handle
    both categorical and continuous mixed data.
    """

    def __init__(self, data: pl.DataFrame) -> None:
        """
        __init__ Initializes the ModelSelection class.

        Args:
            data (pl.DataFrame): the speficied data set.
        """
        self.data = data

    def select_model(self, response_variable: str) -> tuple:
        """
        select_model selects the best model for the given response variable(s)

        Args:
            response_variable (str): the name(s) of the response variable(s) to use.

        Returns:
            tuple: the best model and its associated score.
        """
        # Split the data into training and testing sets
        X = self.data.drop(response_variable, axis=1)
        y = self.data[response_variable]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Define the hyperparameterized models to test
        models = [
            ('Random Forest', RandomForestRegressor(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20]
            }),
            ('Gradient Boosting', GradientBoostingRegressor(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 1]
            }),
            ('Linear Regression', LinearRegression(), {}),
            ('Support Vector Regression', SVR(), {
                'C': [0.1, 1, 10],
                'gamma': [0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }),
            ('XGBoost', xgb.XGBRegressor(), {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 1],
                'n_estimators': [50, 100, 200]
            }),
            ('Multi-layer Perceptron', MLPRegressor(), {
                'hidden_layer_sizes': [(50,), (100,), (200,)],
                'activation': ['relu', 'tanh', 'logistic'],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }),
            ('Convolutional Neural Network', ConvNet(), {
                'epochs': [10, 20, 30],
                'batch_size': [16, 32, 64]
            }),
            ('Recurrent Neural Network', RNN(), {
                'epochs': [10, 20, 30],
                'batch_size': [16, 32, 64]
            }),
            ('Generative Adversarial Network', GAN(), {
                'epochs': [50, 100, 200],
                'batch_size': [16, 32, 64]
            })
        ]

        # Train and evaluate each model on the testing set
        best_model = None
        best_score = float('inf')
        for name, model, params in models:
            # Hyperparameter tuning
            if isinstance(model, ConvNet) or isinstance(model, RNN):
                # Convert data to PyTorch tensors
                train_data = PyTorchDataset(X_train, y_train)
                train_loader = DataLoader(
                    train_data, batch_size=32, shuffle=True)
                # Train model with grid search for hyperparameters
                search = GridSearchCV(model, params, cv=5,
                                      scoring='neg_mean_squared_error')
                search.fit(train_loader)

            else:
                # Train model with grid search for hyperparameters
                search = GridSearchCV(model, params, cv=5,
                                      scoring='neg_mean_squared_error')
                search.fit(X_train, y_train)

            # Evaluate model on testing set
            y_pred = search.predict(X_test)
            score = mean_squared_error(y_test, y_pred)

            # Update best model if current model performs better
            if score < best_score:
                best_model = search.best_estimator_
                best_score = score

        return (best_model, best_score)


class PyTorchDataset(Dataset):
    """
    PyTorchDataset a dataset object used for pytorch associated deep learning models.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X.values).float()
        self.y = torch.tensor(y.values).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ConvNet(nn.Module):
    """
    ConvNet Convolutional Neural Network

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(832, 128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        return x.squeeze(1)


class RNN(nn.Module):
    """
    RNN Recurrent Neural Network (LSTM Variation)

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64,
                             num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.dense1(x)
        return x.squeeze(1)


class GAN(nn.Module):
    """
    GAN Generative Adversarial Network

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(GAN, self).__init__()

        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Loss function
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        # Generate fake images from random noise
        z = torch.randn(x.size(0), 100)
        fake_images = self.generator(z)

        # Discriminate between real and fake images
        real_outputs = self.discriminator(x)
        fake_outputs = self.discriminator(fake_images)

        return real_outputs, fake_outputs

    def train_step(self, x, optimizer):
        # Generate fake images from random noise
        z = torch.randn(x.size(0), 100)
        fake_images = self.generator(z)

        # Discriminate between real and fake images
        real_outputs = self.discriminator(x)
        fake_outputs = self.discriminator(fake_images)

        # Compute the discriminator loss
        d_loss_real = self.loss_fn(real_outputs, torch.ones_like(real_outputs))
        d_loss_fake = self.loss_fn(
            fake_outputs, torch.zeros_like(fake_outputs))
        d_loss = d_loss_real + d_loss_fake

        # Update the discriminator parameters
        optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer.step()

        # Generate new fake images from random noise
        z = torch.randn(x.size(0), 100)
        fake_images = self.generator(z)

        # Compute the generator loss
        g_loss = self.loss_fn(self.discriminator(
            fake_images), torch.ones_like(fake_outputs))

        # Update the generator parameters
        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()

        return d_loss.item() + g_loss.item()


'''
# or maybe just use automl in the select_model function?
 def select_model(self, response_variable: str) -> tuple:
    aml = H2OAutoML(max_models = 20, seed = 1)
    aml.train(x = x, y = y, training_frame = train)
    return aml.leader, aml.leaderboard[0]

'''
