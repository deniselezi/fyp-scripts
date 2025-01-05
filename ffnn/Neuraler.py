import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


seed = 1
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FFNN(nn.Module):
    def __init__(self, input_dim):
        super(FFNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.layers(x)


class Neuraler:
    def __init__(self, season, batch_size=32):
        self.season = season
        self.season_start, self.season_end = self._get_season_dates()
        X_train, y_train, X_test, y_test = self._read_data()
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        X_test_scaled = self.x_scaler.transform(X_test)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        y_test_scaled = self.y_scaler.transform(y_test)

        self.X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        self.y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        self.X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        self.y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

        self.train_loader = DataLoader(
            TensorDataset(self.X_train_tensor, self.y_train_tensor),
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            TensorDataset(self.X_test_tensor, self.y_test_tensor),
            batch_size=batch_size,
            shuffle=False
        )

        input_dim = X_train.shape[1]
        self.model = FFNN(input_dim).to(device)
    
    def fit(self, num_epochs=200, plot_training=True, plot_results=True):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_losses.append(epoch_train_loss / len(self.train_loader))

            self.model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in self.test_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_test_loss += loss.item()
            test_losses.append(epoch_test_loss / len(self.test_loader))

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

        if plot_training:
            self._plot_training(num_epochs, train_losses, test_losses)
        
        self.model.eval()
        with torch.no_grad():
            y_pred_tensor =  self.model( self.X_test_tensor)

        y_pred_scaled = y_pred_tensor.cpu().numpy()
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        y_actual = self.y_scaler.inverse_transform( self.y_test_tensor.cpu().numpy())

        mae = mean_absolute_error(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        correlation = np.corrcoef(y_actual.flatten(), y_pred.flatten())[0, 1]

        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Correlation: {correlation:.2f}")

        if plot_results:
            self._plot_results(y_actual, y_pred)
    

    def _plot_training(self, num_epochs, train_losses, test_losses):
        plt.plot(list(range(num_epochs)), train_losses, color="black", label="Train")
        plt.plot(list(range(num_epochs)), test_losses, color="royalblue", label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("Train vs test loss during training")
        plt.legend()
        plt.show()
    
    def _plot_results(self, y_actual, y_pred):
        days = [i - 3286 for i in list(range(self.season_start, self.season_end + 1))]
        plt.plot(days, y_actual, color="black", label="Actual")
        plt.plot(days, y_pred, color="royalblue", label="Predicted")
        plt.xlabel("Days")
        plt.ylabel("ILI Rates")
        plt.title("Actual vs Predicted ILI Rates (16-17)")
        plt.legend()
        plt.show()

    def _get_season_dates(self):
        """
        Season:
            1 -> 2015 - 2016
            2 -> 2016 - 2017
            3 -> 2017 - 2018
            2 -> 2018 - 2019
        """
        if self.season == 1:
            start = 4262 - 1340 - 1
            end = 4627 - 1340 - 1
        elif self.season == 2:
            start = 4628 - 1340 - 1
            end = 4992 - 1340 - 1
        elif self.season == 3:
            start = 4993 - 1340 - 1
            end = 5357 - 1340 - 1
        elif self.season == 4:
            start = 5358 - 1340 - 1
            end = 5722 - 1340 - 1
        else:
            raise ValueError("Season must be 1, 2, 3 or 4")
        
        return start, end

    def _read_data(self):
        print('Reading data...')
        X = pd.read_csv("../processed_data/final_1000.csv")
        y = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

        X_train, y_train = X[:self.season_start], y[:self.season_start]
        X_test, y_test = X[self.season_start:self.season_end+1], y[self.season_start:self.season_end+1]

        return X_train, y_train, X_test, y_test
