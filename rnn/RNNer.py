import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from EarlyStopper import EarlyStopper

from RNN import GRU


class RNNer:
    def __init__(self, season, window_size=7, hidden_size=50, seed = 1, dropout=0):
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.season = season
        self.season_start, self.season_end = self._get_season_dates()
        X_train, y_train, X_test, y_test = self._read_data(window_size)
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        n_train_samples, window_size, n_features = X_train.shape
        n_test_samples = X_test.shape[0]

        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features) 

        X_train_scaled = self.x_scaler.fit_transform(X_train_reshaped).reshape(n_train_samples, window_size, n_features)
        X_test_scaled = self.x_scaler.transform(X_test_reshaped).reshape(n_test_samples, window_size, n_features)
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()  # Scale targets
        y_test_scaled = self.y_scaler.transform(y_test.reshape(-1, 1)).flatten()       # Scale targets

        days = 365

        # last year as validation set
        X_val = X_train_scaled[self.season_start - days : self.season_start]
        y_val = y_train_scaled[self.season_start - days : self.season_start]
        X_train_scaled = X_train_scaled[: self.season_start - days]
        y_train_scaled = y_train_scaled[: self.season_start - days]

        self.X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(
            device
        )
        self.y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(
            device
        )
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        self.X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        self.y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)

        input_dim = n_features
        self.model = GRU(input_size=input_dim, hidden_size=hidden_size, dropout=dropout).to(device)

    def fit(
        self,
        num_epochs=200,
        plot_training=True,
        plot_results=True,
        stop_early=True,
        patience=10,
        warmup=20,
        lr=0.001,
        batch_size=32
    ):
        self.train_loader = DataLoader(
            TensorDataset(self.X_train_tensor, self.y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        # print(self.X_train_tensor.shape)
        # sys.exit()
        
        train_losses, val_losses, stop_epoch = self._train(
            num_epochs, stop_early, warmup, lr, patience
        )

        print("Final loss:", val_losses[-(patience+1)])

        if plot_training:
            self._plot_training(len(train_losses), train_losses, val_losses)

        self.model.eval()
        with torch.no_grad():
            y_pred_tensor = self.model(self.X_test_tensor)

        y_pred_scaled = y_pred_tensor.cpu().numpy()
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        y_actual = self.y_scaler.inverse_transform(self.y_test_tensor.cpu().numpy())

        mae = mean_absolute_error(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        correlation = np.corrcoef(y_actual.flatten(), y_pred.flatten())[0, 1]

        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Correlation: {correlation:.2f}")

        if plot_results:
            self._plot_results(y_actual, y_pred)

    def _train(self, num_epochs, stop_early, warmup, lr, patience):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        early_stopper = EarlyStopper(patience=patience)

        train_losses = []
        val_losses = []
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
            with torch.no_grad():
                val_pred = self.model(self.X_val_tensor)
                val_loss = criterion(val_pred, self.y_val_tensor)
                val_losses.append(val_loss.item())

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}"
                )

            if stop_early and (epoch + 1) > warmup:
                early_stopper(val_loss, self.model)
                if early_stopper.early_stop:
                    stop_epoch = epoch + 1 - early_stopper.patience
                    print(f"Early stopping at epoch {stop_epoch}")
                    break
            else:
                stop_epoch = (epoch + 1) - patience

        return train_losses, val_losses, stop_epoch

    def _plot_training(self, num_epochs, train_losses, test_losses):
        plt.plot(list(range(num_epochs)), train_losses, color="black", label="Train")
        plt.plot(list(range(num_epochs)), test_losses, color="royalblue", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("Train vs val loss during training")
        plt.legend()
        plt.show()

    def _plot_results(self, y_actual, y_pred):
        days = [
            i - 3286 + 365 for i in list(range(self.season_start, self.season_end + 1))
        ]
        plt.plot(days, y_actual, color="black", label="Actual")
        plt.plot(days, y_pred, color="royalblue", label="Predicted")
        plt.xlabel("Days")
        plt.ylabel("ILI Rates")
        plt.title("Actual vs Predicted ILI Rates")
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

    def _read_data(self, window_size):
        print("Reading data...")
        X = pd.read_csv("../processed_data/final_1000.csv")
        y = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

        # print(X[: self.season_start].shape)
        # print(X[self.season_start : self.season_end + 1].shape)

        print(X.shape, y.shape)

        X, y = self._create_sliding_windows(X, y, window_size=window_size)

        print(X.shape, y.shape)

        # print(X.shape)

        adjusted_start = self.season_start - (window_size - 1)
        adjusted_end = self.season_end - (window_size - 1)

        X_train, y_train = X[:adjusted_start], y[:adjusted_start]
        X_test, y_test = X[adjusted_start:adjusted_end + 1], y[adjusted_start:adjusted_end + 1]

        # print(X_train.shape)
        # print(X_test.shape)

        return X_train, y_train, X_test, y_test

    def _create_sliding_windows(self, X, y, window_size):

        X_windows = []
        y_windows = []

        X, y = X.values, y.values

        n_samples = len(X)

        if window_size > n_samples:
            raise ValueError("Window size must be smaller than or equal to the number of samples in X.")

        X_windows = np.array([X[i:i + window_size] for i in range(n_samples - window_size + 1)])
        y_windows = y[window_size - 1:]  # targets are last day in each window

        return X_windows, y_windows

