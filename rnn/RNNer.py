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

import sys


class RNNer:
    def __init__(self, season, data_path, window_size=7, hidden_size=50, seed=1, dropout=None, hindcasting=False, advanced_validation=True):
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {device}")
        self.season = season
        self.hindcasting = hindcasting
        self.window_size = window_size
        self.season_start, self.season_end = self._get_season_dates()
        X_train, y_train, X_test, y_test = self._read_data(data_path, window_size, hindcasting)
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        n_train_samples, window_size, n_features = X_train.shape
        n_test_samples = X_test.shape[0]

        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features) 

        X_train_scaled = self.x_scaler.fit_transform(X_train_reshaped).reshape(n_train_samples, window_size, n_features)
        X_test_scaled = self.x_scaler.transform(X_test_reshaped).reshape(n_test_samples, window_size, n_features)
        y_train_scaled = self.y_scaler.fit_transform(y_train).reshape(n_train_samples, window_size)
        y_test_scaled = self.y_scaler.transform(y_test).reshape(n_test_samples, window_size)

        days = 365

        if advanced_validation:
            # print(X_train_scaled.shape)
            X_train_scaled, y_train_scaled, X_val, y_val = self._validation_set(X_train_scaled, y_train_scaled)
            # print(X_train_scaled.shape)
            # print(X_val.shape)
        else:  # last year as validation set
            X_val = X_train_scaled[self.season_start - days : self.season_start]
            y_val = y_train_scaled[self.season_start - days : self.season_start]
            X_train_scaled = X_train_scaled[: self.season_start - days]
            y_train_scaled = y_train_scaled[: self.season_start - days]

        self.X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(
            device
        )
        self.y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(
            device
        ) if hindcasting else torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(
            device
        )
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(
            device
        ) if hindcasting else torch.tensor(y_val.reshape(-1, window_size), dtype=torch.float32).to(
            device
        )
        self.X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        self.y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

        input_dim = n_features
        output_size = window_size if hindcasting else 1
        self.model = GRU(
            input_size=input_dim, hidden_size=hidden_size, dropout=dropout, output_size=output_size
        ).to(device)

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
        
        train_losses, val_losses, _ = self._train(
            num_epochs, stop_early, warmup, lr, patience
        )

        final_loss = val_losses[-(patience+1)]
        print("Final loss:", final_loss)

        if plot_training:
            self._plot_training(len(train_losses), train_losses, val_losses)

        self.model.eval()
        with torch.no_grad():
            y_pred_tensor = self.model(self.X_test_tensor)

        y_pred = self.y_scaler.inverse_transform(y_pred_tensor.cpu().numpy())
        y_actual = self.y_scaler.inverse_transform(self.y_test_tensor.cpu().numpy())
        if self.hindcasting:
            y_pred = y_pred[:, -1]
            y_actual = y_actual[:, -1]

        mae = mean_absolute_error(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        correlation = np.corrcoef(y_actual.flatten(), y_pred.flatten())[0, 1]

        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Correlation: {correlation:.2f}")

        if plot_results:
            self._plot_results(y_actual, y_pred)
        
        return mae, mse, correlation, final_loss

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
                # print(batch_X.shape)
                # sys.exit()
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

    def _read_data(self, path, window_size, hindcasting):
        X = pd.read_csv(path)
        y = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

        X, y = self._create_sliding_windows(X, y, window_size, hindcasting)

        adjusted_start = self.season_start - (window_size - 1)
        adjusted_end = self.season_end - (window_size - 1)

        X_train, y_train = X[:adjusted_start], y[:adjusted_start]
        X_test, y_test = X[adjusted_start:adjusted_end + 1], y[adjusted_start:adjusted_end + 1]

        return X_train, y_train, X_test, y_test

    def _create_sliding_windows(self, X, y, window_size, hindcasting):

        X_windows = []
        y_windows = []

        X, y = X.values, y.values

        n_samples = len(X)

        if window_size > n_samples:
            raise ValueError("Window size must be smaller than or equal to the number of samples in X.")

        X_windows = np.array([X[i:i + window_size] for i in range(n_samples - window_size + 1)])
        y_windows = np.array([y[i:i + window_size].flatten() for i in range(n_samples - window_size + 1)]) if hindcasting else y[window_size - 1:]

        return X_windows, y_windows
    

    def _validation_set(self, X_train, y_train, plot_validation=False):
        """
        Selects the validation set using an advanced strategy based on onset, peak, and outset periods.
        """
        # print("validating advancedly")
        # print(X_train.shape)
        # season_start = self.season_start
        season_start = self.season_start - (self.window_size-1)  # need to adjust this value for subsequent calculations
        # number of training samples is reduced after creating sliding windows
        validation_period_start = season_start - 3 * 365
        validation_period_end = season_start
        # print(f"Validation period: {validation_period_start} - {validation_period_end}")
        validation_period_y = y_train[validation_period_start:validation_period_end]
        validation_period_y = validation_period_y.transpose()[-1]
        # print(validation_period_y.shape)
        
        # Calculate threshold
        mean_y = np.mean(y_train[:validation_period_start])
        std_y = np.std(y_train[:validation_period_start])
        threshold = mean_y - 0.25 * std_y
        
        # Find onset period (3rd year in validation period)
        onset_start_idx = None
        for i in range(2 * 365, 3 * 365):
            if (validation_period_y[i:i+14] > threshold).all():
                onset_start_idx = i
                break
        
        if onset_start_idx is None:
            raise ValueError("Onset period not found.")
        
        onset_window_start = validation_period_start + onset_start_idx - 30
        onset_window_end = onset_window_start + 60
        
        # Find peak period (2nd year in validation period)
        peak_idx = np.argmax(validation_period_y[365:2*365]) + 365
        peak_window_start = validation_period_start + peak_idx - 30
        peak_window_end = peak_window_start + 60
        
        # Find outset period (1st year in validation period)
        outset_start_idx = None
        for i in range(365):
            # print(validation_period_y[i])
            if validation_period_y[i] > threshold:
                outset_start_idx = i
        
        if outset_start_idx is None:
            raise ValueError("Outset period not found.")
        
        outset_window_start = validation_period_start + outset_start_idx - 30
        outset_window_end = outset_window_start + 60
        
        # Create validation set
        val_indices = np.concatenate([
            np.arange(onset_window_start, onset_window_end),
            np.arange(peak_window_start, peak_window_end),
            np.arange(outset_window_start, outset_window_end)
        ])
        
        # print("ONSET:", onset_start_idx)
        # print("PEAK:", peak_idx)
        # print("OUTSET:", outset_start_idx)

        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # Remove validation indices from training set
        # print(val_indices.shape)
        train_indices = np.concatenate([
            np.arange(0, validation_period_start),
            np.setdiff1d(np.arange(validation_period_start, validation_period_end), val_indices)
        ])
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        if plot_validation:
            self.plot_validation_set(y_train, train_indices, y_val, val_indices)

        # print("SHAPES")
        # print(X_train.shape)
        # print(X_val.shape)
        # print(X_train.shape[0]+X_val.shape[0])

        # ensure no samples were lost during split
        assert(X_train.shape[0]+X_val.shape[0] == season_start)
        
        return X_train, y_train, X_val, y_val
    

    def plot_validation_set(self, train, train_idx, val, val_idx):
        """
        Plots the time series data, highlighting validation samples in a different color.
        
        Args:
            y (numpy.ndarray): The full target time series.
            train (numpy.ndarray): Indices of training data.
            val (numpy.ndarray): Indices of validation data.
        """
        train_splits = []
        train_idx_splits = []
        val_splits = []
        val_idx_splits = []

        train = train[:, -1]
        val = val[:, -1]

        current_split = []
        current_idx_split = []
        for i in range(len(train_idx)-1):
            current_split.append(train[i])
            current_idx_split.append(train_idx[i])
            if abs(train_idx[i+1] - train_idx[i]) > 1:
                # print(train_idx[i+1], train_idx[i])
                train_splits.append(current_split)
                train_idx_splits.append(current_idx_split)
                current_split = []
                current_idx_split = []
        
        train_splits.append(current_split)
        train_idx_splits.append(current_idx_split)
        
        current_split = []
        current_idx_split = []
        for i in range(len(val_idx)-1):
            current_split.append(val[i])
            current_idx_split.append(val_idx[i])
            if abs(val_idx[i+1] - val_idx[i]) > 1:
                # print(val_idx[i+1], val_idx[i])
                val_splits.append(current_split)
                val_idx_splits.append(current_idx_split)
                current_split = []
                current_idx_split = []
        
        val_splits.append(current_split)
        val_idx_splits.append(current_idx_split)

        # for t in range(len(train_splits)-1):
        #     train_splits[t+1].append(train_splits[t][-1])
        #     train_idx_splits[t+1].append(train_idx_splits[t][-1])
        
        # for t in range(len(val_splits)-1):
        #     val_splits[t+1].append(val_splits[t][-1])
        #     val_idx_splits[t+1].append(val_idx_splits[t][-1])

        plt.figure(figsize=(12, 6))

        assert(len(train_splits) == len(train_idx_splits))
        assert(len(val_splits) == len(val_idx_splits))
        
        for i in range(len(train_idx_splits)):
            train_indices = train_idx_splits[i]
            train = train_splits[i]
            plt.plot(train_indices, train, color="black", label="Training Data")

        for i in range(len(val_idx_splits)):
            val_indices = val_idx_splits[i]
            val = val_splits[i]
            plt.plot(val_indices, val, color="royalblue", label="Validation Data", linewidth=2)

        plt.xlabel("Time (Days)")
        plt.ylabel("ILI Rates")
        plt.title("Validation period")

        plt.show()
        sys.exit()   
