import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from FFNN import FFNN
from EarlyStopper import EarlyStopper


class FFNNer:
    def __init__(self, season, data_path, seed=1, advanced_validation=False):
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.season = season
        self.season_start, self.season_end = self._get_season_dates()
        X_train, y_train, X_test, y_test = self._read_data(data_path)
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        X_test_scaled = self.x_scaler.transform(X_test)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        y_test_scaled = self.y_scaler.transform(y_test)

        days = 365

        if advanced_validation:
            X_train_scaled, y_train_scaled, X_val, y_val = self._validation_set(
                X_train_scaled, y_train_scaled
            )
        else:
            # last year as validation set
            X_val = X_train_scaled[self.season_start - days : self.season_start]
            y_val = y_train_scaled[self.season_start - days : self.season_start]
            X_train_scaled = X_train_scaled[: self.season_start - days]
            y_train_scaled = y_train_scaled[: self.season_start - days]

        self.X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(
            device
        )
        self.y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(
            device
        )
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        self.X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        self.y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

        input_dim = X_train.shape[1]
        self.model = FFNN(input_dim).to(device)

    def fit(
        self,
        num_epochs=500,
        plot_training=True,
        plot_results=True,
        stop_early=True,
        patience=10,
        warmup=20,
        lr=0.001,
        batch_size=32,
        tune=False,
    ):
        self.train_loader = DataLoader(
            TensorDataset(self.X_train_tensor, self.y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        train_losses, val_losses, stop_epoch = self._train(
            num_epochs, stop_early, warmup, lr, patience
        )

        final_loss = val_losses[-(patience + 1)]
        print("Final loss:", final_loss)

        if tune:  # this is a tuning run, i.e. we are interested in validation loss
            return final_loss  # so we return here and dont use test set

        if plot_training:
            self._plot_training(stop_epoch + patience, train_losses, val_losses)

        self.model.eval()
        with torch.no_grad():
            y_pred_tensor = self.model(self.X_test_tensor)

        y_pred_scaled = y_pred_tensor.cpu().numpy()
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        y_actual = self.y_scaler.inverse_transform(self.y_test_tensor.cpu().numpy())

        mae = mean_absolute_error(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        correlation = np.corrcoef(y_actual.flatten(), y_pred.flatten())[0, 1]

        results = [
            f"Mean Absolute Error: {mae:.2f}",
            f"Mean Squared Error: {mse:.2f}",
            f"Correlation: {correlation:.2f}",
        ]
        print(results)

        if plot_results:
            self._plot_results(y_actual, y_pred)

        return results, [mae, mse, correlation]

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
        if self.season == 1:
            days = list(range(1, 367))
        else:
            days = list(range(1, 366))
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

    def _read_data(self, path):
        X = pd.read_csv(path)
        y = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

        X_train, y_train = X[: self.season_start], y[: self.season_start]
        X_test, y_test = (
            X[self.season_start : self.season_end + 1],
            y[self.season_start : self.season_end + 1],
        )

        return X_train, y_train, X_test, y_test

    def _validation_set(self, X_train, y_train, plot_validation=False):
        """
        Selects the validation set using an advanced strategy based on onset, peak, and outset periods.
        """
        season_start = self.season_start
        validation_period_start = season_start - 3 * 365
        validation_period_end = season_start
        validation_period_y = y_train[validation_period_start:validation_period_end]
        validation_period_y = validation_period_y.transpose()[-1]

        # calculate threshold
        mean_y = np.mean(y_train[:validation_period_start])
        std_y = np.std(y_train[:validation_period_start])
        threshold = mean_y - 0.25 * std_y

        # onset period (3rd year in validation period)
        onset_start_idx = None
        for i in range(2 * 365, 3 * 365):
            if (validation_period_y[i : i + 14] > threshold).all():
                onset_start_idx = i
                break

        if onset_start_idx is None:
            raise ValueError("Onset period not found.")

        onset_window_start = validation_period_start + onset_start_idx - 30
        onset_window_end = onset_window_start + 60

        # peak period (2nd year in validation period)
        peak_idx = np.argmax(validation_period_y[365 : 2 * 365]) + 365
        peak_window_start = validation_period_start + peak_idx - 30
        peak_window_end = peak_window_start + 60

        # outset period (1st year in validation period)
        outset_start_idx = None
        for i in range(365):
            if validation_period_y[i] > threshold:
                outset_start_idx = i

        if outset_start_idx is None:
            raise ValueError("Outset period not found.")

        outset_window_start = validation_period_start + outset_start_idx - 30
        outset_window_end = outset_window_start + 60

        # Create validation set
        val_indices = np.concatenate(
            [
                np.arange(onset_window_start, onset_window_end),
                np.arange(peak_window_start, peak_window_end),
                np.arange(outset_window_start, outset_window_end),
            ]
        )

        X_val = X_train[val_indices]
        y_val = y_train[val_indices]

        # remove validation indices from training set
        train_indices = np.concatenate(
            [
                np.arange(0, validation_period_start),
                np.setdiff1d(
                    np.arange(validation_period_start, validation_period_end),
                    val_indices,
                ),
            ]
        )
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        if plot_validation:
            self.plot_validation_set(y_train, train_indices, y_val, val_indices)

        # ensure no samples were lost during split
        assert X_train.shape[0] + X_val.shape[0] == season_start

        return X_train, y_train, X_val, y_val
