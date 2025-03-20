import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class Lassoer:
    def __init__(self, season, alpha=None, model="lasso", advanced_validation=False):
        self.season = season
        self.season_start, self.season_end = self._get_season_dates()
        X_train, y_train, X_test, y_test = self._read_data()
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.X_train_scaled = self.x_scaler.fit_transform(X_train)
        self.X_test_scaled = self.x_scaler.transform(X_test)
        self.y_train_scaled = self.y_scaler.fit_transform(y_train)
        self.y_test_scaled = self.y_scaler.transform(y_test)

        days = 365
        if advanced_validation:
            # print(X_train_scaled.shape)
            self.X_train_scaled, self.y_train_scaled, self.X_val, self.y_val = self._validation_set(self.X_train_scaled, self.y_train_scaled)
            # print(X_train_scaled.shape)
            # print(X_val.shape)
        else:  # last year as validation set
            self.X_val = self.X_train_scaled[self.season_start - days : self.season_start]
            self.y_val = self.y_train_scaled[self.season_start - days : self.season_start]
            self.X_train_scaled = self.X_train_scaled[: self.season_start - days]
            self.y_train_scaled = self.y_train_scaled[: self.season_start - days]

        
        if model == "lasso":
            self.max_iter = 100000
            self.tol = 0.0001
            # if no alpha provided, find optimal alpha for the Lasso model
            best_alpha = alpha if alpha != None else self._find_optimal_alpha()
            
            self.model = Lasso(alpha=best_alpha, max_iter=self.max_iter, tol=self.tol)
            # zero_weights = np.sum(self.model.coef_ == 0)
            # print(f"The model density is {((1000 - zero_weights) / 1000)*100}%")
        else:
            print("model = LinearRegression")
            self.model = LinearRegression()

    def fit(self, show_chart=True):
        self.model.fit(self.X_train_scaled, self.y_train_scaled)

        zero_weights = np.sum(self.model.coef_ == 0)
        print(f"The number of zero weights is {zero_weights}")

        y_pred = self.model.predict(self.X_test_scaled)

        self.og_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        self.og_actual = self.y_scaler.inverse_transform(self.y_test_scaled.reshape(-1, 1))

        print("Mean absolute error: %.2f" % mean_absolute_error(self.og_actual, self.og_pred))
        print("Mean squared error: %.2f" % mean_squared_error(self.og_actual, self.og_pred))
        print("Correlation: %.2f" % np.corrcoef(self.og_actual.reshape(-1), self.og_pred.reshape(-1))[0, 1])

        if show_chart:
            self._show_chart()


    def _show_chart(self):
        days = [i - 2920 for i in list(range(self.season_start, self.season_end+1))]

        plt.plot(days, self.og_actual, color="black", label="Actual")
        plt.plot(days, self.og_pred, color="royalblue", label="Predicted")

        plt.xlabel("Days")
        plt.ylabel("ILI Rates")
        plt.title("Actual vs Predicted ILI Rates (15-16)")
        plt.legend()

        plt.xticks(())
        plt.yticks(())

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


    def _find_optimal_alpha(self):
        # 2015 - 2016 season has 1 extra day as 2016 is a leap year
        print('Finding optimal alpha...')

        # preliminary enumeration needed to find the alphas corresponding to 5% and 95%
        # density. Optimal alpha will be between these 2
        alphas = np.linspace(3.6e-06, 0.001, 100)
        zpercs = []
        for a in alphas:
            print(a)
            model = Lasso(alpha=a, max_iter=self.max_iter, tol=self.tol)
            model.fit(self.X_train_scaled, self.y_train_scaled)
            zero_weights = np.sum(model.coef_ == 0)
            zpercs.append(((1000 - zero_weights) / 1000)*100)  # 1000 here is size of input data

        self._plot_model_density(alphas, zpercs)

        zpercs = np.asarray(zpercs)
        a_1_idx = (np.abs(zpercs - 95)).argmin()
        a_2_idx = (np.abs(zpercs - 5)).argmin()
        
        # and since alphas were generate through linspace
        a_1 = ((0.001 - 0) / 100) * a_1_idx
        a_2 = ((0.001 - 0) / 100) * a_2_idx

        a_2 = a_2 / 10

        print(zpercs)
        print("Alpha 1 is", a_1, "and alpha 2 is", a_2)
        # # input()

        alphas = np.linspace(3.6e-06, a_2, 100)
        maes = []
        for a in alphas:
            print(a)
            model = Lasso(alpha=a, max_iter=self.max_iter, tol=self.tol)
            model.fit(self.X_train_scaled, self.y_train_scaled)
            y_pred = model.predict(self.X_val)
            og_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
            og_actual = self.y_scaler.inverse_transform(self.y_val.reshape(-1, 1))
            maes.append(mean_absolute_error(og_actual, og_pred)) 

        self._plot_mae_difference(alphas, maes)
        
        best_mae_i = np.argmin(maes)
        best_alpha = alphas[best_mae_i]

        print("The optimal alpha is:", best_alpha)

        return best_alpha


    def _plot_mae_difference(self, alphas, maes):
        plt.plot(alphas, maes, color="royalblue")

        plt.xlabel("Alpha")
        plt.ylabel("MAE")
        plt.title("Change in MAE across alphas")
        plt.show()
    

    def _plot_model_density(self, alphas, zfs):
        plt.plot(alphas, zfs, color="royalblue")

        plt.xlabel("Alpha")
        plt.ylabel("Percentage of non-zeroed weights")
        plt.title("Non-zero weights across alphas")
        plt.show()


    def _read_data(self):
        print('Reading data...')
        X = pd.read_csv("../processed_data/final_1000.csv")
        y = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

        X_train, y_train = X[:self.season_start], y[:self.season_start]
        X_test, y_test = X[self.season_start:self.season_end+1], y[self.season_start:self.season_end+1]

        return X_train, y_train, X_test, y_test
    

    def _validation_set(self, X_train, y_train, plot_validation=False):
        """
        Selects the validation set using an advanced strategy based on onset, peak, and outset periods.
        """
        print("validating advancedly")
        # print(X_train.shape)
        season_start = self.season_start
        validation_period_start = season_start - 3 * 365
        validation_period_end = season_start
        print(f"Validation period: {validation_period_start} - {validation_period_end}")
        validation_period_y = y_train[validation_period_start:validation_period_end]
        validation_period_y = validation_period_y.transpose()[-1]
        print(validation_period_y.shape)
        
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
        print(val_indices.shape)
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
    
