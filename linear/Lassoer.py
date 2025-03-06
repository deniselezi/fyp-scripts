import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class Lassoer:
    def __init__(self, season, alpha=None, model="lasso"):
        self.season = season
        self.season_start, self.season_end = self._get_season_dates()
        X_train, y_train, X_test, y_test = self._read_data()
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.X_train_scaled = self.x_scaler.fit_transform(X_train)
        self.X_test_scaled = self.x_scaler.transform(X_test)
        self.y_train_scaled = self.y_scaler.fit_transform(y_train)
        self.y_test_scaled = self.y_scaler.transform(y_test)
        
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
        days = 366 if self.season == 1 else 365
        alpha_X_train = self.X_train_scaled[:self.season_start-days]
        alpha_X_val = self.X_train_scaled[self.season_start-days:self.season_start]
        alpha_Y_train = self.y_train_scaled[:self.season_start-days]
        alpha_Y_val = self.y_train_scaled[self.season_start-days:self.season_start]

        # preliminary enumeration needed to find the alphas corresponding to 5% and 95%
        # density. Optimal alpha will be between these 2
        alphas = np.linspace(0, 0.001, 100)
        zpercs = []
        for a in alphas:
            model = Lasso(alpha=a, max_iter=self.max_iter, tol=self.tol)
            model.fit(alpha_X_train, alpha_Y_train)
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
        # input()

        alphas = np.linspace(a_1, a_2, 1000)
        maes = []
        for a in alphas:
            model = Lasso(alpha=a, max_iter=self.max_iter, tol=self.tol)
            model.fit(alpha_X_train, alpha_Y_train)
            y_pred = model.predict(alpha_X_val)
            og_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
            og_actual = self.y_scaler.inverse_transform(alpha_Y_val.reshape(-1, 1))
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
