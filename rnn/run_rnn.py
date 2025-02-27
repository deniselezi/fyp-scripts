from RNNer import RNNer
import math


# window, hidden_size, num_layers
# 7, 64, 1 -> (1.13, 3.13, 0.96)
# 14, 48, 1 - 
# 21, 32, 2 (15-16)
# 14, 48, 2
# 28, 96, 3 -> 1.27, 3.73


# model = RNNer(1, window_size=14, hidden_size=[100, 50], dropout=0.0, hindcasting=True)
# model.fit(stop_early=True, batch_size=16, lr=0.005, plot_training=True, plot_results=True)

dropouts = [0.05, 0.1]
lrs = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

best_mae = math.inf
best_params = None
f = open("results.txt", "a+")
for d in dropouts:
    for lr in lrs:
        model = RNNer(1, window_size=14, hidden_size=[50], dropout=d, hindcasting=True)
        mae, mse, corr = model.fit(stop_early=True, batch_size=16, lr=lr, plot_training=False, plot_results=False)
        if mae < best_mae:
            best_mae = mae
            best_params = (d, lr)
        f.write(f"For dropout {d} and learning rate {lr}:\n")
        f.write(f"MAE: {mae}\nMSE: {mse}\nCorrelation: {corr}\n\n")

f.write(f"the best parameters are: ({best_params[0]}, {best_params[1]})")
f.close()


# [50], 0.001, 16
# [200, 100], 0.0001, 16