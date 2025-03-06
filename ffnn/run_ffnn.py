from FFNNer import FFNNer

import numpy as np
import math
import random

"""
Best learning rates (last loss, last k losses):
15-16: 0.0059, 0.00373673469387755
16-17: 0.0035, 0.00333265306122449
17-18: 0.0081, 0.00636326530612245
18-19: 0.006969387755102042, 0.005555102040816327

Best learning rates, custom LRs (32 batch size, 16 batch size):
15-16: 0.005 (1.34, 4.83, 0.95), 0.001 (1.45, 5.51, 0.93)
16-17: 0.001 (2.42, 10.53, 0.96), 0.001 (2.64, 12.45, 0.95)
17-18: 0.01 (3.15, 33.32, 0.95), 0.01 (2.26, 14.78, 0.96)
18-19: 0.01 (1.65, 8.24, 0.95), 0.0005 (1.67, 4.91, 0.97)
"""

# batches = [8]
# lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# f = open("results3.txt", "+a")
# for season in range(2):
#     print(f"season {season+1}")
#     f.write(f"Season {season+1}\n")
#     for bsize in batches:
#         best_lr = None
#         best_loss = math.inf
#         for rate in lrs:
#             model = Neuraler(season+1)
#             loss = model.fit(stop_early=True, lr=rate, batch_size=bsize, tune=True)
#             if loss < best_loss:
#                 best_loss = loss
#                 best_lr = rate
#             print(rate, loss)
#         f.write(f"The best learning rate for batch size {bsize} is {best_lr} with an MAE loss of {best_loss}\n")

# f.write("\n")
# f.close()

seeds = [random.randint(0, 10000) for _ in range(5)]
params = [(0.005, 32), (0.001, 32), (0.001, 16), (0.0005, 16)]

f = open("results.txt", "+a")
for season, season_params in enumerate(params):
    lr, batch = season_params
    print(f"season {season+1}")
    f.write(f"Season {season+1}\n\n\n")
    maes = []
    mses = []
    corrs = []
    for seed in seeds:
        f.write(f"Seed: {seed}\n")
        model = FFNNer(season+1, seed=seed)
        results, nums = model.fit(lr=lr, batch_size=batch, plot_training=False, plot_results=False)
        maes.append(float(nums[0]))
        mses.append(float(nums[1]))
        corrs.append(float(nums[2]))
        f.write(f"{results[0]}\n{results[1]}\n{results[2]}\n\n")
    f.write(f"Mean MAE: {np.mean(maes)}\n")
    f.write(f"Mean MSE: {np.mean(mses)}\n")
    f.write(f"Mean correlation: {np.mean(corrs)}\n")
    
    f.write("\n")
f.close()


# model = Neuraler(1)
# model.fit(stop_early=True, lr=0.0005, batch_size=8, tune=False)
