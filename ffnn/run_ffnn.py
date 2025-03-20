from FFNNer import FFNNer

import numpy as np
import math
import random

# data_paths = ["../processed_data/final_200.csv", "../processed_data/final_300.csv", "../processed_data/final_400.csv", "../processed_data/final_500.csv"]
# batches = [8, 16, 32]
# lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
# seasons = [0, 1, 3]
# # lrs = [0.0005]

# f = open("results.txt", "a+")
# for season in seasons:
#     print(f"season {season+1}")
#     f.write(f"Season {season+1}\n")
#     best_loss = math.inf
#     best_lr = None
#     best_d = None
#     best_batch = None
#     for d in data_paths:
#         for bsize in batches:
#             for rate in lrs:
#                 print(d)
#                 print(bsize, rate)
#                 model = FFNNer(season+1, advanced_validation=True, data_path=d)
#                 loss = model.fit(stop_early=True, lr=rate, batch_size=bsize, tune=True)
#                 if loss < best_loss:
#                     best_loss = loss
#                     best_lr = rate
#                     best_batch = bsize
#                     best_d = d
#                 print(rate, loss)
#     f.write(f"The set of parameters for season {season+1} is\nData: {best_d}\nLR: {best_lr}\nBatch size: {best_batch}\n")

# f.write("\n")
# f.close()


"""
Season 1
The best learning rate for batch size 16 is 0.001 with an MAE loss of 0.007026683073490858
The best learning rate for batch size 32 is 0.0005 with an MAE loss of 0.007141199428588152
Season 2
The best learning rate for batch size 16 is 0.0005 with an MAE loss of 0.006337430793792009
The best learning rate for batch size 32 is 0.0005 with an MAE loss of 0.007393523119390011
Season 3
The best learning rate for batch size 16 is 0.0001 with an MAE loss of 0.007590676657855511
The best learning rate for batch size 32 is 0.005 with an MAE loss of 0.00828856322914362
Season 4
The best learning rate for batch size 16 is 0.0001 with an MAE loss of 0.006608229596167803
The best learning rate for batch size 32 is 0.001 with an MAE loss of 0.0071821315214037895
"""


model = FFNNer(4, advanced_validation=True, data_path="../processed_data/final_300.csv")
model.fit(stop_early=True, lr=0.001, batch_size=16, tune=False)




# seeds = [random.randint(0, 10000) for _ in range(5)]
# params = [(0.005, 32), (0.001, 32), (0.001, 16), (0.0005, 16)]

# f = open("results.txt", "+a")
# for season, season_params in enumerate(params):
#     lr, batch = season_params
#     print(f"season {season+1}")
#     f.write(f"Season {season+1}\n\n\n")
#     maes = []
#     mses = []
#     corrs = []
#     for seed in seeds:
#         f.write(f"Seed: {seed}\n")
#         model = FFNNer(season+1, seed=seed)
#         results, nums = model.fit(lr=lr, batch_size=batch, plot_training=False, plot_results=False)
#         maes.append(float(nums[0]))
#         mses.append(float(nums[1]))
#         corrs.append(float(nums[2]))
#         f.write(f"{results[0]}\n{results[1]}\n{results[2]}\n\n")
#     f.write(f"Mean MAE: {np.mean(maes)}\n")
#     f.write(f"Mean MSE: {np.mean(mses)}\n")
#     f.write(f"Mean correlation: {np.mean(corrs)}\n")
    
#     f.write("\n")
# f.close()

