from RNNer import RNNer
import math


model = RNNer(2, window_size=28, data_path="../processed_data/final_300.csv", hidden_size=[200, 100], hindcasting=True, advanced_validation=True)
model.fit(stop_early=True, batch_size=16, lr=0.001, plot_training=True, plot_results=True)

# 0.04830671474337578,  

# data_paths = ["../processed_data/final_600.csv", "../processed_data/final_700.csv", "../processed_data/final_800.csv"]
# # dropouts = [0, 0.1, 0.2]
# batches = [16, 32]
# lrs = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
# structures = [[50, 25], [100, 50], [200, 100]]
# input_lengths = [28]

# data_paths = ["../processed_data/final_200.csv", "../processed_data/final_300.csv"]
# # dropouts = [0, 0.1, 0.2]
# batches = [16, 32]
# lrs = [0.00005, 0.0001, 0.0005, 0.001]
# structures = [[200, 100]]
# input_lengths = [28]


# losses = [0.006782899610698223, 0.005241742357611656, 0.06198093667626381, 0.005801175255328417]

# for window_size in input_lengths:
#     f = open(f"results_{window_size}.txt", "a+")
#     for season in [0, 1, 3]:
#         best_loss= math.inf
#         # best_dropout = None
#         best_lr = None
#         best_batch = None
#         best_struct = None
#         best_d = None
#         print(f"SEASON {season+1}")
#         f.write(f"Season {season+1}\n")
#         for d in data_paths:
#             for struct in structures:
#                 for b in batches:
#                     # for d in dropouts:
#                     for lr in lrs:
#                         print(season+1, d, struct, b, lr)
#                         model = RNNer(season+1, data_path=d, window_size=window_size, hidden_size=struct, hindcasting=True, advanced_validation=True)
#                         mae, mse, corr, loss = model.fit(stop_early=True, batch_size=b, lr=lr, plot_training=False, plot_results=False)
#                         if loss < best_loss:
#                             print(f"Loss update, before: {best_loss}, now: {loss}")
#                             best_loss = loss
#                             best_batch = b
#                             best_struct = struct
#                             best_d = d
#                             best_lr = lr


#         f.write(f"The set of parameters for season {season+1} is\nData: {best_d}\nStructure: {struct}\nLR: {best_lr}\nBatch size: {best_batch}\n\n")
#     f.close()


# [50], 0.001, 16
# [200, 100], 0.0001, 16


# params = [("../processed_data/final_200.csv", 16, 0.0001), ("../processed_data/final_700.csv", 32, 0.01), ("../processed_data/final_200.csv", 16, 0.0001), ("../processed_data/final_700.csv", 32, 0.01), ("../processed_data/final_200.csv", 16, 0.0001), ("../processed_data/final_700.csv", 32, 0.01)]
# final_losses = []  # 0.007944912649691105, 0.03972289711236954

# for path, b, lr in params:
#     model = RNNer(3, window_size=28, data_path=path, hidden_size=[200, 100], hindcasting=True, advanced_validation=True)
#     model.fit(stop_early=True, batch_size=b, lr=lr, plot_training=False, plot_results=False)

# print(final_losses)

