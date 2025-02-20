from RNNer import RNNer

# window, hidden_size, num_layers
# 7, 64, 1 -> (1.13, 3.13, 0.96)
# 14, 48, 1 - 
# 21, 32, 2 (15-16)
# 14, 48, 2
# 28, 96, 3 -> 1.27, 3.73

# model = RNNer(1, window_size=14, hidden_size=[200, 100], dropout=0.5)
model = RNNer(1, window_size=14, hidden_size=[50], dropout=0.0)
model.fit(stop_early=True, batch_size=32, lr=0.01)


# [50], 0.001, 16
# [200, 100], 0.0001, 16