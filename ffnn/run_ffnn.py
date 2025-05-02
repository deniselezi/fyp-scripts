from FFNNer import FFNNer
# import torch


model = FFNNer(4, advanced_validation=True, data_path="../processed_data/final_300.csv")
model.fit(stop_early=True, lr=0.001, batch_size=16, tune=False, plot_training=False)

# to store the model's state_dict
# torch.save(model.model.state_dict(), "ffnn.pt")
