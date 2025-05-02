from RNNer import RNNer
# import torch


model = RNNer(
    4,
    window_size=28,
    data_path="../processed_data/final_200.csv",
    hidden_size=[200, 100],
    hindcasting=True,
    advanced_validation=True,
)
model.fit(
    stop_early=True, batch_size=32, lr=5e-05, plot_training=False, plot_results=True
)

# to store the model's state_dict
# torch.save(model.model.state_dict, "rnn.pt")
