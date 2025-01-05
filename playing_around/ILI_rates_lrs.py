import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.nn import Linear, MSELoss

assert(torch.cuda.is_available() == True)

data_path = "../data/England"

# prepare data
ili_rates = pd.read_csv(data_path+"/ILI_rates_RCGP.csv")
labels = torch.tensor(ili_rates['ILIRate'].values)
dim = labels.size(dim=0)
inputs = torch.tensor(list(range(dim)))

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.fc1 = Linear(input_dim, output_dim)  # input dimension, output dimension
    
    def forward(self, x):
        return self.fc1(x)


model = LinearRegression(dim, dim)
criterion = MSELoss()  # instantiate loss

lr = 0.0000001
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

epochs = 1000
for e in range(epochs):
    preds = model(inputs.float())
    optimizer.zero_grad()

    loss = criterion(preds.float(), labels.float())  # equivalent to (output - label)**2 for each output and label, since LSR

    loss.backward()

    optimizer.step()
    
    if (e+1) == 1 or (e+1) % 100 == 0:
        print('epoch {}, loss {}'.format(e+1, loss.data))


preds = model(inputs.float())

np_inputs = inputs.detach().cpu().numpy()
np_labels = labels.detach().cpu().numpy()
plt.scatter(x=np_inputs, y=np_labels, c='b', marker='x', s=10)

np_preds = preds.detach().cpu().numpy()
plt.plot(np_inputs, np_preds)
plt.show()

