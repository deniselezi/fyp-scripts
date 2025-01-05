import pandas as pd

data_path = "../data/England"

q_freq_sparse = pd.read_csv(data_path+"/q_freq_sparse.csv", nrows=1000)

print(list(q_freq_sparse))

print(q_freq_sparse)
