import pandas as pd

data = pd.read_csv("../processed_data/q_freq_dense_smooth.csv", header=None)

start_idx = 1340 - 1
end_idx = 5722 - 1

# get all columns which do not contain just 0s, and save as new dataframe
filtered_data = data.loc[start_idx:end_idx, data.loc[start_idx:end_idx].mean() != 0]
filtered_data.to_csv(
    "../processed_data/density_filtered.csv", index=False, header=False
)
