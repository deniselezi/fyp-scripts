import pandas as pd

start_idx = 1340 - 1

end_idx = 5722 - 1

data_path = "./processed_data/filtered_data.csv"

data = pd.read_csv(data_path)

# align both frequencies and ILI rates to be between
# the start_idx date and the end_idx date
filtered_data = data.iloc[start_idx : end_idx + 1]

filtered_data.to_csv("./processed_data/filtered_data.csv", index=False)

ili_path = "./processed_data/ILI_rates_RCGP_daily.csv"

ili = pd.read_csv(ili_path, header=None)

filtered_ili = ili.iloc[start_idx : end_idx + 1]

filtered_ili.to_csv("./processed_data/filtered_ili.csv", index=False)
