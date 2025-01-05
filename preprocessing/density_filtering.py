import pandas as pd

# Read the CSV file
data = pd.read_csv("../processed_data/q_freq_dense_smooth.csv", header=None)

start_idx = 1340 - 1

end_idx = 5722 - 1

# columns with mean is 0
# columns_to_remove = set(data.loc[start_idx:end_idx].mean()[data.loc[start_idx:end_idx].mean() == 0].index)

# Load the queries file
# queries = pd.read_csv("../../data/England/queries.txt", header=None)

# Filter queries based on columns to remove
# filtered_queries = queries.loc[~queries.index.isin(columns_to_remove)]

# filtered_queries.to_csv("../processed_data/filtered_queries.csv", header=False, index=False)

# Filter out columns where the mean equals 0
filtered_data = data.loc[start_idx:end_idx, data.loc[start_idx:end_idx].mean() != 0]

# Save the new DataFrame to another CSV file
filtered_data.to_csv("../processed_data/density_filtered.csv", index=False, header=False)
