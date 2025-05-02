import pandas as pd

top_n = 2000

data = pd.read_csv("../processed_data/density_filtered.csv", header=None)
print(data.shape)
max_id = data.shape[1] - 1
scores = pd.read_csv("../processed_data/scores.csv", nrows=top_n)
queries = pd.read_csv("../processed_data/filtered_queries.csv", header=None)

idxs = [i - 1 for i in scores.iloc[:, 0]]
print("Best query:", idxs[0])
print(max(idxs))

filtered_queries = queries.iloc[idxs].transpose()
# print(filtered_queries)
filtered_data = data.iloc[:, idxs]
filtered_data.columns = filtered_queries.iloc[0]  # set queries as header

print(filtered_data)
print(filtered_data.shape)
filtered_data.to_csv("../processed_data/semantic_filtered.csv", index=False)
