import pandas as pd
import numpy as np

data = pd.read_csv("../processed_data/semantic_filtered.csv")

ili_rates = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

similarity_scores = {}

c = 0

ili_vector = ili_rates.iloc[:, 0].values.reshape(-1, 1)
for col in data.columns:
    if c % 50 == 0:
        print(c)
    c += 1

    data_vector = data[col].values.reshape(-1, 1)

    similarity = np.corrcoef(ili_vector[:, 0], data_vector[:, 0])[0, 1]

    similarity_scores[col] = similarity

similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
similarity_df = pd.DataFrame(similarity_scores, columns=["Query", "Correlation"])
similarity_df.to_csv("../processed_data/pearson_correlation_scores.csv", index=False)
print("Similarity scores saved to '../processed_data/similarity_scores.csv'.")

# sorted_columns = sorted(similarity_scores, key=similarity_scores.get, reverse=True)

# top_cols = sorted_columns[:800]

# top_data = data[top_cols]

# print(top_data)

# # save data
# top_data.to_csv("../processed_data/final_800.csv", index=False)

# print(f"Top columns saved")
