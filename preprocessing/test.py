import pandas as pd

data_path = "../processed_data/q_freq_dense_smooth.csv"
dense_df = pd.read_csv(data_path, header=None)

# "treating flu symptoms is at 19181 in queries file, so at index 19180 in dense"

# extract column at index 19180 from dense_df here as a series
dense_s = dense_df.iloc[:, 73]

data_path = "../processed_data/final_data_pearson_1000.csv"
processed = pd.read_csv(data_path)

# extract column at index 0 from processed here as a series
processed_s = processed.iloc[:, 4]

start_idx = 1340 - 1
end_idx = 5722 - 1

# restrict dense_df series to be between start_idx and end_idx, then compare it to
# the one extrated from processed

dense_series = dense_s.iloc[start_idx:end_idx + 1].reset_index(drop=True)
processed_series = processed_s.reset_index(drop=True)

print(dense_series)
print(processed_series)

# assert(dense_series.equals(processed_series) == True)

eq = dense_series.eq(processed_series)

# for i in range (0, eq.size):
#     if eq.iloc[i] == False:
#         print("Mismatch at index", i)

print("Freq at index 1")
print("Original:", dense_series.iloc[1])
print("After processing:", processed_series.iloc[1])

print("Freq at index 60")
print("Original:", dense_series.iloc[60])
print("After processing:", processed_series.iloc[60])

print("Freq at index 237")
print("Original:", dense_series.iloc[237])
print("After processing:", processed_series.iloc[237])
