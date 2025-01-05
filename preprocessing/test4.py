import pandas as pd

i = 326
j = 344

data_path = "../processed_data/q_freq_dense_smooth.csv"
dense_df = pd.read_csv(data_path, header=None)
dense_s = dense_df.iloc[:, 344]

data_path = "../processed_data/density_filtered.csv"
processed = pd.read_csv(data_path, header=None)
processed_s = processed.iloc[:, 326]

start_idx = 1340 - 1
end_idx = 5722 - 1

# restrict dense_df series to be between start_idx and end_idx, then compare it to
# the one extrated from processed

dense_series = dense_s.iloc[start_idx:end_idx + 1].reset_index(drop=True)
processed_series = processed_s.reset_index(drop=True)

# print(dense_series)
# print(processed_series)

# assert(dense_series.equals(processed_series))

eq = dense_series.eq(processed_series)

# for i in range (0, eq.size):
#     if eq.iloc[i] == False:
#         print("Mismatch at index", i)

print("Freq at index 2")
print("Original:", dense_series.iloc[2])
print("After processing:", processed_series.iloc[2])

print("Freq at index 145")
print("Original:", dense_series.iloc[145])
print("After processing:", processed_series.iloc[145])

print("Freq at index 567")
print("Original:", dense_series.iloc[567])
print("After processing:", processed_series.iloc[567])
