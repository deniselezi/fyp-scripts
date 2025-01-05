import pandas as pd

data_path = "../data/England/Q_freq_sparse.csv"
data = pd.read_csv(data_path, header=None)

last_date = int(data.max()[0])
n_queries = int(data.max()[1])
print(last_date, n_queries)

dense_df = data.pivot_table(index=data.columns[0], 
                            columns=data.columns[1], 
                            values=data.columns[2], 
                            fill_value=0)

# fill missing dates (days with no samples for any query) with 0s
dense_df = dense_df.reindex(range(1, last_date+1), fill_value=0)

dense_df = dense_df.reindex(columns=range(1, n_queries+1), fill_value=0)  # added this to not skip over missing columns

print(dense_df)

# save to csv
dense_df.to_csv("./processed_data/q_freq_dense.csv", header=False, index=False)
