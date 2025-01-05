"""
This script extracts a single time series from the sparse matrix.
Set query to the index (float) of the query for which you wish to
extract the time series.
"""

import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mpl_dates


data_path = "../data/England/Q_freq_sparse.csv"
data = pd.read_csv(data_path, header=None)
# data = pd.read_csv(data_path)

last_date = int(data.max()[0])

query = 1.0

q1_samples = data.loc[data[data.columns[1]] == query]
print("Query 1 Time Series")
print(q1_samples)

q1_dates = []
q1_freqs = []

# print(q1_samples.iloc[:, 0])
sample_dates = q1_samples.iloc[:, 0].values
for i in range(1, last_date+1):
    q1_dates.append(i)
    if float(i) in sample_dates:
        freq = q1_samples.loc[q1_samples.iloc[:, 0] == float(i), q1_samples.columns[2]].values
        # if freq[0] < 100:
        #     q1_freqs.append(freq[0])
        # else:
        #     q1_freqs.append(0)  # avoid outliers
        q1_freqs.append(freq[0])
    else:
        q1_freqs.append(0)  # add removed 0s (privacy preservation)

assert(len(q1_dates) == len(q1_freqs))

data = None  # free memory

q1_series = pd.DataFrame(zip(q1_dates, q1_freqs))
print(q1_series)

dates_path = "../data/England/dates.txt"
dates_file = open(dates_path)
dates = dates_file.readlines()
dates_file.close()

# q1_dates = dates

q1_dates = [dates[i-1] for i in q1_dates]

# print(dates)

# print([mpl_dates.num2date(mpl_dates.datestr2num(d)) for d in q1_dates])

q1_dates = [mpl_dates.num2date(mpl_dates.datestr2num(d)) for d in q1_dates]

# print(q1_series.iloc[:, 2])

# write Q1 dataframe to csv
q1_series.to_csv("./processed_data/q2_time_series.csv", header=False, index=False)

# plot time series
plt.title('"flu medicine" query frequency over time')
plt.xlabel("Time")
plt.ylabel("Freq")
plt.plot(q1_dates, q1_freqs)
plt.show()
