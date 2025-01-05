"""
This script handles the smoothing of the frequency values.
The smoothing is applied to reduce noise and variability in the data,
which could impact the predictive accuracy of models.

The smoothing process involves moving backwards through the data,
re-computing each frequency as a weighted average of all its
preceding frequencies. This average is computed using the current
frequency and the previous 13 values (when the look back window is 14).

The exact formulation of the smoothing can be found in the report.
"""

import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

import pandas as pd


data_path = "./processed_data/q1_time_series.csv"
data = pd.read_csv(data_path, header=None)

smoothed_freqs = []
freqs = data.iloc[:, 1]

print(data)

for i in range(len(freqs)-1, -1, -1):  # iterate over series of frequencies
    j = i
    c = 14

    # previous terms
    num = 0
    denom = 0
    while j >= 0 and c > 0:
        term = c / 14 # 1, 1/2, 1/3 etc.
        # print(term)
        num += (term * freqs[j])
        denom += term
        j -= 1
        c -= 1

    smoothed = num / denom

    smoothed_freqs.append(smoothed)

# print(smoothed_freqs[::-1])  # reverse to get the correct ordering,
# since we iterated through the series backwards

assert(len(freqs) == len(smoothed_freqs))

smoothed_freqs.reverse()

def save_smoothing():
    dates = data.iloc[:, 0]
    queries = data.iloc[:, 1]
    smooth_file = open("./processed_data/q1_time_series_smoothed.csv", 'w')
    for i in range(len(smoothed_freqs)):
        smooth_file.write(f"{dates[i]},{queries[i]},{smoothed_freqs[i]}\n")

    smooth_file.close()


def plot_smoothing():
    dates_path = "../data/England/dates.txt"
    dates_file = open(dates_path)
    dates = dates_file.readlines()
    dates_file.close()
    q_dates = [dates[int(i)-1] for i in data.iloc[:, 0]]
    mpldates = [mpl_dates.num2date(mpl_dates.datestr2num(d)) for d in q_dates]
    print(mpldates)
    plt.plot(mpldates, freqs, c='b', label="Normal")
    plt.plot(mpldates, smoothed_freqs, c='r', label="Smoothed")
    plt.legend(loc="upper right")
    plt.xlabel("Time")
    plt.ylabel("Freqs")
    plt.show()


save_smoothing()
