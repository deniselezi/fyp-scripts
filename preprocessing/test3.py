import pandas as pd
from matplotlib import pyplot as plt

og = pd.read_csv("../processed_data/q1_time_series.csv")
smoothed = pd.read_csv("../processed_data/q1_time_series_smoothed.csv")

start_idx = 1340 - 1
end_idx = 5722 - 1

og = og.iloc[start_idx:end_idx+1, :]
smoothed = smoothed.iloc[start_idx:end_idx+1, :]

print(og.shape)
print(smoothed.shape)

x_axis = list(range(0, max(og.shape)))
x_axis = [start_idx + i for i in x_axis]

# print(x_axis)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, og, label='Original', color='black')
plt.plot(x_axis, smoothed, label='Smoothed', color='royalblue')

# Add labels, title, and legend
plt.title('Smoothed vs Original "flu medicine"', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
