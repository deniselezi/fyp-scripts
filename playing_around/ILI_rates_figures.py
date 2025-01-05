from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

data_path = "../data/England"

ili_rates = pd.read_csv(data_path+"/ILI_rates_RCGP.csv")

# print(ili_rates.iloc[:, 2])
# print(len(ili_rates.index))

seasons_map = {
    "January": "Winter",
    "February": "Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Autumn",
    "October": "Autumn",
    "November": "Autumn",
    "December": "Winter"
}

dates = defaultdict(list)
rates = defaultdict(list)

for idx, row in ili_rates.iterrows():
    timestamp = pd.to_datetime(row['WeekStart'])
    dates[seasons_map[timestamp.month_name()]].append(timestamp)
    rates[seasons_map[timestamp.month_name()]].append(row['ILIRate'])

# print([d.month_name() for d in dates])

plt.scatter(x=dates["Winter"], y=rates["Winter"], c='b', marker='x', s=10, label="Winter")
plt.scatter(x=dates["Spring"], y=rates["Spring"], c='g', marker='x', s=10, label="Spring")
plt.scatter(x=dates["Summer"], y=rates["Summer"], c='r', marker='x', s=10, label="Summer")
plt.scatter(x=dates["Autumn"], y=rates["Autumn"], c='m', marker='x', s=10, label="Autumn")
plt.legend(loc="upper left")
plt.show()
