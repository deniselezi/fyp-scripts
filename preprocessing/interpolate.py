"""
This script handles the linear interpolation of the ILI rates, which
allows for the conversion of said rates from weekly to daily.

The rates are imported from the ILI_rates_RCGP file with pandas,
and each weekly rate is converted into 7 daily rates, which are written
to a new csv file in the processed_data subfolder.
"""

import pandas as pd


data_path = "../data/England/ILI_rates_RCGP.csv"
data = pd.read_csv(data_path)

weekly_rates = data.iloc[:, 2]

new_data_path = "./processed_data/ILI_rates_RCGP_daily.csv"
daily_rates_file = open(new_data_path, "w")
daily_rates_file.write("WeekStart,WeekEnd,ILIRate\n")
for i in range(len(weekly_rates) - 1):
    # each weekly rate represents the rate for thursday of that week
    thursday = weekly_rates[i]
    next_thursday = weekly_rates[i + 1]

    # linearly interpolate values from this thursday to next thursday
    step = (next_thursday - thursday) / 7

    for j in range(7):
        daily_rates_file.write(f"{thursday + (step)*j}\n")

daily_rates_file.close()
