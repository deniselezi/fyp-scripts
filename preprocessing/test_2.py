import pandas as pd

data_path = "../../data/England/Q_freq_sparse.csv"
data = pd.read_csv(data_path, header=None)

treating_flu_symptoms = data[data.iloc[:, 1] == 19181.0]

data_in_range = treating_flu_symptoms[(treating_flu_symptoms.iloc[:, 0] >= 1340.0) & (treating_flu_symptoms.iloc[:, 0] <= 5722.0)]

print(data_in_range)

