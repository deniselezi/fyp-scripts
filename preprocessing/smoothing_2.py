import pandas as pd

# data_path = "../processed_data/q1_time_series.csv"
# q1_series = pd.read_csv(data_path, header=None)

# k = 1

def smooth_series(freqs):
    """Smooth a series using the provided algorithm."""
    global k
    if k % 20 == 0:
        print(k)
    k += 1

    smoothed_freqs = []
    
    for i in range(len(freqs)-1, -1, -1):  # iterate over series of frequencies
        j = i
        c = 14

        # previous terms
        num = 0
        denom = 0
        while j >= 0 and c > 0:
            term = 1 / (15 - c)  # 1, 1/2, 1/3 etc.
            num += (term * freqs[j])
            denom += term
            j -= 1
            c -= 1

        smoothed = num / denom
        smoothed_freqs.append(smoothed)

    # Reverse the result to match the original order
    return smoothed_freqs[::-1]

# # Apply smoothing to each column in dense_df
# smoothed = smooth_series(q1_series.iloc[:, 1])

# # Convert the smoothed list to a DataFrame
# smoothed_df = pd.DataFrame(smoothed, columns=['Smoothed'])

# # Save the DataFrame to a CSV file
# output_path = "../processed_data/q1_time_series_smoothed.csv"
# smoothed_df.to_csv(output_path, index=False, header=False)

q = pd.read_csv("../processed_data/q1_time_series.csv")
q.to_csv("../processed_data/q1_time_series.csv", index=False)
