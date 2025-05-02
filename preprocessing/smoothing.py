import pandas as pd

data_path = "./processed_data/q_freq_dense.csv"
dense_df = pd.read_csv(data_path, header=None)

k = 1


def smooth_series(freqs):
    """Smooth a series using the provided algorithm."""
    global k
    if k % 20 == 0:
        print(k)
    k += 1

    smoothed_freqs = []

    for i in range(len(freqs) - 1, -1, -1):  # iterate over series of frequencies
        j = i
        c = 14

        # previous terms
        num = 0
        denom = 0
        while j >= 0 and c > 0:
            term = 1 / (15 - c)  # 1, 1/2, 1/3 etc.
            num += term * freqs[j]
            denom += term
            j -= 1
            c -= 1

        smoothed = num / denom
        smoothed_freqs.append(smoothed)

    # Reverse the result to match the original order
    return smoothed_freqs[::-1]


# Apply smoothing to each column in dense_df
smoothed_df = dense_df.apply(smooth_series, axis=0)

# Save the smoothed dataset to a CSV file
output_path = "./processed_data/q_freq_dense_smooth.csv"
smoothed_df.to_csv(output_path, index=False, header=False)
