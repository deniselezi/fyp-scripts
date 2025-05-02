"""
This script provides a series of functions which were used to generate the
visualisations provided in the final report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def plot_ili_rates():
    start_2015 = 4262 - 1340 - 1
    end_2019 = 5722 - 1340 - 1

    ili = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

    ili_slice = ili.iloc[start_2015 : end_2019 + 1].reset_index(drop=True)

    dates = pd.date_range(start="2015-09-01", end="2019-08-31", freq="d")

    assert len(ili_slice) == len(
        dates
    ), f"Mismatch: {len(ili_slice)} values vs {len(dates)} dates"

    plt.figure(figsize=(12, 6))
    plt.plot(dates, ili_slice[0], label="ILI")
    plt.xlabel("Date")
    plt.ylabel("ILI Rate")
    plt.title("Influenza-like illness rates per 100,000 people")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_smoothing():
    """Plot the results of smoothing on the "flu medicine" query."""
    smooth = pd.read_csv("../processed_data/q1_time_series_smoothed.csv")
    not_smooth = pd.read_csv("../processed_data/q1_time_series.csv")

    start_idx = 1340 - 1

    end_idx = 5722 - 1
    smooth_slice = smooth.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    not_smooth_slice = not_smooth.iloc[start_idx : end_idx + 1].reset_index(drop=True)

    dates = pd.date_range(start="2007-09-01", end="2019-08-31", freq="d")

    assert len(smooth_slice) == len(
        dates
    ), f"Mismatch: {len(smooth_slice)} smooth values vs {len(dates)} dates"
    assert len(not_smooth_slice) == len(
        dates
    ), f"Mismatch: {len(not_smooth_slice)} not_smooth values vs {len(dates)} dates"

    plt.figure(figsize=(12, 6))
    plt.plot(dates, not_smooth_slice, label="Original", color="black", alpha=0.8)
    plt.plot(dates, smooth_slice, label="Smoothed", color="royalblue")
    plt.xlabel("Year")
    plt.ylabel("Query frequency")
    plt.title('"Flu Medicine" query - original vs smoothed')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_semantic_correlation_filtering_difference():
    """Identifies queries with a high Pearson correlation to ILI rates
    but missing from the top 1000 cosine similarity queries, and plots them in a bar chart.
    """

    cosine_sim = pd.read_csv("../processed_data/scores.csv", nrows=2000)
    pearson_corr = pd.read_csv("../processed_data/pearson_correlation_scores.csv")

    cosine_queries_top_1000 = set(cosine_sim.iloc[:1000, 2].values)

    queries = {}

    # iterate over the first 1000 queries in pearson_corr
    for i in range(1000):
        query = pearson_corr.iloc[i, 0]
        correlation = pearson_corr.iloc[i, 1]

        if query not in cosine_queries_top_1000:
            # if not present in the top 1000 cosine_sim queries, add it
            # try to find its cosine similarity score from anywhere in the 2000 loaded rows
            score_row = cosine_sim[cosine_sim.iloc[:, 2] == query]
            if not score_row.empty:
                score = score_row.iloc[0, 1]
            else:
                print(query)
                continue
            queries[query] = (score, correlation)

    # sort by correlation in descending order
    sorted_queries = sorted(queries.items(), key=lambda x: x[1][1], reverse=True)

    top_10_queries = sorted_queries[:10]
    print(top_10_queries)

    # plot
    query_labels = [item[0] for item in top_10_queries]
    scores = [item[1][0] for item in top_10_queries]
    correlations = [item[1][1] for item in top_10_queries]

    x = range(len(top_10_queries))
    plt.figure(figsize=(10, 6))
    bar_width = 0.35

    plt.bar(
        x, scores, width=bar_width, label="Cosine Similarity", color="black", alpha=0.9
    )
    plt.bar(
        [p + bar_width for p in x],
        correlations,
        width=bar_width,
        label="Pearson Correlation",
        color="royalblue",
        alpha=0.9,
    )

    plt.ylabel("Scores")
    plt.title("Queries with a high Pearson correlation but a lower cosine similarity")
    plt.xticks([p + bar_width / 2 for p in x], query_labels, rotation=45, ha="right")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_individual_queries(columns):
    """Extracts queries from a final data file and plots them alongside
    the ILI rates to visualise their correlation."""
    start_2015 = 4262 - 1340 - 1
    end_2019 = 5722 - 1340 - 1

    data = pd.read_csv("../processed_data/final_1000.csv")
    data_slice = data.iloc[start_2015 : end_2019 + 1]
    data_slice = data_slice[columns]

    # normalise
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(data_slice)
    data_slice = pd.DataFrame(normalized_array, columns=columns)

    # plot
    dates = pd.date_range(start="2015-09-01", end="2019-08-31", freq="d")

    n_plots = len(columns)
    _, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for idx, col in enumerate(columns):
        axes[idx].plot(dates, data_slice[col], label=col, color="royalblue", alpha=1)
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_yticks([])
        axes[idx].set_title(col)
        axes[idx].grid(False)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


def plot_overfitting_behaviour():
    epochs = np.arange(1, 201)  # 200 epochs

    # dummy training loss
    training_loss = np.exp(-epochs / 50) + np.random.normal(0, 0.01, size=epochs.shape)

    # dummy validation loss
    validation_loss = np.exp(-epochs / 55) + np.random.normal(
        0, 0.015, size=epochs.shape
    )
    validation_loss[120:] += (
        epochs[120:] - 120
    ) * 0.005  # starts rising after epoch 120

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_loss, label="Training Loss", color="black", linewidth=1.5)
    plt.plot(
        epochs,
        validation_loss,
        label="Validation Loss",
        color="royalblue",
        linewidth=1.5,
    )
    plt.axvline(x=120, color="gray", linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.xticks([])
    plt.yticks([])
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ili_single_season_threshold():
    start_2017 = 4993 - 1340 - 1
    end_2018 = 5357 - 1340 - 1

    ili = pd.read_csv("../processed_data/filtered_ili.csv", header=None)

    ili_slice = ili.iloc[start_2017 : end_2018 + 1].reset_index(drop=True)

    dates = pd.date_range(start="2017-09-01", end="2018-08-31", freq="d")

    assert len(ili_slice) == len(
        dates
    ), f"Mismatch: {len(ili_slice)} values vs {len(dates)} dates"

    plt.figure(figsize=(12, 6))
    plt.plot(dates, ili_slice[0], label="ILI")
    plt.axhline(y=15, color="olive", linestyle="--", label="Threshold")
    plt.xlabel("Date")
    plt.ylabel("ILI Rate")
    plt.title("Influenza-like illness rates per 100,000 people")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# plot_semantic_correlation_filtering_difference()
# plot_smoothing()
# plot_individual_queries(["flu treatment", "boots flu jabs", "best medicine for cough"])
# plot_overfitting_behaviour()
plot_ili_single_season_threshold()
