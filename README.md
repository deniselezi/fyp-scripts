# fyp-scripts

Scripts related to my final year project at University College London (UCL).

The `preprocessing/` directory contains the scripts used to clean the data and perform feature selection. They were applied in the following order:
1. `dense_matrix.py`, to generate a dataset from the sparse representation of the raw data
2. `smoothing.py`, to smooth the frequencies
3. `interpolate.py` to generate the daily ILI rates and `filter_dates.py` to align the frequencies and ILI rates to be between 2007 and 2019
4. `density_filtering.py`, to remove frequency series with only zeroes
5. `sbert.py` and `semantic_filtering.py` to select the top 2000 queries by cosine similarity with the reference embeddings
6. `ili_rates_filtering.py` to select the top 1000 queries by Pearson correlation with the ILI rates

Other directories contain modelling-related code for the three models. Each directory contains the model's class, and an additional class to facilitate running the model.
