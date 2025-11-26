import numpy as np

np.random.seed(42)

data = np.random.randn(1000, 6)

mask = np.random.rand(1000, 6) < 0.1
data[mask] = np.nan

col_means = np.nanmean(data, axis=0)
idx = np.where(np.isnan(data))
data[idx] = np.take(col_means, idx[1])

data = (data - data.mean(axis=0)) / data.std(axis=0)

filtered_data = data[data[:, 2] > data[:, 2].mean()]

np.random.shuffle(filtered_data)

split = int(0.8 * len(filtered_data))
train = filtered_data[:split]
val = filtered_data[split:]

