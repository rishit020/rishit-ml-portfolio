import numpy as np

np.random.seed(42)

data = np.random.randint(0, 100, size=(100, 6))

mean_values = np.sum(data) / data.size
sorted_data = np.sort(data, axis=None)
median_value = (sorted_data[299] + sorted_data[300]) / 2
mode = np.bincount(data.flatten()).argmax()

z_scores = (data - mean_values) / np.std(data)
min_max_scaled = (data - np.min(data)) / (np.max(data) - np.min(data))

labels = np.random.randint(0, 4, size=20)
num_classes = len(np.unique(labels))
one_hot = np.zeros((labels.size, num_classes), dtype=int)
one_hot[np.arange(labels.size), labels] = 1

X = np.random.randn(1000, 6)
means = X.mean(axis=0)
stds = X.std(axis=0)
X_std = (X - means) / stds
n = X_std.shape[0]
corr_matrix = (X_std.T @ X_std) / (n - 1)

