import numpy as np

data = np.random.randn(10000, 20)

indices = np.random.permutation(len(data))
shuffled_data = data[indices]

def batch_generator(data, batch_size, batch_index):
    start = batch_index * batch_size
    end = start + batch_size
    return data[start:end]

batch_size = 64

num_batches = int(np.ceil(len(shuffled_data) / batch_size))

for batch_index in range(num_batches):
    batch = batch_generator(shuffled_data, batch_size, batch_index)

    batch_expanded = np.expand_dims(batch, axis=-1)

    batch_squeezed = np.squeeze(batch_expanded)

    print(f"Batch {batch_index+1}/{num_batches}")
    print("Original shape:", batch.shape)
    print("Expanded shape:", batch_expanded.shape)
    print("Squeezed shape:", batch_squeezed.shape)
    print("-" * 40)
