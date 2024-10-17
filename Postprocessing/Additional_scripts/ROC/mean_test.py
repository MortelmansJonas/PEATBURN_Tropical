import numpy as np

array = np.random.randint(0, 100, size=(10))
mean = np.mean(array, axis=0)
print(mean)