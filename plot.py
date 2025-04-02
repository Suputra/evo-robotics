#! /Users/saahas/morpho-rl/.venv/bin/python
import matplotlib.pyplot as plt
import numpy as np

data = np.load("data/rightleg.npy")
plt.plot(data)
plt.show()
