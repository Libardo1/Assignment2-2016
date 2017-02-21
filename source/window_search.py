import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from q2_NER import NERModel, test_NER, Config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %matplotlib inline


WINDOWS = [3, 4, 5, 6]
results = []
times = []

for window in WINDOWS:
    config = Config(window_size=window)
    val_loss, duration = test_NER(config,
                                  save=False,
                                  verbose=False,
                                  debug=True)
    results.append(val_loss)
    times.append(duration)

best_result = min(list(zip(results, WINDOWS, times)))
result_string = """In an experiment with {0} windows sizes
the best size is {1} with val_loss = {2}. Using
this size the training will take {3} seconds""".format(len(WINDOWS),
                                                       best_result[1],
                                                       best_result[0],
                                                       best_result[2])
print(result_string)

# Make a plot of dropout vs loss
plt.plot(WINDOWS, results)
plt.xlabel("window")
plt.ylabel("loss")
plt.savefig("NER_window_loss.png")
# plt.show()

plt.plot(WINDOWS, times)
plt.xlabel("window")
plt.ylabel("time")
plt.savefig("NER_window_time.png")
