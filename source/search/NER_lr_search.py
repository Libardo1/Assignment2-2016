import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
from q2_NER import test_NER, Config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %matplotlib inline

number_of_exp = 10
part1 = number_of_exp/2
part2 = number_of_exp - part1
lr1 = np.random.random_sample([part1]) / 100
lr2 = np.random.random_sample([part2]) / 100
LEARNING_RATE = np.concatenate((lr1, lr2))
LEARNING_RATE.sort()
results = []
times = []

for lr in LEARNING_RATE:
    config = Config(lr=lr)
    val_loss, duration = test_NER(config,
                                  save=False,
                                  verbose=False,
                                  debug=False,
                                  search=True)
    results.append(val_loss)
    times.append(duration)


LEARNING_RATE = list(LEARNING_RATE)
best_result = min(list(zip(results, LEARNING_RATE, times)))
result_string = """In an experiment with {0} random constants
the best learning rate constant is {1} with val_loss = {2}. Using
this constant the training will take {3} seconds""".format(number_of_exp,
                                                           best_result[1],
                                                           best_result[0],
                                                           best_result[2])
print(result_string)

# Make a plot of lr vs loss
plt.plot(LEARNING_RATE, results)
plt.xscale('log')
plt.xlabel("learning rate")
plt.ylabel("loss")
plt.savefig("NER_lr.png")
