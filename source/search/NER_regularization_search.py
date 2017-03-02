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
reg1 = np.random.random_sample([part1]) / 1000
reg2 = np.random.random_sample([part2]) / 10000
REGULARIZATION = np.concatenate((reg1, reg2))
REGULARIZATION.sort()
results = []
times = []

for regularizer in REGULARIZATION:
    config = Config(l2=regularizer)
    val_loss, duration = test_NER(config,
                                  save=False,
                                  verbose=False,
                                  debug=False,
                                  search=True)
    results.append(val_loss)
    times.append(duration)


REGULARIZATION = list(REGULARIZATION)
best_result = min(list(zip(results, REGULARIZATION, times)))
result_string = """In an experiment with {0} random constants
the best regularization constant is {1} with val_loss = {2}. Using
this constant the training will take {3} seconds""".format(number_of_exp,
                                                           best_result[1],
                                                           best_result[0],
                                                           best_result[2])
print(result_string)

# Make a plot of regularization vs loss
plt.plot(REGULARIZATION, results)
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("loss")
plt.savefig("NER_reg.png")
