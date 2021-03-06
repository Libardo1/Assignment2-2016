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
DROPOUT = np.random.random_sample([number_of_exp])
DROPOUT.sort()
results = []
times = []

for dropout in DROPOUT:
    config = Config(dropout=dropout)
    val_loss, duration = test_NER(config,
                                  save=False,
                                  verbose=False,
                                  debug=False,
                                  search=True)
    results.append(val_loss)
    times.append(duration)


DROPOUT = list(DROPOUT)
best_result = min(list(zip(results, DROPOUT, times)))
result_string = """In an experiment with {0} random constants
the best dropout constant is {1} with val_loss = {2}. Using
this constant the training will take {3} seconds""".format(number_of_exp,
                                                           best_result[1],
                                                           best_result[0],
                                                           best_result[2])
print(result_string)

# Make a plot of dropout vs loss
plt.plot(DROPOUT, results)
plt.xlabel("dropout")
plt.ylabel("loss")
plt.savefig("NER_dropout.png")
