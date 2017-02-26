import numpy as np
from q3_RNNLM import test_RNNLM, Config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %matplotlib inline

number_of_exp = 10
DROPOUT = np.random.random_sample([number_of_exp])
DROPOUT.sort()
results = []
times = []

for i, dropout in enumerate(DROPOUT):
    print("\n ({0} of {1})".format(i, number_of_exp))
    config = Config(dropout=dropout)
    val_pp, duration = test_RNNLM(config,
                                  save=False,
                                  debug=True)
    results.append(val_pp)
    times.append(duration)


DROPOUT = list(DROPOUT)
best_result = min(list(zip(results, DROPOUT, times)))
result_string = """In an experiment with {0} random constants
the best dropout constant is {1} with validation perplexity = {2}. Using
this constant the training will take {3} seconds""".format(number_of_exp,
                                                           best_result[1],
                                                           best_result[0],
                                                           best_result[2])
print(result_string)

# Make a plot of dropout vs loss
plt.plot(DROPOUT, results)
plt.xlabel("dropout")
plt.ylabel("perplexity")
plt.savefig("RNNLM_dropout.png")
plt.show()
