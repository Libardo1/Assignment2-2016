import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from q2_NER import test_NER, Config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %matplotlib inline


BATCH = [14, 24, 34, 44, 54, 64, 74, 84, 94, 104]
results = []
times = []

for batch in BATCH:
    config = Config(batch_size=batch)
    val_loss, duration = test_NER(config,
                                  save=False,
                                  verbose=False,
                                  debug=False)
    results.append(val_loss)
    times.append(duration)

best_result = min(list(zip(results, BATCH, times)))
result_string = """In an experiment with {0} batch sizes
the best size is {1} with val_loss = {2}. Using
this size the training will take {3} seconds""".format(len(BATCH),
                                                       best_result[1],
                                                       best_result[0],
                                                       best_result[2])
print(result_string)

# Make a plot of batch vs loss
plt.plot(BATCH, results)
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig("NER_batch_loss.png")
# plt.show()

# Make a plot of batch vs time
plt.plot(BATCH, times)
plt.xlabel("batch")
plt.ylabel("time")
plt.savefig("NER_batch_time.png")
