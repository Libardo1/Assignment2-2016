import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

import numpy as np
from q3_RNNLM import test_RNNLM, Config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-p",
                    "--password",
                    type=str,
                    default='None',
                    help="""Password for the robotanara mail.(default=None)""")

args = parser.parse_args()
number_of_exp = 10
part1 = number_of_exp/2
part2 = number_of_exp - part1
lr1 = np.random.random_sample([part1]) / 100
lr2 = np.random.random_sample([part2]) / 100
LEARNING_RATE = np.concatenate((lr1, lr2))
LEARNING_RATE.sort()
results = []
times = []

for i, lr in enumerate(LEARNING_RATE):
    print("\n ({0} of {1})".format(i+1, number_of_exp))
    config = Config(lr=lr)
    val_pp, duration = test_RNNLM(config,
                                  save=False,
                                  debug=False,
                                  generate=False,
                                  search=True)
    results.append(val_pp)
    times.append(duration)


LEARNING_RATE = list(LEARNING_RATE)
best_result = min(list(zip(results, LEARNING_RATE, times)))
result_string = """In an experiment with {0} random constants
the best learning rate constant is {1} with validation perplexity = {2}. Using
this constant the training will take {3} seconds""".format(number_of_exp,
                                                           best_result[1],
                                                           best_result[0],
                                                           best_result[2])

file = open("RNNLM_lr.txt", "w")
file.write(result_string)
file.close()


# Make a plot of lr vs perplexity
plt.plot(LEARNING_RATE, results)
plt.xscale('log')
plt.xlabel("learning rate")
plt.ylabel("perplexity")
plt.savefig("RNNLM_lr.png")


# Sending an email with the results
if args.password != "None":
    cwd = os.getcwd()

    script_name = "RNNLM_lr_search.py"

    fromaddr = "robotanara@gmail.com"
    toaddr = "felipessalvador@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "End of {}".format(script_name)

    body = """Dear Felipe,
    the script {} is done.
    A review of the process can be seen in the following attachments.

    Best,
    Nara""".format(script_name)
    msg.attach(MIMEText(body, 'plain'))

    filename1 = "RNNLM_lr.txt"
    attachment1 = open(cwd + '/' + filename1, "rb")

    part1 = MIMEBase('application', 'octet-stream')
    part1.set_payload((attachment1).read())
    encoders.encode_base64(part1)
    part1.add_header('Content-Disposition',
                     "attachment; filename= %s" % filename1)

    msg.attach(part1)

    filename2 = "RNNLM_lr.png"
    attachment2 = open(cwd + '/' + filename2, "rb")

    part2 = MIMEBase('application', 'octet-stream')
    part2.set_payload((attachment2).read())
    encoders.encode_base64(part2)
    part2.add_header('Content-Disposition',
                     "attachment; filename= %s" % filename2)

    msg.attach(part2)

    password = args.password

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, password)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
