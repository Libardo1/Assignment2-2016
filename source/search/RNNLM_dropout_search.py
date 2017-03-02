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
DROPOUT = np.random.random_sample([number_of_exp])
DROPOUT.sort()
results = []
times = []

for i, dropout in enumerate(DROPOUT):
    print("\n ({0} of {1})".format(i+1, number_of_exp))
    config = Config(dropout=dropout)
    val_pp, duration = test_RNNLM(config,
                                  save=False,
                                  debug=True,
                                  generate=False,
                                  search=True)
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

file = open("RNNLM_dropout.txt", "w")
file.write(result_string)
file.close()

# Make a plot of dropout vs loss
plt.plot(DROPOUT, results)
plt.xlabel("dropout")
plt.ylabel("perplexity")
plt.savefig("RNNLM_dropout.png")
plt.show()

# Sending an email with the results
if args.password != "None":
    cwd = os.getcwd()

    script_name = "RNNLM_dropout_search.py"

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

    filename1 = "RNNLM_dropout.txt"
    attachment1 = open(cwd + '/' + filename1, "rb")

    part1 = MIMEBase('application', 'octet-stream')
    part1.set_payload((attachment1).read())
    encoders.encode_base64(part1)
    part1.add_header('Content-Disposition',
                     "attachment; filename= %s" % filename1)

    msg.attach(part1)

    filename2 = "RNNLM_dropout.png"
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
