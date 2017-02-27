import os
import argparse
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

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

STEP = list(range(1, 21))
number_of_exp = len(STEP)
results = []
times = []

for i, step in enumerate(STEP):
    print("\n ({0} of {1})".format(i+1, number_of_exp))
    config = Config(num_steps=step)
    val_pp, duration = test_RNNLM(config,
                                  save=False,
                                  debug=False)
    results.append(val_pp)
    times.append(duration)

best_result = min(list(zip(results, STEP, times)))
result_string = """In an experiment with {0} number of steps
the best num_step is {1} with validation perplexity = {2}. Using
this number the training will take {3} seconds""".format(len(STEP),
                                                       best_result[1],
                                                       best_result[0],
                                                       best_result[2])


file = open("RNNLM_step.txt", "w")
file.write(result_string)
file.close()

# Make a plot of step vs perplexity
plt.plot(STEP, results)
plt.xlabel("step")
plt.ylabel("perplexity")
plt.savefig("RNNLM_step_pp.png")


# Make a plot of step vs time
plt.plot(STEP, times)
plt.xlabel("step")
plt.ylabel("time")
plt.savefig("RNNLM_step_time.png")

# Sending an email with the results
if args.password != "None":
    cwd = os.getcwd()

    script_name = "RNNLM_step_search.py"

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

    filename1 = "RNNLM_step.txt"
    attachment1 = open(cwd + '/' + filename1, "rb")

    part1 = MIMEBase('application', 'octet-stream')
    part1.set_payload((attachment1).read())
    encoders.encode_base64(part1)
    part1.add_header('Content-Disposition',
                     "attachment; filename= %s" % filename1)

    msg.attach(part1)

    filename2 = "RNNLM_step_pp.png"
    attachment2 = open(cwd + '/' + filename2, "rb")

    part2 = MIMEBase('application', 'octet-stream')
    part2.set_payload((attachment2).read())
    encoders.encode_base64(part2)
    part2.add_header('Content-Disposition',
                     "attachment; filename= %s" % filename2)

    msg.attach(part2)

    filename3 = "RNNLM_step_time.png"
    attachment3 = open(cwd + '/' + filename3, "rb")

    part3 = MIMEBase('application', 'octet-stream')
    part3.set_payload((attachment3).read())
    encoders.encode_base64(part3)
    part3.add_header('Content-Disposition',
                     "attachment; filename= %s" % filename3)

    msg.attach(part3)

    password = args.password

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, password)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
