# The code was adopted from https://www.alexkras.com/how-to-restart-python-script-after-exception-and-run-it-forever/
from subprocess import Popen
import sys
 
filename = sys.argv[1]
while True:
    print("\nStarting " + filename)
    p = Popen("python " + filename, shell=True)
    p.wait()