#! /usr/bin/python3
# python cleaner.py targetname

import sys
import subprocess

# Execute the shell command
# Replace with your desired command
# command = "ps -ax -o pid,args | grep -e \"python -m unittest\" -e test_  -e udiscovery | grep -v grep "
if len(sys.argv) > 1:
    command = f'ps -ax -o pid,args | grep -e {sys.argv[1]} | grep -v grep '
    output = subprocess.check_output(command, shell=True, encoding="utf-8")
    lines = output.splitlines()
    # pids = ['-'+str(line.split()[0]) for line in lines]
    pids = [str(line.split()[0]) for line in lines]
    subprocess.check_call("kill -9 " +" ".join(pids), shell=True)
else:
    pass
