#!/usr/bin/env python
import subprocess
import os

esum = 0
fsum = 0
n=20

import os
os.chdir("/home/iwan/EA/Project/RV-GOMEA/RV-GOMEA/RV-GOMEA/")


for i in range(n):
    rc = subprocess.call("./script.sh")

    with open('./statistics.dat', 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()

    
    split = last_line.strip().split()
    esum += float(split[1])
    print("Evaluation sum: ") 
    print(esum)
    
    fsum += float(split[3])
    print("Fitness sum: ")
    print(fsum)

print("Average evals and fitness:")
print(esum/n)
print(fsum/n)