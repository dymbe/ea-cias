#!/usr/bin/env python
import subprocess
import os
 

n=50
vtr = 0.39820

import os
os.chdir("/home/iwan/EA/Project/RV-GOMEA/RV-GOMEA/ea-cias/RV-GOMEA/")

pops = [8, 16, 32, 64, 128, 256, 512]
# pops = [64, 128, 256, 512]
# pops = [1024]
# pops = [8, 16, 32, 64]
tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
fscores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for j in range(len(pops)):
    
    esum = 0
    fsum = 0
    
    for i in range(n):
        rc = subprocess.call(["./script.sh", str(pops[j])])

        with open('./statistics.dat', 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()

        
        split = last_line.strip().split()
        esum += float(split[1])
        
        fsum += float(split[3])
    
    fscores[j] = fsum/n

print("Average evals and fitness:")

for k in range(len(pops)):
    print("Population: " + str(pops[k]))
    print(str(fscores[k]).replace('.',','))
    print("Difference from vtr:")
    print(vtr + fscores[k])
    print("===============")