import os
from tqdm import *
import subprocess
import random

num_runs = 50
x_cord2 = random.sample(range(1050, 3000), num_runs)
x_cord = []
y_cord = random.sample(range(50,200), num_runs)

for x in x_cord2:
    x_cord.append(random.randint(700,x-50))

x_cord = [round(x / 1000, 2) for x in x_cord]
x_cord2 = [round(x2 / 1000, 2) for x2 in x_cord2]
y_cord = [round(y / 1000, 2) for y in y_cord]

start=100
for i in tqdm(range(num_runs)):
    with open(os.devnull, 'w') as devnull:
        # Remove any previous simulation file
        cmd = "rm -rf simulation_data/" + str(i+start)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Copy the OpenFOAM forwardStep directory
        cmd = "cp -a ./original/. ./simulation_data/" + str(i+start)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Remove the blockMeshDict file from system directory
        cmd = "rm -f ./simulation_data/" + str(i+start) + "/system/blockMeshDict"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Execute python program to write a blockMeshDict file
        cmd = "python gen_blockMeshDict.py" + " " + str(x_cord[i]) + " " + str(x_cord2[i]) + " " + str(y_cord[i])
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the blockMeshDict file to system directory
        cmd = "mv blockMeshDict ./simulation_data/" + str(i+start) + "/system"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the cellInformation file to home directory
        cmd = "mv cellInformation ./simulation_data/" + str(i+start)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)
