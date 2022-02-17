import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
from part1_algorithms import dtw
from scipy.spatial.distance import cdist
from extension2_algorithm import center_traj_1_rd, center_traj_2_rd

def split_trajectory(data):
    """
    split the data of points into the form of array of trajectories.
    """
    cur_traj_index = data[0,0]
    trajs = []
    traj = []
    for row in data:
        if row[0] == cur_traj_index:
            traj.append(row[1:3])
        else:
            trajs.append(np.array(traj))
            cur_traj_index = row[0]
            traj = []
            traj.append(row[1:3])
    trajs = np.array(trajs)
    return trajs

## prepare data
# read data
data = np.genfromtxt('highway.csv', delimiter=',')
# remove first line
data = np.delete(data, 0, axis=0)
# get list of datasets coming from lane1
mask_1 = data[:,3] == 1
data_1 = data[mask_1,:]
data_1 = data_1[:, 0:3] # remove last column
test_data = split_trajectory(data_1)

# compute center traj with different sample size on lane1
centers_fun1 = []
runtime_fun1 = []
cost_fun1 = []

for size in np.arange(0.1, 1.1, 0.1):
    (center_fun1, dist, runtime) = center_traj_1_rd(test_data, dtw, size)
    centers_fun1.append(center_fun1)
    cost_fun1.append(dist)
    runtime_fun1.append(runtime)

centers_fun2 = []
runtime_fun2 = []
cost_fun2 = []

for size in np.arange(0.1, 1.1, 0.1):
    (center_fun2, dist, runtime) = center_traj_2_rd(test_data, 16, dtw, size)
    centers_fun2.append(center_fun1)
    cost_fun2.append(dist)
    runtime_fun2.append(runtime)

## draw plot
plt.title('Runtime with Different Sample Size (Function 1)', fontsize="12")
plt.ylabel('Runtime', fontsize="12")
plt.xlabel('Sample Proportion',fontsize="12")
x = np.arange(0.1, 1.1, 0.1)
plt.plot(x, runtime_fun1)
plt.savefig("sample_runtime_fun1", dpi=1000)
plt.show()

## draw plot
plt.title('Cost with Different Sample Size (Function 1)', fontsize="12")
plt.ylabel('Cost', fontsize="12")
plt.xlabel('Sample Proportion',fontsize="12")
x = np.arange(0.1, 1.1, 0.1)
plt.plot(x, cost_fun1)
plt.savefig("sample_cost_fun1", dpi=1000)
plt.show()

## draw plot
plt.title('Runtime with Different Sample Size (Function 2)', fontsize="12")
plt.ylabel('Runtime', fontsize="12")
plt.xlabel('Sample Proportion',fontsize="12")
x = np.arange(0.1, 1.1, 0.1)
plt.plot(x, runtime_fun2)
plt.savefig("sample_runtime_fun2", dpi=1000)
plt.show()

## draw plot
plt.title('Cost with Different Sample Size (Function 2)', fontsize="12")
plt.ylabel('Cost', fontsize="12")
plt.xlabel('Sample Proportion',fontsize="12")
x = np.arange(0.1, 1.1, 0.1)
plt.plot(x, cost_fun2)
plt.savefig("sample_cost_fun2", dpi=1000)
plt.show()
