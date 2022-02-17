import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
from part1_algorithms import dtw
from scipy.spatial.distance import cdist
from part2_algorithm import center_traj_1, center_traj_2

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
# get list of datasets coming from lane1, lane3,4, lane4,5
lanes = [1, 3, 4, 5]
data_lanes = []
for i in lanes:
    mask_1 = data[:,3] == i
    data_1 = data[mask_1,:]
    data_1 = data_1[:, 0:3] # remove last column
    trajs_1 = split_trajectory(data_1)
    data_lanes.append(trajs_1)

data_test_1 = data_lanes[0]
data_test_2 = np.concatenate((data_lanes[1], data_lanes[2]), axis = 0)
data_test_3 = np.concatenate((data_lanes[2], data_lanes[3]), axis = 0)

data_test = [data_test_1, data_test_2, data_test_3]


## compute center traj of 3 test datasets
# using two center computing functions
centers_fun1 = []
centers_fun2 = []
runtime_fun1 = []
runtime_fun2 = []
cost_fun1 = []
cost_fun2 = []
for trajs in data_test:
    start = timeit.default_timer()
    (center_fun1, dist) = center_traj_1(trajs, dtw)
    end = timeit.default_timer()
    runtime_fun1.append(end - start)
    centers_fun1.append(center_fun1)
    cost_fun1.append(dist)

    start = timeit.default_timer()
    (center_fun2, dist) = center_traj_2(trajs, 16, dtw)
    end = timeit.default_timer()
    runtime_fun2.append(end - start)
    centers_fun2.append(center_fun2)
    cost_fun2.append(dist)

## draw plot of cost
x_labels = ["lane1", "lane3,4", "lane4,5"]
plt.figure(figsize=(6.5, 4))

# x range
x = np.arange(3)
# draw bar plt
plt.bar(x + 0.00, cost_fun1, color='#81b214', width=0.3, label="Function 1")
plt.bar(x + 0.30, cost_fun2, color='#ffcc29', width=0.3, label="Function 2")

# title
plt.title('Comapring Cost (Distance Sum) of Two Functions', fontsize="12")
plt.ylabel('Cost', fontsize="12")
plt.xlabel('Dataset',fontsize="12")
# ticks
plt.xticks(x + 0.15, x_labels, fontsize='12')
# generate plt and save
plt.legend(loc="upper left")
plt.savefig("cost_comparision", dpi=1000)
plt.show()

## draw plot of runtime
x_labels = ["lane1", "lane3,4", "lane4,5"]
plt.figure(figsize=(6.5, 4))

# x range
x = np.arange(3)
# draw bar plt
plt.bar(x + 0.00, runtime_fun1, color='#81b214', width=0.3, label="Function 1")
plt.bar(x + 0.30, runtime_fun2, color='#ffcc29', width=0.3, label="Function 2")

# title
plt.title('Comapring Runtime of Two Functions', fontsize="12")
plt.ylabel('Runtime', fontsize="12")
plt.xlabel('Dataset',fontsize="12")
# ticks
plt.xticks(x + 0.15, x_labels, fontsize='12')
# generate plt and save
plt.legend(loc="upper left")
plt.savefig("runtime_comparision", dpi=1000)
plt.show()


## visualize center for lane 1
x = centers_fun1[0][:, 0]
y = centers_fun1[0][:, 1]
plt.plot(x, y, color='red', label="Center of Function 1")

x = centers_fun2[0][:, 0]
y = centers_fun2[0][:, 1]
plt.plot(x, y, color='black', label="Center of Function 2")

for traj in data_test[0]:
    x = traj[:, 0]
    y = traj[:, 1]
    plt.plot(x, y, color='blue', alpha=0.04)
plt.legend(loc="upper left")
plt.savefig("visual_lane_1", dpi=1000)
plt.show()


## visualize center for lane 3,4
x = centers_fun1[1][:,0]
y = centers_fun1[1][:,1]
plt.plot(x, y, color = 'red', label="Center of Function 1")

x = centers_fun2[1][:,0]
y = centers_fun2[1][:,1]
plt.plot(x, y, color = 'black', label="Center of Function 2")

for traj in data_test[1]:
    x = traj[:,0]
    y = traj[:,1]
    plt.plot(x, y, color = 'blue', alpha=0.04)
plt.legend(loc="upper left")
plt.savefig("visual_lane_34", dpi=1000)
plt.show()


## visualize center for lane 4,5
x = centers_fun1[2][:,0]
y = centers_fun1[2][:,1]
plt.plot(x, y, color = 'red', label="Center of Function 1")

x = centers_fun2[2][:,0]
y = centers_fun2[2][:,1]
plt.plot(x, y, color = 'black', label="Center of Function 2")

for traj in data_test[2]:
    x = traj[:,0]
    y = traj[:,1]
    plt.plot(x, y, color = 'blue', alpha=0.04)
plt.legend(loc="upper left")
plt.savefig("visual_lane_45", dpi=1000)
plt.show()
