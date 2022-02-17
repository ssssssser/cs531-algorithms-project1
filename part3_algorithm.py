import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import copy
import timeit
highway = pd.read_csv('highway.csv')
from copy import deepcopy


# distance between two trajectoeis:
# cumpute the distance between two points

def distance(A, B):
    dist = np.square(A[0] - B[0]) + np.square(A[1] - B[1])
    return dist

def dtw2(t1, t2):
    """
    :param t1:trajectory 1, in the form of numpy array
    :param t2:trajectory 2, in the form of numpy array
    :return: distance between the two trajectories
    """
    # get the length of each trajectory
    x = t1.shape[0]
    y = t2.shape[0]
    # create a matrix to store xxx values
    dp = np.zeros((x, y))
    # base cases
    dp[0][0] = distance(t1[0], t2[0])
    for i in range(1, x):
        dp[i][0] = dp[i - 1][0] + distance(t1[i], t2[0])
    for j in range(1, y):
        dp[0][j] = dp[0][j - 1] + distance(t1[0], t2[j])
    # fill the DP table
    for i in range(1, x):
        for j in range(1, y):
            min_dis = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
            dp[i][j] = distance(t1[i], t2[j]) + min_dis
    dp_dc = deepcopy(dp[x - 1][y - 1])
    return dp_dc

# compute the dtw distance of a specific trajectory named "one_sample" to every trajectory in set "X"
def dtw_distance(one_sample, X, dist_func):
    distances = []  # a matrix to store distance between "one_sample" and every one of "X"
    for i in range(X.shape[0]):
        dist = dist_func(one_sample, X[i])
        distances.append(dist)
    distances_dc = deepcopy(distances)
    return distances_dc

# copy "center_traj_1" and "center_traj_2" from part2：
def center_traj_1(trajs, dist_fun):
    """
    trajs: set of trajectories, in the form of numpy array
    dist_fun: the function to compute the distance of two trajectories
    return: the center trajectory, distance from center to all trajs in set
    """
    if not isinstance(trajs, np.ndarray):
        raise Exception('Expected np.ndarray for first variable')
    # get the number of all trajectories
    n = trajs.shape[0]
    # use a 2D array to store the distance of every two trajs
    dist_matrix = np.zeros((n, n))
    # compute the distance of two trajs
    for i in np.arange(0, n):
        for j in np.arange(i, n):
            distance = dist_fun(trajs[i], trajs[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    # compute the distance from evey traj to all others
    dist_sum = dist_matrix.sum(axis=1)
    center_index = np.argmin(dist_sum)
    center_traj = trajs[center_index]
    min_distance_sum = np.min(dist_sum)
    # deepcopy
    center_traj_dc = deepcopy(center_traj)
    min_distance_sum_dc = deepcopy(min_distance_sum)
    return (center_traj_dc, min_distance_sum_dc)

def center_traj_2(trajs, h, dist_fun):
    """
    trajs: set of trajectories, in the form of numpy array
           one row for one trajectory
    h: number of segments
    dist_fun: the distance function used to compute distance
              from center to other trajs
    return: the center trajectory, distance from center to all trajs in set
    """
    if not isinstance(trajs, np.ndarray):
        raise Exception('Expected np.ndarray for first variable')
    # get the number of all trajectories
    n = trajs.shape[0]
    # compute average points
    avg_points_set = np.zeros((n, h + 1, 2))
    for i in range(n):
        traj = trajs[i]
        avg_points = get_avg_points(traj, h)
        avg_points_set[i] = avg_points

    # get the average of all avg_points
    center_traj = np.zeros((n, h + 1, 2))
    if len(avg_points_set) != 0:
        center_traj = avg_points_set.mean(axis=0)

    # get the distance form center to all other nodes
    dist_sum = 0
    for traj in trajs:
        dist_sum += dist_fun(center_traj, traj)

    center_traj_dc = deepcopy(center_traj)
    dist_sum_dc = deepcopy(dist_sum)
    return (center_traj_dc, dist_sum_dc)

def get_avg_points(traj, h):
    """
    This is a help function for center_traj_2
    traj: one trajectory in the form of array of points
    h: number of segments
    return: the points sampled from the trajectory
    """
    k = traj.shape[0]
    arc_lengths = np.zeros(k - 1)
    for i in range(k - 1):
        point1 = traj[i]
        point2 = traj[i + 1]
        arc_lengths[i] = np.sqrt(np.square(point1[0] - point2[0]) \
                                 + np.square(point1[0] - point2[0]))
    total_length = arc_lengths.sum(axis=0)
    # distance btw two avg points
    part = total_length / h
    # compute the avg points
    cur_traj_point_index = 0
    cur_traj_length = 0

    avg_points = np.zeros((h + 1, 2))
    for i in range(h):
        avg_point = np.zeros(2)
        cur_avg_point_length = i * part
        while cur_traj_length < cur_avg_point_length:
            cur_traj_length += arc_lengths[cur_traj_point_index]
            cur_traj_point_index += 1
        diff = cur_traj_length - cur_avg_point_length
        point1 = traj[cur_traj_point_index - 1]
        point2 = traj[cur_traj_point_index]
        cur_arc_length = arc_lengths[cur_traj_point_index - 1]
        avg_point[0] = point2[0] + (diff / cur_arc_length) * (point1[0] - point2[0])
        avg_point[1] = point2[1] + (diff / cur_arc_length) * (point1[1] - point2[1])
        avg_points[i] = avg_point
    # last avg point is the last point of traj
    avg_points[h] = traj[-1]

    avg_points_dc = deepcopy(avg_points)
    return avg_points_dc

###functions
#transfer origin data to numpy array form (keep only x and y coordinate)
def init_trajs(df):
    trajs = []
    ids = list(set(df['id']))
    for i in ids:
        traj = df.loc[df['id']==i]
        traj = traj[['x','y']].values
        trajs.append(traj)
    return np.array(trajs)

#computer center for each cluster using func1
def compute_center(trajs,n_clusters, membership, dist_func):
    """
    trajs: set of trajectories in the form of array
    n_clusters: number of clusters
    membership: the cluster that each trajectories belong to
    dist_func: function to calculate distance
    return: set of centers and corresponding costs for each cluster
    """
    centers = []
    costs = []
    for i in range(n_clusters):
        i_th_cluster = trajs[membership==i]
        i_th_center, cost = center_traj_1(i_th_cluster, dist_func)
        centers.append(i_th_center)
        costs.append(cost)
        #print('this is '+str(i)+'th center:')
        #print(i_th_center)
    return centers, costs

#computer center for each cluster using func2
def compute_center2(trajs,n_clusters, membership, dist_func,h):
    """
    trajs: set of trajectories in the form of array
    n_clusters: number of clusters
    membership: the cluster that each trajectories belong to
    dist_func: function to calculate distance
    return: set of centers and corresponding costs for each cluster
    """
    centers = []
    costs = []
    for i in range(n_clusters):
        i_th_cluster = trajs[membership==i]
        #if generate empty clustering
        if len(i_th_cluster) != 0:
            i_th_center, cost = center_traj_2(i_th_cluster,h, dist_func)
            centers.append(i_th_center)
            costs.append(cost)
        #print('this is '+str(i)+'th center:')
        #print(i_th_center)
    return centers, costs

#re-assign
def compute_membership(t, centers, dist_func):
    """
    t: one trajectory in the form of array of data points
    centers: set of centers of all clusters
    dist_func: function to calculate distance
    return: the membership each trajectories belong to
    """
    dist_to_centers = [dist_func(t, c) for c in centers]
    return np.argmin(dist_to_centers)

def assign(trajs, centers, dist_func):
    """
    trajs: set of trajectories in the form of array
    centers: set of centers of all clusters
    dist_func: function to calculate distance
    return: the membership each trajectories belong to
    """
    membership = np.zeros(len(trajs))
    for i in range(len(trajs)):
        membership[i] = compute_membership(trajs[i], centers, dist_func)
    return membership

#get one trajectory from each lane, for testing
def get_center_lane(highway):
    centers_temp = []
    for i in range(1,9):
        trajs_i = highway[highway['lane']==i]
        ids = list(set(trajs_i['id']))
        trajs_i2 = trajs_i[trajs_i['id']==ids[0]]
        trajs_i2 = trajs_i2[['x','y']].values

        centers_temp.append(trajs_i2)
    return centers_temp

#Plot the trajectories and color them by the cluster assigned by the procedure:
## 8 colors for 8 clusters
def plot_trajectoires(trajs,membership,centers,name):
    """
    trajs: one trajectory in the form of array of points
    membership: each trajectories belong to
    centers: two dimension points
    name: parameter for saving figure
    """
    colors = ['darkred', 'black', 'cyan', 'violet', 'darkorange', 'gold', 'green', 'b']

    plt.figure(figsize=(10,10))  # enlarge the graph
    for i in range(len(centers)):
        i_th_cluster = trajs[membership==i]
        for traj_i in i_th_cluster:
            x = traj_i[:,0]
            y = traj_i[:,1]
            plt.plot(x, y,color = colors[i], alpha = 0.1)
    plt.savefig(name)

###cluster trajectory using center trajectory method
def clustering1(trajs,n_clusters,max_iter, dist_func):
    """
    trajs: set of trajectories in the form of array
    n_clutsers: number of clusters
    max_iter: maximum time of iteration rounds
    dist_func: function to calculate distance
    """
    #Random membership
    membership = np.random.randint(n_clusters, size=len(trajs))
    #Main Loop
    num_iter = 0
    while num_iter < max_iter :
        #center compution
        centers, costs = compute_center(trajs, n_clusters, membership, dist_func)
        print("Total costs at iteration {}: {}".format(num_iter, sum(costs)))
        #re-assignment
        membership_new = assign(trajs, centers, dist_func)
        print("Membership done!")
        total_cost = sum(costs)
        if (membership==membership_new).all()==True:
            break
        else:
            membership = membership_new
            num_iter += 1
    return membership, centers, total_cost
# function to run r times for method1
def run_r_times_cluster1(r, trajs, n_clusters, max_iter, dist_func):
    d1 = []
    count = 0
    while count < r:
        membership_c, centers_c, total_cost = clustering1(trajs,n_clusters, max_iter, dist_func)
        d1.append([membership_c, centers_c, total_cost])
        count = count + 1
    d2 = np.array(d1)
    cost_min_idx = np.argmin(d2[:, 2])
    #return the clustering with minimum sum cost
    return d2[cost_min_idx][0], d2[cost_min_idx][1], d2[cost_min_idx][2]

#use center method 2
def clustering2(trajs,n_clusters,max_iter, dist_func,h):
    """
    trajs: set of trajectories in the form of array
    n_clutsers: number of clusters
    max_iter: maximum time of iteration rounds
    dist_func: function to calculate distance
    h: h-average,parameter in part2 func2
    """
    #Random membership
    membership = np.random.randint(n_clusters, size=len(trajs))
    # Main Loop
    num_iter = 0
    while num_iter < max_iter :
        #center compution
        centers, costs = compute_center2(trajs, n_clusters, membership, dist_func,h)
        print("Total costs at iteration {}: {}".format(num_iter, sum(costs)))
        #re-assignment
        membership_new = assign(trajs, centers, dist_func)
        print("Membership done!")
        total_cost = sum(costs)
        if (membership==membership_new).all()==True:
            break
        else:
            membership = membership_new
            num_iter += 1
    return membership, centers, total_cost

#function to run r times of method 2:
def run_r_times_cluster2(r,trajs,n_clusters,max_iter,dist_func,h):
    d1 = []
    count = 0
    while count < r :
        membership_c,centers_c,total_cost = clustering2(trajs,n_clusters,max_iter, dist_func, h)
        d1.append([membership_c,centers_c,total_cost])
        count +=1
    d2 = np.array(d1)
    cost_min_idx = np.argmin(d2[:,2])
    #return the clustering with minimum sum cost
    return d2[cost_min_idx][0], d2[cost_min_idx][1],d2[cost_min_idx][2]

