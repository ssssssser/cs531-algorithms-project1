import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
from part1_algorithms import dtw
from scipy.spatial.distance import cdist

def center_traj_1(trajs, dist_fun):
    """
    trajs: set of trajectories, in the form of numpy array
    dist_fun: the function to compute the distance of two trajectories
    return: the center trajectory, distance from center to all trajs in set
    """
    if not isinstance(trajs,np.ndarray):
        raise Exception('Expected np.ndarray for first variable')
    # get the number of all trajectories
    n = trajs.shape[0]
    # use a 2D array to store the distance of every two trajs
    dist_matrix = np.zeros((n, n))
    # compute the distance of two trajs
    for i in np.arange(0, n):
        for j in np.arange(i, n):
            distance = dist_fun(trajs[i], trajs[j])
            dist_matrix[i,j] = distance
            dist_matrix[j,i] = distance
    # compute the distance from evey traj to all others
    dist_sum = dist_matrix.sum(axis = 1)
    center_index = np.argmin(dist_sum)
    center_traj = trajs[center_index]
    min_distance_sum = np.min(dist_sum)
    return (center_traj, min_distance_sum)

def center_traj_2(trajs, h, dist_fun):
    """
    trajs: set of trajectories, in the form of numpy array
           one row for one trajectory
    h: number of segments
    dist_fun: the distance function used to compute distance
              from center to other trajs
    return: the center trajectory, distance from center to all trajs in set
    """
    if not isinstance(trajs,np.ndarray):
        raise Exception('Expected np.ndarray for first variable')
    # get the number of all trajectories
    n = trajs.shape[0]
    # compute average points
    avg_points_set = np.zeros((n, h+1,2))
    for i in range(n):
        traj = trajs[i]
        avg_points = get_avg_points(traj, h)
        avg_points_set[i] = avg_points
    # get the average of all avg_points
    center_traj = avg_points_set.mean(axis = 0)
    # get the distance form center to all other nodes
    dist_sum = 0
    for traj in trajs:
        dist_sum += dist_fun(center_traj, traj)

    return (center_traj, dist_sum)


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
    return avg_points