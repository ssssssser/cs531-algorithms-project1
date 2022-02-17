import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import copy
import timeit

###########part1
#compute square of distance between two points
def distance(A,B):
    dist = np.square(A[0]-B[0])+np.square(A[1]-B[1])
    return dist

#3.1.1 implement dtw at O(T1*T2) time
def dtw(t1,t2):
    """
    :param t1:trajectory 1, in the form of numpy array
    :param t2:trajectory 2, in the form of numpy array
    :return: distance between the two trajectories
    """
    #get the length of each trajectory
    x = t1.shape[0]
    y = t2.shape[0]
    #create a matrix to store xxx values
    dp = np.zeros((x, y))
    #base cases
    dp[0][0] = distance(t1[0],t2[0])
    for i in range(1,x):
        dp[i][0] = dp[i-1][0]+distance(t1[i],t2[0])
    for j in range(1,y):
        dp[0][j] = dp[0][j-1]+distance(t1[0],t2[j])

    #fill the DP table
    for i in range(1,x):
        for j in range(1,y):
            min_dis = min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])
            dp[i][j] = distance(t1[i],t2[j]) + min_dis
    return dp[x-1][y-1]

#3.1.2 implement dfd at O(T1*T2) time
def dfd(t1,t2):
    """
    :param t1:trajectory 1, in the form of numpy array
    :param t2:trajectory 2, in the form of numpy array
    :return: distance between the two trajectories
    """
    #get the length of each trajectory
    x = t1.shape[0]
    y = t2.shape[0]
    #create a matrix to store xxx values
    dp = np.zeros((x, y))
    #base cases
    dp[0][0] = distance(t1[0],t2[0])
    for i in range(1,x):
        dp[i][0] = max(dp[i-1][0],distance(t1[i],t2[0]))
    for j in range(1,y):
        dp[0][j] = max(dp[0][j-1],distance(t1[0],t2[j]))
    #fill the DP table
    for i in range(1,x):
        for j in range(1,y):
            min_dis = min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])
            dp[i][j] = max(distance(t1[i],t2[j]),min_dis)
    return dp[x-1][y-1]

#3.1.3 modify to return minimum cost and monotone assignment
#modify dtw
def dtw_assign(t1,t2):
    """
    :param t1:trajectory 1, in the form of numpy array
    :param t2:trajectory 2, in the form of numpy array
    :return: distance between the two trajectories
    :return: the monotone assignment with the minimum cost, in the form of lists
    """
    # get the length of each trajectory
    x = t1.shape[0]
    y = t2.shape[0]
    # create a matrix to store
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
    ###find the monotne assignment, store pairs in list
    #start from the last pair
    m = x-1
    n = y-1
    C = [[m,n]]
    while m >0 or n>0:
        if m == 0:
            pred = [0,n-1]
            n = n-1
        elif n == 0:
            pred = [m-1,0]
            m = m-1
        else:
            pred_dis = min(dp[m-1][n-1], dp[m][n-1], dp[m-1][n])
            if pred_dis == dp[m-1][n-1]:
                pred = [m-1,n-1]
                m = m-1
                n = n-1
            elif pred_dis == dp[m][n-1]:
                pred = [m,n-1]
                n = n-1
            else:
                pred = [m-1,n]
                m = m-1
        C.append(pred)
    C.reverse()
    return dp[x-1][y-1],C
#modify dfd
def dfd_assign(t1,t2):
    """
    :param t1:trajectory 1, in the form of numpy array
    :param t2:trajectory 2, in the form of numpy array
    :return: distance between the two trajectories
    :return: the monotone assignment with the minimum cost, in the form of lists
    """
    #get the length of each trajectory
    x = t1.shape[0]
    y = t2.shape[0]
    #create a matrix to store xxx values
    dp = np.zeros((x, y))
    #base cases
    dp[0][0] = distance(t1[0],t2[0])
    for i in range(1,x):
        dp[i][0] = max(dp[i-1][0],distance(t1[i],t2[0]))
    for j in range(1,y):
        dp[0][j] = max(dp[0][j-1],distance(t1[0],t2[j]))
    #fill the DP table
    for i in range(1,x):
        for j in range(1,y):
            min_dis = min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])
            dp[i][j] = max(distance(t1[i],t2[j]),min_dis)
    ###find the monotne assignment, store pairs in list
    #start from the last pair
    m = x - 1
    n = y - 1
    C = [[m, n]]
    while m > 0 or n > 0:
        if m == 0:
            pred = [0, n - 1]
            n = n - 1
        elif n == 0:
            pred = [m - 1, 0]
            m = m - 1
        else:
            pred_dis = min(dp[m - 1][n - 1], dp[m][n - 1], dp[m - 1][n])
            if pred_dis == dp[m - 1][n - 1]:
                pred = [m - 1, n - 1]
                m = m - 1
                n = n - 1
            elif pred_dis == dp[m][n - 1]:
                pred = [m, n - 1]
                n = n - 1
            else:
                pred = [m - 1, n]
                m = m - 1
        C.append(pred)
    C.reverse()
    return dp[x-1][y-1],C
