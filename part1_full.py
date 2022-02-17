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

#import dataset "highway"
highway = pd.read_csv('/Users/fangzhushen/Desktop/2021FALL/COMPSCI531/project1/highway.csv')

#######3.2.1 compute dtw, dfd for 2 pairs
#process data
lane1 = highway.loc[highway['lane']==1]
#print all ids in lane 1
ids = list(set(lane1['id']))
print(ids)
#choose first pari: id = 5 & id=7
t1 = lane1.loc[lane1['id']==5]
t2 = lane1.loc[lane1['id']==7]
t1 = t1[['x','y']].values
t2 = t2[['x','y']].values

#choose second pair:id = 527 & id=18
#####pair 1
t3 = lane1.loc[lane1['id']==527]
t4 = lane1.loc[lane1['id']==18]
#change dataframe to array, only keep the x and y columns.
t3 = t3[['x','y']].values
t4 = t4[['x','y']].values

#compute for pair1
dtw1,C_dtw1 = dtw_assign(t1,t2)
dfd1,C_dfd1= dfd_assign(t1,t2)
#compuate for pair 2
dtw2,C_dtw2= dtw_assign(t3,t4)
dfd2,C_dfd2= dfd_assign(t3,t4)

#print results
print('For pair 1 (id=5 and id = 7): \n'+'dtw minimum cost = '+str(dtw1) +'\n the monotnone assignment of dtw is:'+str(C_dtw1)
        +'\n dfd minimum cost = '+str(dfd1)+'\n the monotnone assignment of dfd is:'+str(C_dfd1)
        +'\n For pair 2 (id=527 and id = 18): \n'+'dtw minimum cost = '+str(dtw2) +'\n the monotnone assignment of dtw is:'+str(C_dtw2)
        +'\n dfd minimum cost = '+str(dfd2)+'\n the monotnone assignment of dfd is:'+str(C_dfd2))


#3.2.2 Generate figures
#function to generate figure
def gen_figure(id1,id2,method,t1,t2,cost,C,name):
    #plt.figure(figsize=(16,9))
    plt.title(("ID 1(red points):"+id1+", ID 2(black points):"+id2+", Method: "+method+", Cost:"+str(cost)),fontdict={'fontsize':10})
    plt.scatter(t1[:,0],t1[:,1],color="r")
    plt.scatter(t2[:,0],t2[:,1],color="k")
    plt.plot(t1[:,0],t1[:,1],color="r")
    plt.plot(t2[:,0],t2[:,1],color="k")
    for i in range(len(C)):
        p1=C[i][0]
        p2=C[i][1]
        plt.plot([t1[p1][0],t2[p2][0]],[t1[p1][1],t2[p2][1]],linestyle="--",color="k")
    plt.savefig(name)

#generate 4 figures
gen_figure('5','7','DTW',t1,t2,dtw1,C_dtw1,'pair1_dtw.png')

gen_figure('5','7','DFD',t1,t2,dfd1,C_dfd1,'pair1_dfd.png')

gen_figure('517','18','DTW',t3,t4,dtw2,C_dtw2,'pair2_dtw.png')

gen_figure('517','18','DFD',t3,t4,dfd2,C_dfd2,'pair2_dfd.png')


#3.2.3 conducting experiments to compare dtw and dfd distance functions
#for each lane, compute the distance between any two trajectories(i.e., two ids),and
lanes = list(set(highway['lane']))
df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'dfd', 'dtw_time', 'dfd_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                dfd_cost = dfd(t1,t2)
                end = timeit.default_timer()
                dfd_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,dfd_cost,dtw_time,dfd_time]],columns=['lane','id1','id2','dtw_normalized','dfd','dtw_time','dfd_time']))
#process in the copy df1
df1 = df

####Running time analysis
#print the descriptive statistics
print(df1['dtw_time'].describe())
print(df1['dfd_time'].describe())
#plot boxplot
df1.plot(y=['dtw_time','dfd_time'], kind='box')
plt.savefig('runtime_compare_boxplot.png')
plt.show()

####Quality of results analysis
print(df1['dtw_normalized'].describe())
print(df1['dfd'].describe())
#plot boxplot
df1.plot(y=['dtw_normalized'], kind='box')
plt.savefig('quality_dtw_boxplot.png')
plt.show()

df1.plot(y=['dfd'], kind='box')
plt.savefig('quality_dfd_boxplot.png')
plt.show()

## analysis the point of maximum DTD time
max_index = df1['dfd'].argmax()
max_pair = df1.iloc[max_index]
print(max_pair)
#generate the figure of this pair
t1_max = highway.loc[highway['lane'] ==6]
t1_max = t1_max.loc[t1_max['id']==476]
t2_max = highway.loc[highway['lane'] ==6]
t2_max = t2_max.loc[t2_max['id']==590]
t1_max = t1_max[['x','y']].values
t2_max = t2_max[['x','y']].values

dtw_max,C_max = dtw_assign(t1_max,t2_max)
gen_figure('476','590','dtw',t1_max,t2_max,dtw_max,C_max,'max_dtw_cost.png')

dfd_max,C_max = dfd_assign(t1_max,t2_max)
gen_figure('476','590','dfd',t1_max,t2_max,dfd_max,C_max,'max_dfd_cost.png')

#generate figures for pairs have similar dtw cost with the max_dfd_cost pair
df2 = df1[df1['lane']==6]
df2 =df2.sort_values(['dtw_normalized'])
df2 = df2.reset_index()
max_index2 = df2['dfd'].argmax()
max_pair2 = df2.iloc[max_index2]
match_pair1 = df2.iloc[max_index2-1]
match_pair2 = df2.iloc[max_index2+1]

#generate the figure of match_pair1
t1_max = highway.loc[highway['lane'] ==6]
t1_max = t1_max.loc[t1_max['id']==81]
t2_max = highway.loc[highway['lane'] ==6]
t2_max = t2_max.loc[t2_max['id']==377]
t1_max = t1_max[['x','y']].values
t2_max = t2_max[['x','y']].values
dtw_max,C_max = dtw_assign(t1_max,t2_max)
gen_figure('81','377','dtw',t1_max,t2_max,dtw_max,C_max,'match_pair1_dtw.png')
dfd_max,C_max = dfd_assign(t1_max,t2_max)
gen_figure('81','377','dfd',t1_max,t2_max,dfd_max,C_max,'match_pair1_dfd.png')

#generate the figure of match_pair2
t1_max = highway.loc[highway['lane'] ==6]
t1_max = t1_max.loc[t1_max['id']==442]
t2_max = highway.loc[highway['lane'] ==6]
t2_max = t2_max.loc[t2_max['id']==700]
t1_max = t1_max[['x','y']].values
t2_max = t2_max[['x','y']].values
dtw_max,C_max = dtw_assign(t1_max,t2_max)
gen_figure('442','700','dtw',t1_max,t2_max,dtw_max,C_max,'match_pair2_dtw.png')
dfd_max,C_max = dfd_assign(t1_max,t2_max)
gen_figure('442','700','dfd',t1_max,t2_max,dfd_max,C_max,'match_pair2_dfd.png')

