#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import copy
highway = pd.read_csv('highway.csv')

#process data
lane1 = highway.loc[highway['lane']==1]
t1 = lane1.loc[lane1['id']==5]
t2 = lane1.loc[lane1['id']==7]
#change dataframe to array, only keep the x and y columns.
t1 = t1[['x','y']].values
t2 = t2[['x','y']].values


# In[2]:


def isnear(A,B,h) :
    """
    define whether two points is near considering distance versus threshold
    :param A, B: points (x,y) 
    :param h: threshold, determine whether two points is near enough
    """
    indicator = np.sign(h-np.sqrt(np.square(A[0]-B[0])+np.square(A[1]-B[1])))
    if indicator>=0:
        return 1
    else:
        return 0 
    
def lcss_assign(t1,t2,h,n):
    """
    :param t1:trajectory 1, in the form of numpy array
    :param t2:trajectory 2, in the form of numpy array
    :param h: threshold, used in the innear function to define "near"
    :param n: number, the upper bound of the id difference of the selected two points
    :return: distance(similarity) between the two trajectories
    :return: the monotone assignment with the minimum cost, in the form of lists
    :dp1,dp2: recording matrix, can be deleted later
    """
    
    # get the length of each trajectory
    x = t1.shape[0]
    y = t2.shape[0]
    # create a matrix to store
    dp1 = np.zeros((x, y))
    dp2 = np.zeros((x, y))
    C=[]
    
    for i in range(0, x):
        for j in range(0, y):
            if isnear(t1[i],t2[j],h) and abs(i-j)<=n:
                dp1[i][j]=dp1[i-1][j-1]+1
                dp2[i][j]=1
                C.append([i+1,j+1])
            else:
                dp1[i][j]=max(dp1[i-1][j],dp1[i][j-1])
                dp2[i][j]=0
                
    similarity=sum(dp2)/max(x,y)
    
    return similarity,C


# In[3]:


import timeit

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


lanes = list(set(highway['lane']))
df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])


# In[7]:


for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,2,0)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[8]:


df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,2,1)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[15]:


df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,2,2)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[11]:


df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,2,3)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[11]:


df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,3,0)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[12]:


df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,3,1)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[13]:


df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,3,2)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[14]:


df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'lcss', 'dtw_time', 'lcss_time'])
for i in lanes:
    lane = highway.loc[highway['lane']==i]
    ids = list(set(lane['id']))
    #print(ids)
    ids.sort()
    for id1 in ids:
        #print(id1)
        for id2 in ids:
            if id1 < id2:
                t1 = lane.loc[lane['id']==id1]
                t2 = lane.loc[lane['id']==id2]
                t1 = t1[['x','y']].values
                t2 = t2[['x','y']].values
                #二选一 未确定
                #run dtw function and store running time
                start = timeit.default_timer()
                dtw_cost = dtw(t1,t2)
                end = timeit.default_timer()
                dtw_time = end-start
                #run dfd function and store running time
                start = timeit.default_timer()
                lcss_cost,c = lcss_assign(t1,t2,3,3)
                end = timeit.default_timer()
                lcss_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw(t1,t2)/max(t1.shape[0],t2.shape[0])
                # compute for pair1
                #dtw1, C_dtw1 = dtw_assign(t1, t2)
                #dfd1, C_dfd1 = dfd_assign(t1, t2)
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,lcss_cost,dtw_time,lcss_time]],columns=['lane','id1','id2','dtw_normalized','lcss','dtw_time','lcss_time']))
df1 = df
df1['rank_dtw_time'] = df1['dtw_time'].rank()
df1['rank_lcss_time'] = df1['lcss_time'].rank()
df1


# In[20]:


####Running time analysis
#print the descriptive statistics
print(df1['dtw_time'].describe())
print(df1['lcss_time'].describe())
#plot boxplot
df1.plot(y=['dtw_time','lcss_time'], kind='box',figsize=(16,9))
plt.savefig('runtime_compare_boxplot.png')
plt.show()


# In[14]:


df


# In[18]:


####Quality of results analysis
print(df1['dtw_normalized'].describe())
print(df1['lcss'].describe())
#plot boxplot
df1.plot(y=['dtw_normalized'], kind='box',figsize=(16,9))
plt.savefig('quality_dtw_boxplot.png')
plt.show()


# In[19]:


df1.plot(y=['lcss'], kind='box',figsize=(16,9))
plt.savefig('quality_dfd_boxplot.png')
plt.show()


# In[4]:


# define EDR function

def edr_assign(t1, t2, h):
    """
    :param t1:trajectory 1, in the form of numpy array
    :param t2:trajectory 2, in the form of numpy array
    :param h: threshold, used in the innear function to define "near"
    :return: distance(similarity) between the two trajectories
    :return: the monotone assignment with the minimum cost, in the form of lists
    :dp1,dp2: recording matrix, can be deleted later
    """
    # get the length of each trajectory
    x = len(t1)
    y = len(t2)
    # create a matrix to store
    dp1 = [[0] * (y + 1) for _ in range(x + 1)]
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            dist = np.linalg.norm(t1[i-1]-t2[j-1])
            if dist < h:
                subcost = 0
            else:
                subcost = 1
            dp1[i][j] = min(dp1[i][j - 1] + 1, dp1[i - 1][j] + 1, dp1[i - 1][j - 1] + subcost)

    edr = float(dp1[x][y]) / max([x, y])

    return edr


# In[5]:


#4.1 conducting experiments to compare dtw and edr distance functions
#for each lane, compute the distance between any two trajectories(i.e., two ids),and
lanes = list(set(highway['lane']))
df = pd.DataFrame(columns=['lane', 'id1', 'id2', 'dtw_normalized', 'edr', 'dtw_time', 'edr_time'])
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
                #run EDR function and store running time
                start = timeit.default_timer()
                edr_cost = edr_assign(t1,t2,2) #找h=2和另一个试试就行？
                end = timeit.default_timer()
                edr_time = end-start
                #normalize dtw cost.
                #We divide DTW cost by the number of pairs, which is the the minimun number of cells needed to be visited in the DP table
                dtw_cost_nor = dtw_cost/max(t1.shape[0],t2.shape[0])
                df = df.append(pd.DataFrame(data = [[i,id1,id2,dtw_cost_nor,edr_cost,dtw_time,edr_time]],columns=['lane','id1','id2','dtw_normalized','edr','dtw_time','edr_time']))
#process in the copy df1
df1 = df


# In[6]:


####Running time analysis
#print the descriptive statistics
print(df1['dtw_time'].describe())
print(df1['edr_time'].describe())
#plot boxplot
df1.plot(y=['dtw_time','edr_time'], kind='box')
plt.savefig('runtime_compare_boxplot.png')
plt.show()

print(df1['dtw_normalized'].describe())
print(df1['edr'].describe())
df1.plot(y=['edr'], kind='box')
plt.savefig('edr_boxplot.png')
plt.show()

df1.plot(y=['dtw_normalized'], kind='box')
plt.savefig('dtw_boxplot(with edr).png')
plt.show()


# In[ ]:





# In[ ]:




