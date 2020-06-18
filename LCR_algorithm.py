import sys

import numpy as np
import pandas as pd
import time
import os

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

import surprise

import itertools


click_filename = 'test_clicks_rating_100.csv'

df_clicks = pd.read_csv(click_filename, names = ['userId', 'articleId', 'click'])

'''
    Input: m-by-n click matrix M
'''

dat = df_clicks.copy()
M_df = dat.pivot(index = 'userId', columns = 'articleId', values = 'click').fillna(0)

m = len(dat['userId'].unique())
n = len(dat['articleId'].unique())

print('number of users: {} and number of articles: {}'.format(m, n))

M = M_df.values


# users
user_IndtoId = {
    Ind: Id for Ind, Id in
    enumerate(list(M_df.index))
}
user_IdtoInd = {
    Id: Ind for Ind, Id in
    enumerate(list(M_df.index))
}

# articles
article_IndtoId = {
    Ind: Id for Ind, Id in
    enumerate(list(M_df.columns))
}

article_IdtoInd = {
    Id: Ind for Ind, Id in
    enumerate(list(M_df.columns))
}


'''
    Parameters: 
        - number of local models: q in {10, 20, 30, 40, 50}
        - local rank: r in {5, 10, 15, 20}
        - loss function: {log[M], log[A], Exp[M], Hinge[A]} with gamma = 0
            - log[M]: \delta-M * log{1 + exp(gamma - \delta-f) }
        - determined by cross-validation:
            - learning rate: lr
            - regularization coefficient: lamb
        
        - Smoothing kernel: Epanechnikov kernel with kernel bandwidth = 0.8

    Stop criteria:
        - Either the improvement in training error got smaller than some threshold (0.001)
        - Or the algorithm reached 200 iterations
'''

# Initial values:
q = 2

r = 5

# chose log[M] as an example here
lr = 3e-4
lamb = 1


# Step 3: Define A as the set of observed entries in M
# - key: index
# - value: tuple of user index in M (u) and item index in M (i)
A = {}
for i in range(dat.shape[0]):
     A[i] = (user_IdtoInd[dat.iloc[i, 0]], article_IdtoInd[dat.iloc[i, 1]])

print(len(A), dat.shape)


# Define a diction A_ij
# - key: user index
# - value: items clicked by user 
A_ij = {}

for i in range(dat.shape[0]):
    if user_IdtoInd[dat.iloc[i, 0]] not in A_ij:
        A_ij[user_IdtoInd[dat.iloc[i, 0]]] = [article_IdtoInd[dat.iloc[i, 1]]]
    else:
        A_ij[user_IdtoInd[dat.iloc[i, 0]]].append(article_IdtoInd[dat.iloc[i, 1]])

# Step 4: for loop

def Initialize_q_models(q, r, dat):
    U = {} # each U_t: m-by-r
    V = {} # each V_t: n-by-r
    anchors = {} # (u_t, i_t) index
    
    for t in range(1, (q+1)):
        # Step 5: initialize U_t and V_t by using SVD

        tmp_reader = surprise.Reader(rating_scale=(dat.click.min(), dat.click.max()))
        tmp_data = surprise.Dataset.load_from_df(dat, tmp_reader)
        tmp_svd = surprise.SVD(random_state = 123 + t, n_factors = r)

        tmp_output = tmp_svd.fit(tmp_data.build_full_trainset())

        U[t] = tmp_output.pu # user factors (m, r)
        V[t] = tmp_output.qi # item factors (n, r)

        # Step 6: pick an observed pair (u_t, i_t) from M at random

        tmp_anchor = dat[['userId', 'articleId']].sample(1, random_state=123+t)
        anchors[t] = [(user_IdtoInd[u], article_IdtoInd[i]) for u, i in tmp_anchor.values][0]

    return U, V, anchors


# Step 7: end for loop
start_time = time.time()

U_initial, V_initial, anchors = Initialize_q_models(q, r, dat)

print('Running time for step 4-7 of initialization is {}'.format(round(time.time() - start_time, 2)))


'''
    Step 8 - 32: while loop
    
    Define several functions to get for loop results:
    
        - step 10 - 13: estimation step
        - step 14 - 16: estimation step
        - step 18 - 31: update step
'''

def smooth_kernel(s1, s2, h = 0.8):
    d = np.dot(s1, s2) / ( np.sqrt(np.dot(s1, s1))*np.sqrt(np.dot(s2, s2)) )
    if d < h:
        return (1 - d**2)*3/4
    else:
        return 0

# step 10 - 13
def estimation_each_u_i(q, U_t, V_t, A): 
    w = {}
    f = {}

    for _, a in A.items():
        # print(a)
        w[a] = 0
        f[a] = 0
        u_ind = a[0]
        i_ind = a[1]
        
        for t in range(1, q+1):
            ut_ind = anchors[t][0]
            it_ind = anchors[t][1]
            
            w[a] += smooth_kernel(U_t[t][ut_ind], U_t[t][u_ind]) *\
                    smooth_kernel(V_t[t][it_ind], V_t[t][i_ind])
        
        w[a] += 0.00001
        for t in range(1, q+1):
            ut_ind = anchors[t][0]
            it_ind = anchors[t][1]
            
            UV_ui = U_t[t].dot(V_t[t].T)[u_ind][i_ind]
            f[a] += smooth_kernel(U_t[t][ut_ind], U_t[t][u_ind]) *\
                    smooth_kernel(V_t[t][it_ind], V_t[t][i_ind]) /\
                    w[a]*UV_ui
    
    return w, f

# step 14 - 16
def estimation_each_u_i_j(f, A_ij, M, gamma = 0):
    # here we use log[M]
    l = {} 

    for u, ijs in A_ij.items():
        if len(ijs) > 1:        
            tmp_ij_lists = list(itertools.combinations(ijs, 2))
        
            for p in tmp_ij_lists:
                g_uij = f[(u, p[0])] - f[(u, p[1])]
                l[(u, p[0], p[1])] = - (M[u][p[0]] - M[u][p[1]])*np.exp(gamma - g_uij) / \
                                        (1 + np.exp(gamma - g_uij))
     
        elif len(ijs) == 1:
            continue
    return l


# step 18 - 30:
def update_step(q, m, n, r, anchors, A_ij, s_u, U_t, V_t, w, f, l, lr, lamb):

    for t in range(1, q+1):
        ut_ind = anchors[t][0]
        it_ind = anchors[t][1]

        delta_U = np.zeros((m, r))
        delta_V = np.zeros((n, r))

        for u, ijs in A_ij.items():

            if len(ijs) > 1:
                tmp_ij_lists = list(itertools.combinations(ijs, 2))

                for p in tmp_ij_lists:

                    if M[u][p[0]] > M[u][p[1]]:

                        delta_U[u] += l[(u, p[0], p[1])] * ( \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[0]])*V_t[t][p[0]]/w[(u, p[0])] - \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[1]])*V_t[t][p[1]]/w[(u, p[1])] )
                        delta_V[p[0]] += l[(u, p[0], p[1])] * U_t[t][u] * \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[0]])/w[(u, p[0])]
                        delta_V[p[1]] -= l[(u, p[0], p[1])] * U_t[t][u] * \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[1]])/w[(u, p[1])]
                    elif M[u][p[0]] < M[u][p[1]]:
                        
                        delta_U[u] += l[(u, p[0], p[1])] * ( \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[1]])*V_t[t][p[1]]/w[(u, p[1])] - \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[0]])*V_t[t][p[0]]/w[(u, p[0])] )
                        delta_V[p[1]] += l[(u, p[0], p[1])] * U_t[t][u] * \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[1]])/w[(u, p[1])]
                        delta_V[p[0]] -= l[(u, p[0], p[1])] * U_t[t][u] * \
                                        smooth_kernel(U_t[t][ut_ind], U_t[t][u])\
                                            *smooth_kernel(V_t[t][it_ind], V_t[t][p[0]])/w[(u, p[0])]
                        
                    
        for u in range(len(s_u)):
            if s_u[u] != 0:
                U_t[t][u] = U_t[t][u] - lr * (delta_U[u]/m/s_u[u] + lamb*U_t[t][u])
                V_t[t] = V_t[t] - lr * (delta_V/m/s_u[u] + lamb*V_t[t])
            else:
                continue
    return U_t, V_t


# Define s_u as a vector
# - each element is the number of the ordered items rated by user u (index)

s_u = []

for u, ijs in A_ij.items():
    if len(ijs) == 1:
        s_u.append(0)
    else:
        tmp_ij_lists = list(itertools.combinations(ijs, 2))
        tmp_s_u = 0
        for p in tmp_ij_lists:
            if M[u][p[0]] != M[u][p[1]]:
                tmp_s_u += 1
            else:
                continue
        s_u.append(tmp_s_u)
        

# calculating loss function
def loss_fun(f, s_u, A_ij, M, gamma = 0):
    loss = 0
    for u in range(len(s_u)):
        if s_u[u] != 0:
            tmp_ij_lists = list(itertools.combinations(A_ij[u], 2))
            tmp_loss = 0
            for p in tmp_ij_lists:
                g_uij = f[(u, p[0])] - f[(u, p[1])]
                tmp_loss += np.log(1 + np.exp(gamma - g_uij)) * (M[u][p[0]] - M[u][p[1]])
            loss += tmp_loss/s_u[u] 
        else:
            continue
    return loss



'''  
Step 8 - 32:
    stop criteria:
        - Either the improvement in training error got smaller than some threshold (0.001/1e-3)
        - Or the algorithm reached 200 iterations

'''
max_iters = 200
n_iter = 0
precision = 1e-3
precision_step_size = 1
U_t_old = U_initial
V_t_old = V_initial

start_time = time.time()
w_initial, f_initial = estimation_each_u_i(q, U_initial, V_initial, A)
print('Running time for step 10-13: {} seconds'.format(round(time.time() - start_time, 2)))

pre_loss = loss_fun(f_initial, s_u, A_ij, M)

while precision_step_size > precision and n_iter < max_iters:
    # start_time = time.time()
    print('Iteration = {}'.format(n_iter))
    if n_iter == 0:
        w, f = w_initial, f_initial
    else: 
        # step 10 - 13
        w, f = w_update, f_update
        # w, f = estimation_each_u_i(q, U_t_old, V_t_old, A)
    # print('Running time for step 10-13: {} seconds'.format(round(time.time() - start_time, 2)))

    # step 14-16:
    start_time = time.time()
    l = estimation_each_u_i_j(f, A_ij, M)
    print('Running time for step 14-16: {} seconds'.format(round(time.time() - start_time, 2)))

    # step 18 - 30:
    start_time = time.time()
    U_t_update, V_t_update = update_step(q, m, n, r, anchors, A_ij, s_u, U_t_old, V_t_old, w, f, l, lr, lamb)
    print('Running time for step 18-30: {} seconds'.format(round(time.time() - start_time, 2)))

    # calculate the training error improvements: precision_step_size
    start_time = time.time()
    w_update, f_update = estimation_each_u_i(q, U_t_update, V_t_update, A)
    print('Running time for updating step 10-13: {} seconds'.format(round(time.time() - start_time, 2)))
    cur_loss = loss_fun(f_update, s_u, A_ij, M)
    precision_step_size = abs(cur_loss - pre_loss)
    
    U_t_old, V_t_old = U_t_update, V_t_update
    
    n_iter += 1
    
    print('\n')



