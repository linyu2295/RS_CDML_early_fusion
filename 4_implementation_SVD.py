#!/usr/bin/python

import sys

import numpy as np
import pandas as pd
from numpy import asarray, savez_compressed, load

from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import random
import matplotlib.patheffects as PathEffects
from sklearn.metrics.pairwise import pairwise_distances, linear_kernel
from random import sample
import time

from scipy.sparse import csr_matrix
import surprise

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import re


'''
Implement CDML with company data
Article recommendation
    Load Dataset:
        1. Article content:
            - articleId (string)
            - article title (string)
            - category (string)
        2. User behavior:
            - Read/clicks
    output filename
'''

article_filename = sys.argv[1]
click_filename = sys.argv[2]

# top-k recommendation
k = 4

top_time = time.time()

# 1. Read article content features:
#    columns: 
#           articleId (string)
#           raw title context (string)
#           category (string)

df_articles = pd.read_csv(article_filename, names= ['articleId', 'title', 'category'])
print('Article content file (articleId, title, category): shape = {}'.format(df_articles.shape))

## Data preprocessing: 
# (1) drop NaNs
df_articles = df_articles.dropna()
# (2) deduplication
df_articles = df_articles[df_articles['articleId'] != '0']
df_articles.drop_duplicates('articleId', inplace = True)


## Raw title/body context:
# Break up the big title/body context string into a string array of words
df_articles['title'] = df_articles['title'].apply(lambda x: re.findall(r'\w+', x))
# df_articles['body'] = df_articles['body'].apply(lambda x: re.findall(r'\w+', x))

# Sort by articleId
df_articles = df_articles.sort_values(by = 'articleId').reset_index(drop = True)


# 2. User behavior:
#    columns: 
#           userId (string)
#           articleId (string)

df_clicks = pd.read_csv(click_filename, names = ['userId', 'articleId'])
print('User behavior file (userId, articleId): shape = {}'.format(df_clicks.shape))

## Data preprocessing: 
# (1) drop NaNs
df_clicks = df_clicks.dropna()
# (2) deduplication
df_clicks = df_clicks[df_clicks['articleId'] != '0']
df_clicks = df_clicks.drop_duplicates().reset_index(drop=True)

##############################################
# Combine article content and clicks dataset
df_combine = pd.merge(df_clicks, df_articles, how = 'left', on = 'articleId')
# print(df_combine.shape)

print('Number of unique articles: {} and number of unique users: {}'.format(len(df_combine.articleId.unique()), len(df_combine.userId.unique()))) # get number of unique articles and unique users


###################################
'''
    
    Load Co-watched matrix for articles:
        - Node: articles
        - Edge: weighted by co-watch frequency
    
'''

print('Loading Cowatch matrix...')
start_time = time.time()

cowatched_mat_s = load('cowatched_matrix.npz')['arr_0']

print('Complete loading cowatch matrix in {} seconds...'.format(round(time.time() - start_time, 2)))



'''
    6.  SVD
'''

print('Start preparing SVD-based model...')

dat = df_clicks.copy()
dat['click'] = 1

reader = surprise.Reader(rating_scale=(dat.click.min(), dat.click.max()))
data = surprise.Dataset.load_from_df(dat, reader)

svd_alg = surprise.SVD(random_state = 123)

# n_factor: default = 100
svd_output = svd_alg.fit(data.build_full_trainset())

# output
# pu: The user factors (n_users, n_factors)
# qi: The item factors (n_items, n_factors)
# print(svd_output.pu.shape, svd_output.qi.shape)

sim_mat_SVD = 1 - pairwise_distances(svd_output.qi, metric = 'correlation')

del dat, data, reader, svd_output



'''
    Article similary matrix:
        - sim_mat_CDML_early
        - sim_mat_CDML_late
        - sim_mat_content
        - sim_mat_tfidf
        - sim_mat_NMF
        - sim_mat_SVD
        - KNN 
'''

# Evaluation: 
#           - NDCG
#           - MAP


def cal_NDCG(rel_scores, threshold):
    if not threshold:
        # print('Original')
        DCG_k = sum([(2**i[1] - 1)/(np.log2((i[0]+1)+1)) \
                     for i in list(enumerate(rel_scores))])
        IDCG_k = sum([(2**i[1] - 1)/(np.log2((i[0]+1)+1)) \
                      for i in list(enumerate(sorted(rel_scores, reverse=True)))])
        NDCG_k = DCG_k/(IDCG_k+0.0001)
    else:
        # print(threshold)
        DCG_k = sum([(2**int(i[1] >= threshold) - 1)/(np.log2((i[0]+1)+1)) \
                 for i in list(enumerate(rel_scores))])
        IDCG_k = sum([(2**int(i[1] >= threshold) - 1)/(np.log2((i[0]+1)+1)) \
                      for i in list(enumerate(sorted(rel_scores, reverse=True)))])
        NDCG_k = DCG_k/(IDCG_k+0.0001)
    
    return NDCG_k


def cal_MAP(cowatched_mat_s, idx, rel_scores, threshold):    
    num_actual_rel = (cowatched_mat_s[idx] >= threshold).astype(int).sum()
    rel_scores_b = [int(i >= threshold) for i in rel_scores]

    precs = []
    recalls = []
    for indx, rec in enumerate(rel_scores_b):
        precs.append(sum(rel_scores_b[:indx+1])/(indx+1))
        recalls.append(sum(rel_scores_b[:indx+1])/(num_actual_rel+0.0001))

    delta_recalls = recalls - np.concatenate([[0], recalls[:(len(recalls)-1)]])
    # print(delta_recalls)

    MAP = sum(precs*delta_recalls)
    
    return MAP


# Function that get item recommendations
# method: 'standard', 'tripletNN'
def item_recommendations(articleId, item_corr, cowatched_mat_s, method, k):
    #print(method)

    idx = indices[articleId]
    target_category = articleIds[articleIds.articleId == articleId]['category'].values

    sim_scores_all = list(enumerate(item_corr[idx]))
    sim_scores_all.pop(idx)

    sim_scores_all = sorted(sim_scores_all, key = lambda x: x[1], reverse = True)

    # sim_scores = sim_scores[:k] # 
    article_indices = [i[0] for i in sim_scores_all]
    tmp_recommend = articleIds.iloc[article_indices].reset_index(drop = True)

    sim_scores = []
    kk = 1
    for i in range(tmp_recommend.shape[0]):
        if tmp_recommend['category'][i] != target_category and kk <= k:
            sim_scores.append(sim_scores_all[i])
            kk += 1
        elif tmp_recommend['category'][i] == target_category and kk <= k:
            continue
        elif kk > k:
            break

    del article_indices, tmp_recommend

    rel_scores = [cowatched_mat_s[idx, j] for j in [i[0] for i in sim_scores]]
    #print('# of cowatched users (rel_scores): '.format(rel_scores))

    ## (1) NDCG
    NDCG_k = cal_NDCG(rel_scores, threshold = None)
    #print('NDCG_orginal = {}'.format(NDCG_k))

    NDCG_k_b = cal_NDCG(rel_scores, threshold = 1)
    #print('NDCG_binary = {}'.format(NDCG_k_b))

    NDCG_k_b10 = cal_NDCG(rel_scores, threshold = 10)
    #print('NDCG_threshold10 = {}'.format(NDCG_k_b10))

    ## (2) MAP

    MAP_b = cal_MAP(cowatched_mat_s, idx, rel_scores, threshold = 1)
    #print('MAP_binary = {}'.format(MAP_b))

    MAP_b10 = cal_MAP(cowatched_mat_s, idx, rel_scores, threshold = 10)
    #print('MAP_threshold10 = {}'.format(MAP_b10))

    article_indices = [i[0] for i in sim_scores]
    tmp_recommend = articleIds.iloc[article_indices].reset_index(drop = True)

    return pd.concat([tmp_recommend, pd.DataFrame(rel_scores, columns=['rel_score'])], axis = 1), \
            NDCG_k, NDCG_k_b, NDCG_k_b10, MAP_b, MAP_b10


# evaluation with articleId_lists
def get_eval_results(articleId_lists, item_corr, cowatched_mat_s, method, k):
    print(method)
    
    df_metrics = pd.DataFrame(index = articleId_lists, columns = ['NDCG_orginal', 'NDCG_binary', 'NDCG_threshold10',
                                                                   'MAP_binary', 'MAP_threshold10'])

    df_popular_cnts = np.zeros((len(articleId_lists), k))
    df_rel_scores = np.zeros((len(articleId_lists), k))
    
    for i, target_id in enumerate(articleId_lists):
        tmp_results = item_recommendations(target_id, item_corr, cowatched_mat_s, method = method, k = k)
        df_popular_cnts[i] = tmp_results[0].cnts
        df_rel_scores[i] = tmp_results[0].rel_score
        
        # NDCG
        df_metrics.NDCG_orginal[target_id] = tmp_results[1]
        df_metrics.NDCG_binary[target_id] = tmp_results[2]
        df_metrics.NDCG_threshold10[target_id] = tmp_results[3]
        
        # MAP
        df_metrics.MAP_binary[target_id] = tmp_results[4]
        df_metrics.MAP_threshold10[target_id] = tmp_results[5]
            
    return df_metrics, df_popular_cnts, df_rel_scores







# Build a 1-dimensional array with articleIds
article_cnts = pd.DataFrame(df_clicks.groupby('articleId').size(), columns = ['cnts']).reset_index()

df_articles_cnts = pd.merge(df_articles, article_cnts, how = 'left', on = 'articleId') # articleId, title, body, cnts

articleIds = df_articles_cnts.copy()
indices = pd.Series(df_articles_cnts.index, index = df_articles_cnts['articleId'])


#------------------------------------------------------------------------------------------
# evaluation articleIds

eval_num = 200
articleId_lists =  list(df_articles.sample(eval_num, random_state=123, replace = False)['articleId']) 

print('Evaluation with {} articles...'.format(eval_num))


#------------------------------------------------------------------------------------------
# 6. SVD

SVD_df_metrics, SVD_popular_cnts, SVD_rel_scores = get_eval_results(articleId_lists, 
                                                                    item_corr = sim_mat_SVD,
                                                                    cowatched_mat_s = cowatched_mat_s,
                                                                    method = 'SVD',
                                                                    k = k)

#------------------------------------------------------------------------------------------
# Comparison results
comparison_results = pd.DataFrame([SVD_df_metrics.mean(axis = 0, skipna = True)], 
                                index=['SVD_mean']) 

print('Comparison results for top-{} recommendaitons ...'.format(k))
print(comparison_results)

output_filename = '4_comparison_results_SVD.csv'
comparison_results.to_csv(output_filename)

print('Complete running SVD-based recommendation models in {} seconds...'.format(round(time.time() - top_time,2)))


