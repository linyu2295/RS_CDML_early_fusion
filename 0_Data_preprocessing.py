#!/usr/bin/python

import sys

import numpy as np
import pandas as pd
from numpy import asarray, savez_compressed, load

import time
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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

# output_filename = sys.argv[3]

# top-k recommendation
k = 4


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
df_clicks.drop_duplicates(inplace = True)



##############################################
# Combine article content and clicks dataset
df_combine = pd.merge(df_clicks, df_articles, how = 'left', on = 'articleId')
# print(df_combine.shape)

print('Number of unique articles: {} and number of unique users: {}'.format(len(df_combine.articleId.unique()), len(df_combine.userId.unique()))) # get number of unique articles and unique users



'''
    Construct co-watched graph for articles:
        - Node: articles
        - Edge: weighted by co-watch frequency
'''
df_user_data = df_combine[['userId', 'articleId']]
df_user_data = df_user_data.sort_values(by = 'articleId').reset_index(drop = True)


# Construct a user dictionary where the key is the articleId and the values are the userIds who watched this article
start_time = time.time()

d_user = {}

for i in range(df_user_data.shape[0]):
    if df_user_data['articleId'][i] not in d_user:
        d_user[df_user_data['articleId'][i]] = set()
    else:
        d_user[df_user_data['articleId'][i]].add(df_user_data['userId'][i])

# print(len(d_user))

print('Running time for constructing a user dictionary (d_user): {} seconds'.format(round(time.time() - start_time, 2)))


##############################################
# Create article-by-article user co-watched matrix
#   - cell_{i, j}: # of users who watched both article i and j

n_articles = len(df_articles['articleId'])
cowatched_mat = np.zeros((n_articles, n_articles))

for r in range(n_articles):
    for c in range(r+1, n_articles):
        cowatched_mat[r, c] = len(d_user[df_articles['articleId'][r]].intersection(d_user[df_articles['articleId'][c]]))

# Co-watched matrix: symmetric
cowatched_mat_s = cowatched_mat + cowatched_mat.T
# print(cowatched_mat_s[:5, :5])


savez_compressed('cowatched_matrix.npz', asarray(cowatched_mat_s))

print('Complete data preprocessing...')


