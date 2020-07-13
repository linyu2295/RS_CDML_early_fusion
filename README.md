# Collaborative Deep Metric Learning-based Recommender System

# Install packages:

1. Spacy-model for word embedding
  - Convert article content into word vectors
  - https://github.com/explosion/spacy-models/releases//tag/en_core_web_lg-2.2.5
  - Installation

``` 
    pip install spacy
    python -m spacy download en_core_web_lg
```  

2. Embedding Neural Network for content features:
  - tensorflow
  - keras
  
3. Input Data (cleaned data, no NaNs and deduplicated):
  - Article contents: articleId (string), title (string), category (string)
  - User behavior: UserId (string), articleId (string)


# Run python script

Five arguements:
  - two input filenames: TEST_ARTICLES.csv (articleId, title, category) and TEST_CLICKS.csv (userId, articleId)
  - NUM_ARTICLES: number of articles selected from
  - TOP_NUM_ARTICLES: select NUM_ARTICLEs from the top popular articles 
  - NUM_EVAL_ARTICLES: number of articles for evaluation
  - K: top-K recommendations
  - SEED: random seed
  
```
  python Comparison_results_companydata.py TEST_ARTICLES.csv TEST_CLICKS.csv NUM_ARTICLES TOP_NUM_ARTICLES NUM_EVAL_ARTICLES K SEED
```


