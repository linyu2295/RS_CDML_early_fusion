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

Test case:
  - two input filenames: test_articles.csv (articleId, title, category) and test_clicks.csv (userId, articleId)

Steps:

  - First run 0_Data_preprocessing.py to preprocess the input dataset
```
  python 0_Data_preprocessing.py test_articles.csv test_clicks.csv
```
  - Then run other five files separately
```
  python 1_implementation_CDML_and_Content_only.py test_articles.csv test_clicks.csv
  python 2_implementation_TFIDF.py test_articles.csv test_clicks.csv
  python 3_implementation_NMF.py test_articles.csv test_clicks.csv
  python 4_implementation_SVD.py test_articles.csv test_clicks.csv
  python 5_implementation_KNN.py test_articles.csv test_clicks.csv
``` 

  - The comparison results will be saved in 5 different csv files:
```
  1_comparison_results_CDML_and_content_only.csv
  2_comparison_results_TFIDF.csv
  3_comparison_results_NMF.csv
  4_comparison_results_SVD.csv
  5_comparison_results_KNN.csv
```

