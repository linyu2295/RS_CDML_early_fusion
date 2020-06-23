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


# Run python script with three filename arguments

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
``` 

