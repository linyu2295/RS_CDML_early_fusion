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
  
# Run python script with three filename arguments

Test case:
  - two input filenames: test_articles.csv (articleId, title, category) and test_clicks.csv (userId, articleId)
  - one output filename: test_comparison_results.csv

```
  python CDML_implementation_new.py test_articles.csv test_clicks.csv test_comparison_results.csv
``` 
