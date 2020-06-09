# Recommender System

1. Collaborative Deep Metric Learning with early fusion
2. Collaborative Deep Metric Learning with late fusion

# Install packages:

1. Spacy-model for word embedding
  - Convert article content into word vectors
  - https://github.com/explosion/spacy-models/releases//tag/en_core_web_lg-2.2.5
  - Installation
    pip install spacy
    python -m spacy download en_core_web_lg
  
2. Embedding Neural Network for content features:
  - tensorflow
  - keras
  
# Run python script with three filenames arguments

Test case:
  - two input filenames: test_articles.csv and test_clicks.csv
  - one output filename: test_comparison_results.csv

  python CDML_implementation_new.py test_articles.csv test_clicks.csv test_comparison_results.csv
  
