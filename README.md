# NLP project - reddit question classification
Author: Shuang Yuan 
University of Southern California  
e-mail: shuangy@usc.edu  
----------------
## 0. Pipeline
    0.1 reddit_praw/reddit_praw.py: extract data  
    0.2 reddit_praw/data_integration.py: integrate data in data/raw folder  
    0.3 model/data_loader.py: split data.csv into test.csv and train.csv  
    0.4 model/word2vec.py: train word2vec  
    0.5 model/reddit_tc_*.py: models  
## 1. reddit_praw
    Tool for scraping titles of subreddits from reddit.com.  
  
    run reddit_praw.py file, read labels from 'labels' file and write results to '../data/raw/' folder  
## 2. data
    data.csv: all raw data including data and labels.  
    train.csv: training data.  
    test.csv: testing data.  

    raw folder: all raw data.  
    word2vec: word embeddings including GloVe and word vectors that we train. Note: since the files are so huge that they cannot be uploaded, we delete data in the files.   
## 3. model
    Model folder includes model files and tools for preprocessing data.  
