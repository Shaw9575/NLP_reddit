import praw
from praw.models import MoreComments
import sys
import numpy as np

'''
This script read necessary information from labels and crawl data 
from reddit by API Reddit provides
'''

def search_with(subreddit, keyword, limit):
    gen = None
    gen = subreddit.search(keyword, limit=limit)
    return gen


def generator_with(reddit, label, sublabel, mode, limit):
    gen = None
    if mode == 'hot':
        subreddit = reddit.subreddit(sublabel)
        gen = subreddit.hot(limit=limit)
    elif mode == 'rising':
        subreddit = reddit.subreddit(sublabel)
        gen = subreddit.rising(limit=limit)
    elif mode == 'top':
        subreddit = reddit.subreddit(sublabel)
        gen = subreddit.top(limit=limit)
    elif mode == 'new':
        subreddit = reddit.subreddit(sublabel)
        gen = subreddit.new(limit=limit)
    elif mode == 'keyword':
        subreddit = reddit.subreddit(label)
        gen = subreddit.search(sublabel, limit=limit)

    return gen
        
def clean_title(title):
    return title.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

def crawl(labels, path='../data/raw/', limit=1000):
    '''
    if mode == 'keyword', we will search for results regarding keywords under 'topic' reddit. 
    Otherwise, we will search for results under 'keyword' reddit by 'new', 'rising', 'top', or
    'hot' function.

    Inputs:
    - limit: the maximum number of data to be retrieved once.
    - path: path for 'labels' file
    - labels: list[list[]]
    '''
    reddit = praw.Reddit(user_agent='ios::textclassification::',
                     client_id='yRaWU_KMA2WMGA', client_secret="_ljiXu3dPHd5lhOlZc78M97V-iU",
                     username='ZhengCao', password='cz19951029')    
    
    for label, sublabel, mode in labels:

        fn = path + '%s_%s_%s.csv' % (label, sublabel, mode) 
        # subreddit = reddit.subreddit(sublabel)
        # subreddit = reddit.subreddit(label)
        with open(fn, 'w', encoding='utf-8') as f:
            # gen = generator_with(subreddit, mode, limit)
            # gen = search_with(subreddit, sublabel, limit)
            gen = generator_with(reddit, label, sublabel, mode, limit)
            for submission in gen:
                f.write('%s\t%s\t%s\n' % (submission.id, clean_title(submission.title), label))


def read_labels(fn):
    '''
    In 'labels' file, each row contains following data: topic, keywords, and mode
    '''
    args = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                arg = line.strip().split(' ')
                keywords = '+'.join(arg[1:-1])
                args.append((arg[0], keywords, arg[-1]))
                
    return args

if __name__ == '__main__':
    # fn = sys.argv[1]
    # write_to_path = sys.argv[2]
    fn = 'labels'
    write_to_path = '../data/raw/'

    labels = read_labels(fn)
    crawl(labels, write_to_path)

