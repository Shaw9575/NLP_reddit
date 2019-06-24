
# coding: utf-8


import numpy as np

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from gensim.utils import simple_preprocess
from gensim.utils import simple_tokenize


# command line to convert glove file to word2vec file.
# python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>

def load_wv(fn='../data/word2vec/glove.twitter.27B/word2vec.50d.txt'):
    '''
    load word vector from fn

    Returns:
    - wv: dictionary, {key: value} corresponds to {word : vectors}
    '''

    wv = KeyedVectors.load_word2vec_format(fn, binary=False)
    return wv

def load_wv_from_model(fn='../data/word2vec/retrained_word2vec/reddit_word2vec'):
    return load_model_with(fn).wv


def preprocess(sentence):
    '''
    Inputs:
    - sentence: string

    Return:
    - sentence: list, separted word in each cell.
    '''
    sentence = simple_preprocess(sentence)
    return sentence

def load_corpus(fn='../data/data.csv'):
    '''
    load sentences from fn and preprocess them

    Return:
    - sentences: list[list[]], each sentence is preprocessed. 
    '''

    sentences = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sentence = line.split('\t')[0].strip()
#             sentence = sentence.split(' ')
            sentences.append(preprocess(sentence))
    return sentences

def retrain_model(sentences, dim, pretrain_fn='../data/word2vec/glove.twitter.27B/word2vec.50d.txt'):
    '''
    retrain model based on existing word2vector
    '''

    model = Word2Vec(size=dim, min_count=5)
    model.build_vocab(sentences)
    total_examples = model.corpus_count
    
    print('load pre-trained vectors...')
    glove_wv = load_wv(pretrain_fn)
    
    print('intersect glove vectors')
    model.build_vocab([list(glove_wv.vocab.keys())], update=True)
    model.intersect_word2vec_format(pretrain_fn, binary=False, lockf=1.0)
    
    print('train model...')
    model.train(sentences, total_examples=total_examples, epochs=model.iter)
    return model

def build_model(sentences, dim):
    '''
    Inputs:
    - sentences: list[list[]], training data which is tokenized.
    - dim: int, dimensions of vectors.

    Returns:
    - model: word2vec which contains model parameters and word2vectors
    '''
    model = Word2Vec(sentences, size=dim, min_count=2)
    model.train(sentences, total_examples=model.corpus_count, epochs=10)
    return model

def save_model_to(model, fn='../data/word2vec/retrained_word2vec/reddit_word2vec'):
    print('saving word2vec...')
    model.save(fn)


def load_model_with(fn='../data/word2vec/retrained_word2vec/reddit_word2vec'):
    print('loading word2vec...')
    model = Word2Vec.load(fn)
    return model

def retrain():
    sentences = load_corpus()
    model = retrain_model(sentences, 50)
    save_model_to(model)

    model = load_model_with()
    # print(model.wv.vocab.keys())

    print(model.similar_by_word('teacher', topn=10))    

if __name__ == '__main__':
    # sentences = load_corpus()
    # print(sentences)
    # model = build_model(sentences, 50)
    # save_model_to(model, '../data/word2vec/retrained_word2vec/test')

    # model = load_model_with('../data/word2vec/retrained_word2vec/test')
    # print(model.wv.vocab.keys())

    # print(model.similar_by_word('teacher', topn=10))

    retrain()


