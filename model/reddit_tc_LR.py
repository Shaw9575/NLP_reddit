import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

import word2vec
from data_loader import load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from utils import plot_confusion_matrix
from utils import timeSince

import time
import math


dim_word2vec = 200

batch_size = 16
epochs = 5
learning_rate = 1e-1
model_fn = '../results/rnn_tc/rnn_tc_50d.11160425.pt'

glov_wv = word2vec.load_wv_from_model('../data/word2vec/retrained_word2vec/reddit_word2vec_200d')

data_train, label_train = load_data('../data/train.csv')
data_test, label_test = load_data('../data/test.csv')

#check number of labels in each class
tmp = label_test.tolist()
for label in np.unique(label_test):
    print('{}: {}'.format(label, tmp.count(label)))

tmp2 = label_train.tolist()
for label in np.unique(label_train):
    print('{}: {}'.format(label, tmp2.count(label)))

def avg_length(sentences):
    N = len(sentences)
    all_length = 0
    for sentence in sentences:
        all_length += len(sentence)

    return all_length / N    

def txt2vector(sentences, dic, dim, pad_length=14):
    vectors = []
    count = 0
    
    for sentence in sentences:
        vector = []
        for word in word2vec.preprocess(sentence[0]):
            if len(vector) <= pad_length and word in dic:
                vector.append(dic[word])
            
            if word not in dic:
                count += 1
        
        while len(vector) <= pad_length:
            vector.append(np.ones(dim))
            
        vectors.append(vector)
        
    print('missing words: %d, avg missing words: %f' % (count, count / sentences.shape[0]))
    return np.array(vectors)

data_train_vec = txt2vector(data_train, glov_wv, dim_word2vec, 15)
data_test_vec = txt2vector(data_test, glov_wv, dim_word2vec, 15)

all_categories = np.unique(label_train).tolist()
print('categories:', all_categories)

def label2vector(labels, all_categories):
    return np.array([all_categories.index(label) for label in labels])

label_train_vec = label2vector(label_train, all_categories)
label_test_vec = label2vector(label_test, all_categories)


def batch_iter(X, y, batch_size=64, shuffle=True):
    N = len(X)
    num_batch = int((N - 1) / batch_size) + 1
    
    if shuffle:
        indices = np.random.permutation(np.arange(N))
        X_shuffle = X[indices]
        y_shuffle = y[indices]
    else:
        X_shuffle = X
        y_shuffle = y
    
    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min((i+1) * batch_size, N)
        yield torch.tensor(X_shuffle[start_idx: end_idx], dtype=torch.float), torch.tensor(y_shuffle[start_idx: end_idx])


class LR_tc(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR_tc, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out


n_class = len(all_categories)
n_training = len(data_train)

model = LR_tc(dim_word2vec*batch_size, n_class)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

losses = []

start = time.time()
min_loss = 1e+5

for epoch in range(epochs):
    idx = 0
    for X_batch, y_batch in batch_iter(data_train_vec, label_train_vec, batch_size):
        cur_batch_size = y_batch.size()[0]
        new_X = X_batch.reshape(-1,batch_size*dim_word2vec)
        
        optimizer.zero_grad()
        
        output = model(new_X)
        loss = criterion(output, y_batch)
        
        loss.backward(retain_graph=True)        
        optimizer.step()        
        
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model, model_fn)
        
        idx += 1
        
        if idx % 100 == 0:
            losses.append(loss.item())
            print('%s, %d epoch, %d index, %f loss' % (timeSince(start),epoch, idx, loss.item())) 


plt.figure()
plt.plot(np.arange(len(losses))/10.0, losses)
plt.show()


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i]

y_preds = []
for X_batch, y_batch in batch_iter(data_test_vec, label_test_vec, 1, False):
    
    new_X = X_batch.reshape(-1,batch_size*dim_word2vec)
    
    output = model(new_X)    

    for r in output:
        y_preds.append(category_from_output(r, all_categories))

cnf_matrix = confusion_matrix(label_test, y_preds)

print('parameters: epochs:%d, batch_size:%d, learning_rate:%f.' % (epochs, batch_size, learning_rate))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=all_categories, title='Confusion matrix')
plt.show()

print(classification_report(label_test, y_preds))

