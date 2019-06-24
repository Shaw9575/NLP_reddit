
# coding: utf-8

# In[87]:


import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

import word2vec
from data_loader import load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils import plot_confusion_matrix
from utils import timeSince

import time


dim_word2vec = 50
seq_length = 15

#CNN parameters
filter_sizes = [2, 3, 4, 5, 6, 7, 10]
num_filter = 100
fc_input_size = len(filter_sizes) * num_filter
fc_hd_size = 200
dropout_p = 0.5


batch_size = 16
epochs = 15
learning_rate = 1e-4
model_fn = '../results/rnn_tc/rnn_tc_LSTM.11180355.pt'


glov_wv = word2vec.load_wv_from_model('../data/word2vec/retrained_word2vec/reddit_word2vec')


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
            if len(vector) < pad_length and word in dic:
                vector.append(dic[word])
            
            if word not in dic:
                count += 1
        
        while len(vector) < pad_length:
            vector.append(np.ones(dim))
            
        vectors.append(vector)
        
    print('missing words: %d, avg missing words: %f' % (count, count / sentences.shape[0]))
    return np.array(vectors)

data_train_vec = txt2vector(data_train, glov_wv, dim_word2vec, seq_length)
data_test_vec = txt2vector(data_test, glov_wv, dim_word2vec, seq_length)

all_categories = np.unique(label_train).tolist()
print('categories:', all_categories)

def label2vector(labels, all_categories):
    return np.array([all_categories.index(label) for label in labels])

label_train_vec = label2vector(label_train, all_categories)
label_test_vec = label2vector(label_test, all_categories)



class CNN(nn.Module):
    def __init__(self, seq_length ,embedding_size, filter_sizes, num_filter, fc_input_size, fc_hd_1, n_class, dropout_p):
        super(CNN, self).__init__()
        self.num_filter = num_filter
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        self.num_filters_total = len(self.filter_sizes) * num_filter
        
        self.in_channel = 1
        self.out_channel = num_filter

        self.convs = []
        self.pools = []
        for filter_size in filter_sizes:            
            self.convs.append(nn.Conv2d(self.in_channel, self.out_channel, (filter_size, embedding_size)))
            self.pools.append(nn.MaxPool2d((seq_length-filter_size+1, 1), stride=1))
        
        self.fc = nn.Linear(fc_input_size, fc_hd_1)
        self.fc_hd = nn.Linear(fc_hd_1, n_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input, is_train):
        pooled_inputs = []
        for conv, pool in zip(self.convs, self.pools):
            conved_input = conv(input)
            pooled_input = pool(conved_input)
            pooled_inputs.append(pooled_input)    
        output = torch.cat(pooled_inputs, 3)
        output = torch.reshape(output, (-1, self.num_filters_total))       
        output = self.fc(output)
        output = self.relu(output)
        if is_train:
            output = self.dropout(output)

        output = self.fc_hd(output)                
        output = self.softmax(output)
        return output


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



n_class = len(all_categories)
n_training = len(data_train)

model = CNN(seq_length, dim_word2vec, filter_sizes, num_filter, fc_input_size, fc_hd_size, n_class, dropout_p)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.NLLLoss()

losses = []

start = time.time()
min_loss = 1e+5

for epoch in range(epochs):
    scheduler.step()
    idx = 0
    for X_batch, y_batch in batch_iter(data_train_vec, label_train_vec, batch_size):
        cur_batch_size = y_batch.size()[0]
        X_batch = X_batch.reshape((cur_batch_size, 1, seq_length, dim_word2vec))
        
        output = model(X_batch, True)
        loss = criterion(output, y_batch)

        optimizer.zero_grad()
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
plt.plot(np.arange(len(losses)), losses)
plt.show()



def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i]

def test(X, y, y_vec):
    model = torch.load(model_fn)
    
    y_preds = []
    for X_batch, y_batch in batch_iter(X, y_vec, 1, False):

        cur_batch_size = y_batch.size()[0]
        X_batch = X_batch.reshape((cur_batch_size, 1, seq_length, dim_word2vec))

        output = model(X_batch, False)
        for r in output:
            y_preds.append(category_from_output(r, all_categories))

    cnf_matrix = confusion_matrix(y, y_preds)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=all_categories, title='Confusion matrix')
    plt.show()

    print(classification_report(y, y_preds))
    
test(data_test_vec, label_test, label_test_vec)

