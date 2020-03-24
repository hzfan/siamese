#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.parameter import ParameterDict
from mxnet import init

import re
import itertools
import datetime


# In[2]:


TRAIN_CSV = 'data/train-mini.csv'
TEST_CSV = 'data/test-mini.csv'
MODEL_SAVING_DIR = 'model/'


# In[26]:


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

# Load training and test set
train_df = pd.read_csv(TRAIN_CSV, dtype={"id": int, "qid1": int, "qid2": int,
                                         "question1": str, "question2": str, "is_duplicate": int})
test_df = pd.read_csv(TEST_CSV, dtype={"test_id": int,"question1": str, "question2": str})
# train_df = pd.read_csv(TRAIN_CSV)
# test_df = pd.read_csv(TEST_CSV)

stops = set(stopwords.words('english'))

# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

questions_cols = ['question1', 'question2']

# Iterate over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in tqdm(dataset.iterrows(), desc=("train" if dataset is train_df else "test")):
        # Iterate through the text of both questions of the row
        for question in questions_cols:

            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):

                # Check for unwanted words
                if word in stops:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.at[index, question] = q2n

embedding_dim = 128
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0


# In[27]:


#Pad the sequences to maxlen.
#if sentences is greater than maxlen, truncates the sentences
#if sentences is less the 500, pads with value 0 (most commonly occurrning word)
def pad_sequences(sentences,maxlen=500,value=0):
    """
    Pads all sentences to the same length. The length is defined by maxlen.
    Returns padded sentences.
    """
    padded_sentences = []
    for sen in sentences:
        new_sentence = []
        if(len(sen) > maxlen):
            new_sentence = sen[:maxlen]
            padded_sentences.append(new_sentence)
        else:
            num_padding = maxlen - len(sen)
            new_sentence = np.append(sen,[value] * num_padding)
            padded_sentences.append(new_sentence)
    return padded_sentences


max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                     train_df.question2.map(lambda x: len(x)).max(),
                     test_df.question1.map(lambda x: len(x)).max(),
                     test_df.question2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = int(0.1 * len(train_df))
training_size = len(train_df) - validation_size

X = train_df[questions_cols]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

print(type(X_train.question1))
print(type(Y_train))

# Split to dicts
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': test_df.question1, 'right': test_df.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

print(type(Y_train))
print(Y_train.shape)

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# # Make sure everything is ok
assert len(X_train['left']) == len(X_train['right'])
assert len(X_train['left']) == len(Y_train)

# X_train['left']/X_train['right'] is a list of str (m, l)
# Y_train is numpy ndarray (m,)

Y_net_train = {'label' : Y_train}
Y_net_validation = {'label' : Y_validation}


# In[24]:


class Siamese(gluon.HybridBlock):
    def __init__(self, embedding_dim, **kwargs):
        super(Siamese, self).__init__(**kwargs)
        self.encoder = gluon.rnn.LSTM(50,
                                      bidirectional=True, input_size=embedding_dim)
        self.dropout = gluon.nn.Dropout(0.3)
        self.dense = gluon.nn.Dense(32, activation="relu")
     
    def hybrid_forward(self, F, input0, input1):
        out0emb = input0
        out0 = self.encoder(out0emb)
        out1emb = input1
        out1 = self.encoder(out1emb)
        out0 = self.dense(self.dropout(out0))
        out1 = self.dense(self.dropout(out1))
        batchsize = out1.shape[0]
        xx = out0.reshape(batchsize, -1)
        yy = out1.reshape(batchsize, -1)
        manhattan_dis = F.exp(-F.sum(F.abs(xx - yy), axis=1, keepdims = True)) + 0.0001
        return manhattan_dis


class Embedding(gluon.HybridBlock):
    def __init__(self, input_dim, embedding_dim, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.embedding = nn.Embedding(input_dim, embedding_dim)
    
    def hybrid_forward(self, F, input):
        emb = self.embedding(input)
        return emb


#initialize the networknet
input_dim = len(vocabulary) + 1
ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
net1 = {c: Embedding(input_dim, embedding_dim // len(ctx)) for c in ctx}
net2 = Siamese(embedding_dim)
#check the gpus
print(ctx)
#initialize the network using the context of GPU
for (k, v) in net1.items():
    v.initialize(init=init.Normal(sigma=0.01), ctx=k)
net2.initialize(init=init.Normal(sigma=0.01), ctx=ctx)


# In[28]:


print(type(embeddings))
embeddings = np.split(embeddings, len(net1), axis=1)
for i, (k, v) in enumerate(net1.items()):
    print(embeddings[i].shape)
    gpuembeddings = (mx.nd.array(embeddings[i])).as_in_context(k)
    v.embedding.weight.set_data(gpuembeddings)


# In[29]:


trainer1 = {k: gluon.Trainer(v.collect_params(), 
                             'adadelta',
                             {'clip_gradient': 1.25}) for (k, v) in net1.items()}
trainer2 = gluon.Trainer(net2.collect_params(),
                         'adadelta',
                         {'clip_gradient': 1.25})
loss = gluon.loss.L2Loss()


# # Train

# In[30]:


def train_model(dataiter, epoch):
    train_loss = 0
    total_size = 0
    for i, batch in enumerate(dataiter):
        with mx.autograd.record():
            # iterate over the left and right question
            embs = []
            data_lists = []
            for k in range(2):
                print(batch.data[k].shape)
                embedding = [net1[c](batch.data[k].as_in_context(c)) for c in ctx]
                print(embedding[0].shape)
                embs.append(embedding)
                # data_list[i][j] is the ith part of embedding of sub-batch j (on gpu(j))
                # data_list[i][j] is of shape (B / len(ctx), embedding_dim / len(ctx))
                data_list = [gluon.utils.split_and_load(e, ctx, even_split=True) for e in embedding]
                print(data_list[0][0].shape)
                data_list = [mx.nd.concat(*[subemb[j] for subemb in data_list], dim=1) for j in range(len(ctx))]
                data_lists.append(data_list)
            data_list1, data_list2 = data_lists[0], data_lists[1]
            print(data_list1[0].shape)
            print(data_list2[0].shape)
            label_list = gluon.utils.split_and_load(batch.label[0], ctx, even_split=True)
            losses = [loss(net2(X1, X2), Y) for X1, X2, Y in zip(data_list1, data_list2, label_list)] 

#         print(embedding[0].grad.ctx)
#         print(embedding[1].grad.ctx)
#         print(embedding[0]._fresh_grad)
#         print(net1[ctx[0]].embedding.weight._data[0]._fresh_grad)
#         print(net2.dense.weight._data[0]._fresh_grad)
        for i, l in enumerate(losses):
            l.backward(retain_graph=True)
            for k, v in trainer1.items():
                v.step(batch.data[0].shape[0])
#         print(embedding[0]._fresh_grad)
#         print(net1[ctx[0]].embedding.weight._data[0]._fresh_grad)
#         print(net2.dense.weight._data[0]._fresh_grad)
        trainer2.step(batch.data[0].shape[0])
        total_size += batch.data[0].shape[0]
        train_loss += sum([l.sum().asscalar() for l in losses])
    mx.nd.waitall()
    return train_loss / total_size


seed = 0
mx.random.seed(seed)

training_loss = []
validation_loss = []
BATCH_SIZE = 1000
LEARNING_R = 0.001
EPOCHS = 10
THRESHOLD = 0.5
dataiter = mx.io.NDArrayIter(X_train, Y_net_train, BATCH_SIZE, True, last_batch_handle='discard')
valdataiter = mx.io.NDArrayIter(X_validation, Y_net_validation, BATCH_SIZE, True, last_batch_handle='discard')
accuracy_lst = []
for epoch in range(EPOCHS):
    dataiter.reset()
    valdataiter.reset()
    train_loss = train_model(dataiter, epoch)
    print(train_loss)


# # Validate

# In[9]:


def validate_model(valdataiter):
    test_loss = 0.
    total_size = 0
    auc_scores = []
    auc_labels = []
    for batch in valdataiter:
        # Do forward pass on a batch of validation data
        data_lists = []
        for k in range(2):
            embedding = [net1[c](batch.data[k].as_in_context(c)) for c in ctx]
            # data_list[i][j] is the ith part of embedding of sub-batch j (on gpu(j))
            # data_list[i][j] is of shape (B / len(ctx), embedding_dim / len(ctx))
            data_list = [gluon.utils.split_and_load(e, ctx, even_split=False) for e in embedding]
            data_list = [mx.nd.concat(*[subemb[j] for subemb in data_list], dim=1) for j in range(len(ctx))]
            data_lists.append(data_list)
        data_list1, data_list2 = data_lists[0], data_lists[1]
        labels = gluon.utils.split_and_load(batch.label[0], ctx, even_split=False)
        scores = [net2(X1, X2) for X1, X2 in zip(data_list1, data_list2)]
        pys = [loss(s, Y) for s, Y in zip(scores, labels)]
        test_loss += sum([l.sum().asscalar() for l in pys])
        total_size += batch.data[0].shape[0]
        # batch.label[0] is ndarray of shape (B,)
        # scores is a list of scores in different gpus
        auc_scores.extend([float(item.asscalar()) for score in scores for item in list(score)])
        auc_labels.extend([int(item.asscalar())   for label in labels for item in list(label)])
    auc = roc_auc_score(auc_labels, auc_scores)
    return test_loss / total_size, auc

valdataiter.reset()
val_loss, auc = validate_model(valdataiter)
print("{:>12} = {}".format("val_loss", val_loss))
print("{:>12} = {}".format("auc", auc))


# In[ ]:





# In[10]:


import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.utils import split_and_load

ctx = [mx.gpu(0), mx.gpu(1)]


class Net1(HybridBlock):
    def __init__(self):
        super(Net1, self).__init__()
        self.weight = self.params.get('weight', shape=(4, 2),
                                      dtype='float32', init=None, allow_deferred_init=False)

    def hybrid_forward(self, F, embedding, weight):
        return embedding * weight


class Net2(HybridBlock):
    def __init__(self):
        super(Net2, self).__init__()
        self.weight = self.params.get('weight', shape=(4, 2),
                                      dtype='float32', init=None, allow_deferred_init=False)

    def hybrid_forward(self, F, embedding, weight):
        return embedding * weight

net1 = Net1()
net2 = Net2()
net1.initialize(ctx=[mx.gpu(0)])
net2.initialize(ctx=[mx.gpu(1)])
net1.weight.set_data(mx.nd.ones((4, 2)) * 2)
net2.weight.set_data(mx.nd.ones((4, 2)) * 3)
with mx.autograd.record():
    embedding = mx.nd.ones((4, 2)).as_in_context(mx.gpu(0))
#     embedding.attach_grad()
    out1 = net1(embedding)
    out2 = net2(out1.as_in_context(mx.gpu(1)))
    loss = out2.sum()

loss.backward()
print(embedding.grad)
print(net1.weight._data[0].grad)
print(net2.weight._data[0].grad)
print(net1.weight._data[0]._fresh_grad)
print(net2.weight._data[0]._fresh_grad)

# for l in losses:
#     print("=" * 64)
#     l.backward()
#     print(embedding.grad)

# with mx.autograd.record():
#     a = mx.nd.ones((4, 2))
#     b = 2 * a
#     b.attach_grad()
#     c = b + 1
#     loss = c.sum()
# loss.backward()
# print(b.grad)


# In[11]:


import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
from mxnet.gluon.utils import split_and_load

ctx = [mx.gpu(0), mx.gpu(1)]


class Net1(HybridBlock):
    def __init__(self):
        super(Net1, self).__init__()
        self.weight = self.params.get('weight', shape=(4, 2),
                                      dtype='float32', init=None, allow_deferred_init=False)

    def hybrid_forward(self, F, embedding, weight):
        return embedding * weight  # * 2


class Net2(HybridBlock):
    def __init__(self):
        super(Net2, self).__init__()
        self.weight = self.params.get('weight', shape=(2, 2),
                                      dtype='float32', init=None, allow_deferred_init=False)

    def hybrid_forward(self, F, embedding, weight):
        return embedding * weight  # * 3

net1 = [Net1(), Net1()]
net2 = Net2()
net1[0].initialize(ctx=[mx.gpu(0)])
net1[1].initialize(ctx=[mx.gpu(1)])
net2.initialize(ctx=[mx.gpu(0), mx.gpu(1)])
net1[0].weight.set_data(mx.nd.ones((4, 2)) * 2)
net1[1].weight.set_data(mx.nd.ones((4, 2)) * 2)
net2.weight.set_data(mx.nd.ones((2, 2)) * 3)

trainer1 = [gluon.Trainer(net.collect_params(), 
                          'sgd') for net in net1]
trainer2 = gluon.Trainer(net2.collect_params(),
                         'sgd')

with mx.autograd.record():
    embedding = mx.nd.ones((4, 2)).as_in_context(mx.gpu(0))
#     embedding.attach_grad()
    out1_0 = net1[0](embedding)
    out1_1 = net1[1](out1_0.as_in_context(mx.gpu(1)))
#     out1_1.attach_grad()
#     losses = [out1_1.sum()]
    out2 = [net2(out1_1[0: 2].as_in_context(mx.gpu(0))), net2(out1_1[2:].as_in_context(mx.gpu(1)))]
    losses = [out2[0].sum(), out2[1].sum()]

for l in losses:
    print("=" * 64)
    l.backward(retain_graph=True)
#     out1_1.backward()
#     print(embedding.grad)
    print(net1[0].weight._data[0].grad)
    print(net1[1].weight._data[0].grad)
    print(net2.weight._data[0].grad)
#     print(net1[0].weight._data[0]._fresh_grad)
#     print(net1[1].weight._data[0]._fresh_grad)
#     print(net2.weight._data[0]._fresh_grad)
    print(net1[0].weight._data[0])
    print(net1[1].weight._data[0])
    print(net2.weight._data[0])
    trainer1[0].step(4)
    trainer1[1].step(4)
    print(net1[0].weight._data[0])
    print(net1[1].weight._data[0])
    print(net2.weight._data[0])
print("=" * 64)
trainer2.step(4)
print(net1[0].weight._data[0])
print(net1[1].weight._data[0])
print(net2.weight._data[0])


# In[ ]:




