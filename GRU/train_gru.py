import numpy as np
from gru import GRU
import time
import datetime
from utils import *

learning_rate=0.001
vocabulary_size=2000
embedding_dim=48
hidden_dim=128
nepochs=20
model_output_file="model_output_file.mof"
input_data_file="./data/reddit-comments-2015.csv"
print_every=25000

if not model_output_file:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, vocabulary_size, embedding_dim, hidden_dim)

# Load data
x_train, y_train, word2index, index2word = load_data(input_data_file, vocabulary_size)

model=GRU(vocabulary_size,hidden_dim=hidden_dim,bptt_truncate=-1)

t1=time.time()
model.sgd_step(x_train[10],y_train[10],learning_rate)
t2=time.time()
print "SGD Step time: %f (ms)" %((t2-t1)*1000)




def sgd_callback(model,num_examples_seen):
    loss=model.calculate_loss(x_train[:10000],y_train[:10000])
    print("train instance: ",num_examples_seen,"Loss:",loss)
    generate_sentences(model,10,index2word,word2index)


def train_with_sgd(model,X_train,y_train,learning_rate=0.001,nepoch=20,callback=None):
    num_examples_seen=0
    for epoch in range(nepochs):
        for i in np.random.permutation(len(y_train)):
            model.sgd_step(X_train[i],y_train[i],learning_rate)
            num_examples_seen+=1
            if num_examples_seen%print_every==0:
                callback(model,num_examples_seen)


for epoch in nepochs:
    train_with_sgd(model, x_train, y_train, learning_rate=learning_rate, nepoch=1, decay=0.9,
    callback_every=print_every, callback=sgd_callback)