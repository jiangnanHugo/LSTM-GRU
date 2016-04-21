import csv
import itertools
import numpy as np
import nltk
import time
import sys
import operator
import io
import array
from datetime import datetime

SENTENCE_START_TOKEN="SENTENCE_START"
SENTENCE_END_TOKEN="SENTENCE_END"
UNKNOWN_TOKEN="UNKNOWN_TOKEN"

def load_data(filename="data/reddit-comments-2015-08.csv",vocabulary_size=2000,mini_sent_characters=0):
    word2index=[]
    index2word=[]

    # Read the data and append "SENTENCE_START" and "SENTENCE_END" tokens
    print("Reading CSV file...")
    with open(filename,'rt')as f:
        reader=csv.reader(f,skipinitialspace=True)
        reader.next()
        # Split full comments into sentences.
        sentences=itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        # filter sentences
        sentences=[s for s in sentences if len(s) >= mini_sent_characters]
        sentences=[s for s in sentences if "http" not in s]
        # append "SENTENCE_START" and "SENTENCES_END"
        sentences=["%s %s %s" %(SENTENCE_START_TOKEN,x, SENTENCE_END_TOKEN) for x in sentences]
    print("Parsed %d sentences." %(len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences=[nltk.word_tokenize(sent) for sent in sentences]

    # Count the word_frequencies
    word_freq=nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print ("Found %d unique words tokens." %(len(word_freq.items())))

    # Get the most common words and build index2word and word2index vectors
    vocab=sorted(word_freq.items(),key=lambda x:(x[1],x[0]),reverse=True)[:vocabulary_size-2]
    print ("Using vocabulary size %d." %vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times"%(vocab[-1][0],vocab[-1][1]))

    sorted_vocab=sorted(vocab,key=operator.itemgetter(1))
    index2word=["<MASK/>",UNKNOWN_TOKEN]+[x[0] for x in sorted_vocab]
    word2index=dict([(w,i) for i,w in enumerate(index2word)])

    # Replace all words not in our vocabulary with the unknown token
    for i,sent in enumerate(tokenized_sentences):
        tokenized_sentences[i]=[w if w in word2index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train=np.asarray([[word2index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train=np.asarray([[word2index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train,y_train,word2index,index2word

def print_sentences(s,index2word):
    sentence=[index2word[x] for x in s[1:-1]]
    print(" ".join(sentence))
    sys.stdout.flush()

def generate_sentence(model,index2word,word2index,min_length=5):
    # We start the sentence with the start token
    new_sentence=[word2index[SENTENCE_START_TOKEN]]
    # Erpeat unitl we get an end token
    while not new_sentence[-1] ==word2index[SENTENCE_END_TOKEN]:
        next_word_probs=model.predict(new_sentence)[-1]
        samples=np.random.multinomial(1,next_word_probs)
        sampled_word=np.argmax(samples)
        new_sentence.append(sampled_word)

        # Sometimes we get stuck if the sentence becomes too long.
        if len(new_sentence) >100 or sampled_word== word2index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence)<min_length:
        return None
    return new_sentence

def generate_sentences(model,n,index2word,word2index):
    for i in range(n):
        sent=None
        while not sent:
            sent=generate_sentence(model,index2word,word2index)
        print_sentences(sent,index2word)




