from collections import Counter
from utils import data

f_word_assign = open('word-assignments.dat');

n_topics = 20;
dict_topics = dict();
for i in range(n_topics):
    dict_topics[i] = Counter();

print "idx2word"
word2idx = data.load('slda_word2idx.pkl.gz');
idx2word = dict();
for word in word2idx:
    idx2word[ word2idx[word] ] = word;

print "topics distr"
for line in f_word_assign.readlines():
    word_assigns = line.split();
    for word_assign in word_assigns[1:]:
        idx, topic = word_assign.split(':');        
        word = idx2word[ int(idx) ];
        dict_topics[int(topic)][word] += 1;
