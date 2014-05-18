import sys

# Tools
from utils import disp, data
from copy import copy, deepcopy

# vector lib
from scipy import io

# parsing
import json
from utils.tokenizer import Tokenizer
from gensim import corpora, models

root = data.getParent(__file__)

# huang data to be found on Huang website
vocab_filename = root + "/dataset/huang/vocab.mat"
wordrep_filename = root + "/dataset/huang/wordreps_orig.mat"

# yelp data
dataset_train_filename = root + "/dataset/yelp_academic_dataset_review_training_sample.json"
dataset_holdout_filename = root + "/dataset/holdout/yelp_academic_dataset_review_holdout.json"

# Variables

vocabmat = io.loadmat(vocab_filename)
vocab = vocabmat['vocab'][0]
token2id = {v[0]: k for k, v in enumerate(list(vocab))}
token2id_init = deepcopy(token2id)
dfs = {k:0 for k in xrange(len(vocab))}

# number of reviews a token has to appear to be kept
hardthreshold = 2

reviews_score = dict()
dictionary_train = corpora.Dictionary()
tok = Tokenizer(preserve_case=False)

def generateExample(filename):
  # extracting tokens
  for line in data.generateLine(filename):
    review = json.loads(line)
    tokens = tok.tokenize(review['text'])
    stars = int(review['stars'])
    yield tokens, stars

def generateReview(filename):
  for tokens, stars in generateExample(filename):
    yield tokens

class Corpus(object):
  def __init__(self, dictionary_train, generator, filesource):
    self.dictionary_train = dictionary_train
    self.generator = generator
    self.filesource = filesource

  def __iter__(self):
    # using argument each time to rewind generator
    for tokens in self.generator(self.filesource):
      yield self.dictionary_train.doc2bow(tokens)

print "> Loading data into dictionary_train"
dictionary_train = corpora.Dictionary()
# indexes
dictionary_train.token2id = token2id
dictionary_train.dfs = dfs
# docs
print "- Filtering"
print "- Number of tokens in dict before adding:", len(dictionary_train.token2id)
dictionary_train.add_documents(generateReview(dataset_train_filename))
print "- Number of tokens in dict after adding:", len(dictionary_train.token2id)
print "- Number of tokens in dict used before prune:", len(filter(lambda x: x>0, dictionary_train.dfs.values()))
dictionary_train.filter_tokens(bad_ids=None, good_ids=token2id_init.values())
print "- Number of tokens in dict after prune:", len(dictionary_train.token2id)
print "- Number of tokens in dict used after prune:", len(filter(lambda x: x>0, dictionary_train.dfs.values()))
# filtering
#dictionary_train.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
print "< Loaded"

print "> Creating corpus_train structure"
corpus_train = Corpus(dictionary_train, generateReview, dataset_train_filename)
corpora.MmCorpus.serialize('/tmp/yelp-huang-corpus_train.mm', corpus_train)

from operator import itemgetter

from gensim.models import tfidfmodel
import numpy as np

print "> Computing tfidf"
tfidf_model = tfidfmodel.TfidfModel(corpus_train, normalize=True)
wordreps = io.loadmat(wordrep_filename)
prototypes = wordreps['oWe']
corpus_train_tfidf = tfidf_model[corpus_train]

print "> Get document vectors"
X_train = []
Y_train = []
X_holdout = []
Y_holdout = []
leftout = []

print "- Training set"
i = 0
for tokens, stars in generateExample(dataset_train_filename):
  i += 1
  disp.tempPrint(str(i))
  doc_tfidf = tfidf_model[dictionary_train.doc2bow(tokens)]
  if not len(doc_tfidf):
    leftout.append(i)
    continue

  tokenids, weights = zip(*doc_tfidf)
  # weight are already normalized at this point

  doc_proto = np.multiply(prototypes[:, tokenids], weights).sum(axis=1)
  assert(len(doc_proto) == 50)
  X_train.append([doc_proto])
  # change range for classification
  Y_train.append(stars-1)

print "- Test set"
i = 0
for tokens, stars in generateExample(dataset_holdout_filename):
  i += 1
  disp.tempPrint(str(i))
  doc_tfidf = tfidf_model[dictionary_train.doc2bow(tokens)]
  if not len(doc_tfidf):
    leftout.append(i)
    continue

  # weight are already normalized at this point
  tokenids, weights = zip(*doc_tfidf)

  # compute prototype average
  doc_proto = np.multiply(prototypes[:, tokenids], weights).sum(axis=1)
  assert(len(doc_proto) == 50)
  X_holdout.append([doc_proto])
  # change range for classification
  Y_holdout.append(stars-1)

print "- Number of docs tokens leftout: ", len(leftout)

print "< Document vectors loaded"

print "> Create training set"
X_train = np.concatenate(X_train, axis=0)
Y_train = np.array(Y_train)
assert(X_train.shape[0] == len(Y_train))

print "> Create holdout set"
X_holdout = np.concatenate(X_holdout, axis=0)
Y_holdout = np.array(Y_holdout)
assert(X_holdout.shape[0] == len(Y_holdout))

permut = np.random.permutation(len(Y_holdout))
print "- Create valid set"
X_valid = X_holdout[permut[0:len(permut)/2], :]
Y_valid = Y_holdout[permut[0:len(permut)/2]]
assert(X_valid.shape[0] == len(Y_valid))

print "- Create test subet"
X_test = X_holdout[permut[len(permut)/2::], :]
Y_test = Y_holdout[permut[len(permut)/2::]]
assert(X_test.shape[0] == len(Y_test))
print "< Test sets created"


#data.save((
#    (X_train, Y_train), 
#    (X_valid, Y_valid),
#    (X_test, Y_test),
#), "data.pkl.gz"
#)
