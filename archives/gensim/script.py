import sys

# Tools
from utils import disp, data

# parsing
import json
from tokenizer import Tokenizer
from gensim import corpora, models

# storing data
from collections import Counter

root = data.getParent(__file__)
#filename = root + "/dataset/yelp_academic_dataset_review_training_small.json"
filename = sys.argv[1]

# Variables

# number of reviews a token has to appear to be kept
hardthreshold = 2

reviews_score = dict()
dictionary = corpora.Dictionary()
tok = Tokenizer(preserve_case=True)


def generateExample(filename):
  # extracting tokens
  for line in data.generateLine(filename):
    review = json.loads(line)
    tokens = tok.ngrams(review['text'], 1, 3, string=True)
    yield tokens

class Corpus(object):
  def __init__(self, filename, verbose=True):
    self.filename = filename
    print "> Loading file into dictionary"
    self.dictionary = corpora.Dictionary(tokens for tokens in generateExample(self.filename))
    print "Loaded"
    print "Filtering"
    self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
    print "Filtered"
    print "> Done"

  def __iter__(self):
    for tokens in generateExample(self.filename):
      yield self.dictionary.doc2bow(tokens)

corpus = Corpus(filename)
corpus.dictionary.save('/tmp/yelp-dict.data')
corpora.MmCorpus.serialize('/tmp/yelp-corpus.mm', corpus)

