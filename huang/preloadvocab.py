# Tools
from utils import data
from collections import Counter

# parsing
import json
from utils.tokenizer import Tokenizer
from gensim import corpora

root = data.getParent(__file__)

# huang data to be found on Huang website
prevocab_filename = root + "/huang/trainEmb/data/vocab.txt"
predfs_filename = root + "/huang/trainEmb/data/df.txt"
precorpus_folder = root + "/huang/trainEmb/data/corpus/"
vocab_filename = root + "/dataset/huang/vocab.mat"
wordrep_filename = root + "/dataset/huang/wordreps_orig.mat"

# yelp data
dataset_train_filename = root + "/dataset/yelp_academic_dataset_review_training.json"

# generators
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
# docs
dictionary_train = corpora.Dictionary(generateReview(dataset_train_filename))
print "- Number of tokens", len(dictionary_train.token2id)
print "> Filter dictionary_train"
dictionary_train.filter_extremes(no_below=3, no_above=1.1, keep_n=190000)
print "- Number of tokens after filtering", len(dictionary_train.token2id)
corpus = Corpus(dictionary_train, generateReview, dataset_train_filename)
tf = Counter()

print "> Compute global term frequency"
ndoc = 0
j = 1
f = open(precorpus_folder + str(j) + '.txt', 'w')
for doc in corpus:
  ndoc += 1
  if ndoc % 1000 == 0:
    f.truncate()
    f.close()
    j += 1
    f = open(precorpus_folder + str(j) + '.txt', 'w')
  
  for tokenid, freq in doc:
    tf[tokenid] += freq
    f.write(str(tokenid + 1) + '\n')
  f.write('eeeoddd\n')

f.close()

print "> Saving to file"
maxInd = max(dictionary_train.token2id.values())
with open(prevocab_filename, 'w') as vf, open(predfs_filename, 'w') as df:
  df.write(str(ndoc) + '\n')

  for word, tokenid in dictionary_train.token2id.iteritems():
    vf.write(word + ' ' + str(tokenid + 1) + ' ' + str(tf[tokenid]) + '\n')
    df.write(str(dictionary_train.dfs[tokenid]) + '\n')

  df.write(str(maxInd + 2) + '\n')
  df.write(str(maxInd + 3))
  vf.write('<s> ' + str(maxInd + 2) + ' ' + str(ndoc) + '\n')
  vf.write('</s> ' + str(maxInd + 3) + ' ' + str(ndoc))
