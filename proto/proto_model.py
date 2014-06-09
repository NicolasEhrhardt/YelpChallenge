# Tools
from utils import data
import itertools

# generating reviews
from generator import generateYelpSentenceExample

# parsing
from gensim import corpora
from gensim.models import tfidfmodel

# vectors libs
from scipy import io
from scipy.spatial.distance import cosine
import numpy as np

root = data.getParent(__file__)

# huang data to be found on Huang website
wordrep_filename = root + "/huang/kept/prototypes_yelptraining.mat"

# yelp data
dataset_train_filename = root + "/dataset/yelp_academic_dataset_review_training.json"
dataset_holdout_filename = root + "/dataset/holdout/yelp_academic_dataset_review_holdout.json"

corpus_filename = root + '/computed/corpustrain.mm'
dict_filename = root + '/computed/corpustrain.dict'
tfidf_filename = root + '/computed/tfidf.model'
weights_filename = root + '/computed/proto_tfidf_noregul_weights.counter'
training_filename = root + '/computed/prototypes_sentence_cos_noregul_tfidf.pkl.gz'

corpus_train = corpora.MmCorpus(corpus_filename)
dictionary_train = corpora.Dictionary.load(dict_filename)
tfidf_model = tfidfmodel.TfidfModel.load(tfidf_filename)
corpus_train_tfidf = tfidf_model[corpus_train]
weights = data.load(weights_filename)


wordreps = io.loadmat(wordrep_filename)
prototypes = wordreps['We']

numbercosines = 5

def getValues(weights, filename):
  X = []
  Y = []
  leftouts = 0

  for sentence_tokens, stars in generateYelpSentenceExample(filename):
    sentence_proto = []
    sentence_weights = []

    for tokens in sentence_tokens:
      doc = dictionary_train.doc2bow(tokens)
      if len(doc) == 0:
        continue
      
      tokenids, freq = zip(*doc)
      w = [weights[tokenid] for tokenid in tokenids]
      tot = sum(w)
      w = [v / tot for v in w]
      # weight are already normalized at this point
      current_proto = np.multiply(prototypes[:, tokenids], w).sum(axis=1) 
      assert(len(current_proto) == 50)
      sentence_weights.append(tot)
      sentence_proto.append(current_proto)
    
    if len(sentence_proto) == 0:
      leftouts += 1
      continue

    pairs = itertools.combinations(sentence_proto, 2)
    dist = [0]*numbercosines
    for v1, v2 in pairs:
      dist.append(cosine(v1, v2))
    dist.sort(reverse=True)

    tot = sum(sentence_weights)
    w = [v / tot for v in sentence_weights]
    doc_proto = np.multiply(np.transpose(sentence_proto), w).sum(axis=1)
    doc_proto = np.append(doc_proto, dist[0: numbercosines])
    assert(len(doc_proto) == 50 + numbercosines)
    X.append([doc_proto])
    # change range for classification
    Y.append(stars-1)

  return (X, Y, leftouts)

print("> Get training matrix")
X_train, Y_train, leftouts_train = getValues(weights, dataset_train_filename)

print("> Get holdout matrix")
X_holdout, Y_holdout, leftouts_holdout = getValues(weights, dataset_holdout_filename)

print("- Number of docs leftout: %d" % (leftouts_holdout + leftouts_train))

print("< Document vectors loaded")

print("> Create training set")
X_train = np.concatenate(X_train, axis=0)
Y_train = np.array(Y_train)
assert(X_train.shape[0] == len(Y_train))

print("> Create holdout set")
X_holdout = np.concatenate(X_holdout, axis=0)
Y_holdout = np.array(Y_holdout)
assert(X_holdout.shape[0] == len(Y_holdout))

permut = np.random.permutation(len(Y_holdout))
print("- Create valid set")
X_valid = X_holdout[permut[0:len(permut)/2], :]
Y_valid = Y_holdout[permut[0:len(permut)/2]]
assert(X_valid.shape[0] == len(Y_valid))

print "- Create test subet"
X_test = X_holdout[permut[len(permut)/2::], :]
Y_test = Y_holdout[permut[len(permut)/2::]]
assert(X_test.shape[0] == len(Y_test))
print "< Test sets created"


data.save((
    (X_train, Y_train), 
    (X_valid, Y_valid),
    (X_test, Y_test),
), training_filename
)
