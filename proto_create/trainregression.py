# Tools
from collections import Counter
import math
from utils import data

# generating reviews
from generator import generateYelpExample

# parsing
from gensim import corpora
from gensim.models import tfidfmodel

root = data.getParent(__file__)

# yelp data
dataset_train_filename = root + "/dataset/yelp_academic_dataset_review_training.json"

corpus_filename = '/tmp/yelp_proto_corpustrain.mm'
dict_filename = '/tmp/yelp_proto_corpustrain.dict'
tfidf_filename = '/tmp/yelp_proto_tfidf.model'
weights_filename = '/tmp/yelp_proto_weights.counter'

corpus_train = corpora.MmCorpus(corpus_filename)
dictionary_train = corpora.Dictionary.load(dict_filename)
tfidf_model = tfidfmodel.TfidfModel.load(tfidf_filename)
corpus_train_tfidf = tfidf_model[corpus_train]

print "> Train regression"

def dot(csparse, c):
  r = 0
  for key in csparse:
    r += csparse[key] * c[key]
  return r

def generateTfidf(dataset_train_filename):
  for tokens, stars in generateYelpExample(dataset_train_filename):
    doc_tfidf = tfidf_model[dictionary_train.doc2bow(tokens)]
    if not len(doc_tfidf):
      continue
    
    yield stars, Counter(dict(doc_tfidf))

def sgd(alpha=0.04, epsilon=0.01, alapcoeff=.01, nEpoch=1000):
  weights = Counter()

  # parameters
  size = 0 # will be computed at first iteration
  epoch = 0
  bias = 0
  alphainit = alpha

  # variables
  RMSE = 2 * epsilon
  first = True

  #delta_weights = Counter()
  #delta_bias = 0

  while epoch < nEpoch:
    print("> Epoch %d" % epoch)

    # Update weights 
    for target, doc in generateTfidf(dataset_train_filename):
      if first:
        size += 1

      coeff = target
      coeff -= dot(doc, weights)
      coeff -= bias

      #old_delta_weights = copy(weights)
      #old_delta_bias = coeff
      for token in doc:
        weights[token] += alpha * coeff * doc[token]
        #delta_weights[token] = coeff * TFIDF[review][token]
        
      #for w in weights:
      #  if math.isnan(weights[w]):
      #    print "n"

      # Constant term
      bias += alpha * coeff
      #delta_bias = coeff

      #if alapcoeff:
        #alpha = alpha * (1 + alapcoeff * (delta_bias * old_delta_bias + dot(old_delta_weights, delta_weights))) 
    # Compute the approximation error

    error = 0.
    for target, doc in generateTfidf(dataset_train_filename):
      errori = target
      errori -= dot(doc, weights)
      errori -= bias

      error += errori ** 2.0

    epoch += 1
    
    RMSE_old = RMSE
    RMSE = error / size
    RMSE_delta = (RMSE_old - RMSE)

    if first:
      first = False
    else:
      alpha = alphainit / math.sqrt(epoch)

    print('Error = %f \nAlpha = %f \nImprovement = %f' % (RMSE, alpha, RMSE_delta))
    
    if abs(RMSE_delta) < epsilon:
      break

  return weights

weights = sgd()
data.save(weights, weights_filename)

print "Done"
