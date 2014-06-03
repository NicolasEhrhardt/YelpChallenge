# Tools
from collections import Counter
import math
from utils import data

# generating reviews
from generator import generateYelpExample

# parsing
from gensim import corpora
from gensim.models import tfidfmodel

# SGD 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.sparse import csr_matrix

root = data.getParent(__file__)

# yelp data
dataset_train_filename = root + "/dataset/yelp_academic_dataset_review_training.json"

corpus_filename = '/tmp/yelp_proto_corpustrain.mm'
dict_filename = '/tmp/yelp_proto_corpustrain.dict'
tfidf_filename = '/tmp/yelp_proto_tfidf.model'
lin_reg_filename = '/tmp/yelp_proto_wf_lin_reg.model'
weights_filename = '/tmp/yelp_proto_wf_weights.counter'


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


"""-------- Distributional models for the words in the different documents ------"""

# Yields a tfidf representation of the reviews in the input filename
def generateTfidf(dataset_train_filename):
  for tokens, stars in generateYelpExample(dataset_train_filename):
    doc_tfidf = tfidf_model[dictionary_train.doc2bow(tokens)]
    if not len(doc_tfidf):
      continue
    
    yield stars, Counter(dict(doc_tfidf))


# Yields a document frequency representation of the reviews in the input filename
def generateWordFreq(dataset_train_filename):
    for tokens, stars in generateYelpExample(dataset_train_filename):
        doc_wordFreq = Counter(dict( dictionary_train.doc2bow(tokens) ) )
        if len(doc_wordFreq):
            doc_nwords = sum(doc_wordFreq.values()) * 1.
            for key in doc_wordFreq:
                doc_wordFreq[key] /= doc_nwords
            yield stars, doc_wordFreq;

"""-------- Tool for converting the output of the models above into the right matrix to
                                feed to the scipy SGD algorithm                         -----"""
# Generates a scipy.sparse.csr_matrix. This will be used in the scikit-learn implementation
# of the SGD algorithm. Hopefully improving the running speed of the algorithm
def generateScipyCSRMatrix(dataset_train_filename):
    # Inputs to the scipy.sparse.csr_matrix generator
    data = []
    indices = []
    indptr = []
    target = []

    idoc = 0
    idx = 0

    # Putting the data into the right format
    # In the line below, change the generator to the desired model above
    for star, doc in generateWordFreq(dataset_train_filename):
        indptr.append(idx)
        target.append(star)
        for key in doc:
            data.append( doc[key] )
            indices.append( key-1 )
            idx += 1

        idoc += 1
        
    indptr.append(idx)

    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)
    target = np.array(target)

    # generating csr matrix
    csr_train = csr_matrix( (data, indices, indptr), shape = (idoc, max(indices)+1 ) )
    return (csr_train,target)


"""
    # Wrapper around scipy's SGD Regressor
    @params : - alpha_p is the starting learning rate. It will be decreased as a function of the iteration number.
              - nEpoch is the number of iterations before stopping the SGD.
"""
def sgd(alpha_p=0.04, epsilon_p=0.01, alapcoeff_p=.01, nEpoch_p=1000):
    # We leave all the work to scikit learn
    # See here for a tutorial http://scikit-learn.org/stable/modules/sgd.html#sgd

    # loading the sparse data matrix
    print(" > generating the training matrix ")
    # X_train, Y_train = generateScipyCSRMatrix(dataset_train_filename)
    file_csr_train = "/tmp/csr_train.csr"
    # data.save((X_train,Y_train),file_csr_train);
    X_train, Y_train = data.load(file_csr_train);
    # fitting the linear regression model
    print(" > fitting the model ")
    lin_reg_model = linear_model.SGDRegressor(eta0=alpha_p, n_iter=nEpoch_p,shuffle=True, verbose=1,alpha=0);
    lin_reg_model.fit(X_train, Y_train)

    print(" > making the prediction ")
    Y_pred = lin_reg_model.predict(X_train)
    RMSE = mean_squared_error( Y_train, Y_pred )
    print("Training RMSE : " + str( RMSE ) );
    
    return lin_reg_model

# Actually performing the linear regression
alpha_opt_word_freq = 70.
lin_reg_model = sgd(alpha_p=alpha_opt_word_freq)

coeffs = lin_reg_model.coef_ #coefficients of the linear regression

# converting the coefficients to the weights format required by proto_model
weights = Counter()
for i in xrange( len(coeffs) ):
    weights[i+1] = coeffs[i]

# A good scientist always save their dataset
data.save(lin_reg_model, lin_reg_filename)
data.save(weights, weights_filename)

print "Done"
