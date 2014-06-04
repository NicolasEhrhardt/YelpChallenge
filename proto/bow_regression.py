# Tools
from collections import Counter
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
dataset_train_filename = root + '/dataset/yelp_academic_dataset_review_training.json'

corpus_filename = root + '/computed/corpustrain.mm'
dict_filename = root + '/computed/corpustrain.dict'
tfidf_filename = root + '/computed/tfidf.model'
lin_reg_filename = root + '/computed/proto_tfidf_regul_lin_reg.model'
weights_filename = root + '/computed/proto_tfidf_regul_weights.counter'
file_csr_train = root + '/computed/proto_tfidf_regul_csrtrain.csr'

print('> Load data')
corpus_train = corpora.MmCorpus(corpus_filename)
dictionary_train = corpora.Dictionary.load(dict_filename)
tfidf_model = tfidfmodel.TfidfModel.load(tfidf_filename)
corpus_train_tfidf = tfidf_model[corpus_train]

##################################################################
# Distributional models for the words in the different documents #
##################################################################

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
            doc_nwords = float(sum(doc_wordFreq.values()))
            for key in doc_wordFreq:
                doc_wordFreq[key] /= doc_nwords
            yield stars, doc_wordFreq;

chosenGenerator = generateTfidf

#########
# Tools #
#########

# Generates a scipy.sparse.csr_matrix. This will be used in the scikit-learn implementation
# of the SGD algorithm. Hopefully improving the running speed of the algorithm

def generateScipyCSRMatrix(generator, dataset_train_filename):
    # Inputs to the scipy.sparse.csr_matrix generator
    data = []
    indices = []
    indptr = []
    target = []

    idoc = 0
    idx = 0

    # Putting the data into the right format
    # In the line below, change the generator to the desired model above
    for star, doc in generator(dataset_train_filename):
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

###########
# Process #
###########

print('> Generating training matrix ')
X_train, Y_train = generateScipyCSRMatrix(chosenGenerator, dataset_train_filename)
data.save((X_train,Y_train), file_csr_train);

# fitting the linear regression model
print('> Fitting the model ')
# Actually performing the linear regression
alpha_opt_word_freq = 70.
lin_reg_model = linear_model.SGDRegressor(
  eta0=0.04, # starting learning rate
  n_iter=300, # max number of epochs
  shuffle=True, 
  verbose=0, 
  alpha=0.000001, # regularization constant
);

lin_reg_model.fit(X_train, Y_train)
coeffs = lin_reg_model.coef_ # coefficients of the linear regression

print('> Predict on train set')
Y_pred = lin_reg_model.predict(X_train)
RMSE = mean_squared_error( Y_train, Y_pred )
print('- Training RMSE : ' + str( RMSE ) );

# converting the coefficients to the weights format required by proto_model (matlab matrix indexes start at 1...)
weights = Counter()
for i in xrange( len(coeffs) ):
    weights[i+1] = coeffs[i]

# A good scientist always save his dataset
data.save(lin_reg_model, lin_reg_filename)
data.save(weights, weights_filename)
