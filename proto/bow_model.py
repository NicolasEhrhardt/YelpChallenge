# Tools
from utils import data
from copy import deepcopy

# vector lib for loading matlab matrix
from scipy import io

# generating reviews
from generator import generateYelpReview, YelpCorpus

# parsing
from gensim import corpora
from gensim.models import tfidfmodel


root = data.getParent(__file__)

# huang data to be found on Huang website
vocab_filename = root +('/huang/kept/vocab_yelptraining.mat')

# yelp data
dataset_train_filename = root +('/dataset/yelp_academic_dataset_review_training.json')

corpus_filename = root + '/computed/corpustrain.mm'
dict_filename = root + '/computed/corpustrain.dict'
tfidf_filename = root + '/computed/tfidf.model'

# Hack for filtering
vocabmat = io.loadmat(vocab_filename)
vocab = vocabmat['vocab'][0]
token2id = {v[0]: k for k, v in enumerate(list(vocab))}
token2id_init = deepcopy(token2id)
dfs = {k:0 for k in xrange(len(vocab))}

print('> Loading data into dictionary_train')
dictionary_train = corpora.Dictionary()
# indexes
dictionary_train.token2id = token2id
dictionary_train.dfs = dfs

# docs
print('- Filtering')
print('- Number of tokens in dict before adding: %d' % len(dictionary_train.token2id))
dictionary_train.add_documents(generateYelpReview(dataset_train_filename))
print('- Number of tokens in dict after adding: %d' % len(dictionary_train.token2id))
print('- Number of tokens in dict used before prune: %d' % len(filter(lambda x: x>0, dictionary_train.dfs.values())))
dictionary_train.filter_tokens(bad_ids=None, good_ids=token2id_init.values())
print('- Number of tokens in dict after prune : %d' % len(dictionary_train.token2id))
print('- Number of tokens in dict used after prune: %d' % len(filter(lambda x: x>0, dictionary_train.dfs.values())))
# filtering
# this filtering has been done in the predload of huangs vector
# dictionary_train.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
print('< Loaded')

print('> Creating corpus_train structure')
corpus_train = YelpCorpus(dictionary_train, generateYelpReview, dataset_train_filename)

print('> Saving corpus to disk')
corpora.MmCorpus.serialize(corpus_filename, corpus_train)
corpus_train = corpora.MmCorpus(corpus_filename)
dictionary_train.save(dict_filename)

print('> Computing tfidf')
def unitvec(vec):
  length = 1.0 * sum(val for _, val in vec)
  if length != 1.0:
    return [(termid, val / length) for termid, val in vec]
  else:
    return list(vec)

tfidf_model = tfidfmodel.TfidfModel(corpus_train, normalize=True)
print('> Saving tfidf model')
tfidf_model.save(tfidf_filename)
