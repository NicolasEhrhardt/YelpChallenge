from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel

corpus = corpora.MmCorpus.load('/tmp/yelp-huang-corpus.mm')

tfidf = TfidfModel(corpus)


