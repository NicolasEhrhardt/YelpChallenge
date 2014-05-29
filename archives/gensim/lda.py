from gensim import corpora, models

dictionary = corpora.Dictionary.load('/tmp/yelp-dict.data')
corpus = corpora.MmCorpus('/tmp/yelp-corpus.mm')
#tfidf = models.TfidfModel(corpus)
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
#log_ent = models.LogEntropyModel(corpus)
#print lda[[(0, 1), (1, 1), (2, 1)]]
#print lda.print_topics(20)

i = 0
for docbow in corpus:
  print lda[docbow]
  i += 1
  if i == 5:
    break
