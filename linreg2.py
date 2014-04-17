# DB
from pymongo import MongoClient

# Tools
from persutils import disp, data

# ML
import nltk
from collections import Counter, OrderedDict
import numpy as np
from scipy import stats
import statsmodels.api as sm
from pandas import *

# Connect to DB

client = MongoClient()
db = client.yelp
review_collection = db.yelp_academic_dataset_review

# Starting learning
trainRatio = .8
n = review_collection.count()

print "Total reviews:", n

reviews_feature = OrderedDict()
reviews_score = OrderedDict()
alltoken = set()

# extracting tokens
i = 0
for review in review_collection.find():
  i += 1
  if i == 1000: break

  disp.tempPrint(str(i))

  text = nltk.tokenize.word_tokenize(review['text'])
  score = float(review['stars'])
  
  reviews_feature[review['review_id']] = Counter(text)
  reviews_score[review['review_id']] = score
  alltoken.update(set(text))

train = DataFrame(reviews_feature).transpose().fillna(0)
target = Series(reviews_score)

print len(alltoken)
print target.shape
print train.shape

# Computing TF-IDF

maxFreq = np.tile(train.max(axis=1), (train.shape[1], 1)).T
TF = .5 + .5 * np.divide(train, maxFreq)

DF = train.apply(lambda row: len(row.nonzero()[0]), axis=0)
IDF = np.log(np.divide(train.shape[0] ,DF))

TFIDF = np.multiply(TF, IDF)

print IDF.shape
print TF.shape
print TFIDF.shape
#print TFIDF.head

logit = sm.Logit(target, TFIDF)


# fit the model
result = logit.fit()

print result.summary()
