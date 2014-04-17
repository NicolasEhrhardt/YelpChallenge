# DB
from pymongo import MongoClient

# Tools
from persutils import disp, data

# ML
import nltk
from collections import Counter
from math import log
#import numpy as np
#from scipy import stats
#import statsmodels.api as sm
#from pandas import *

# Connect to DB

client = MongoClient()
db = client.yelp
review_collection = db.yelp_academic_dataset_review

# Starting learning
trainRatio = .8
n = review_collection.count()
samplesize = 50000

print "Total reviews:", n
print "Sample size:", samplesize

reviews_feature = dict()
reviews_score = dict()
alltoken = dict()

# extracting tokens
i = 0
for review in review_collection.find():
  i += 1
  if i == samplesize: break

  disp.tempPrint(str(i))

  text = nltk.tokenize.word_tokenize(review['text'])
  score = float(review['stars'])
  
  reviews_feature[review['review_id']] = Counter(text)
  reviews_score[review['review_id']] = score
  
  for token in text:
    if token not in alltoken:
      alltoken[token] = Counter()

    alltoken[token][review['review_id']] += 1

print "End of full scan"

data.saveFile(alltoken, 'data/alltoken.pkl')
data.saveFile(reviews_feature, 'data/reviews_feature.pkl')
data.saveFile(reviews_score, 'data/reviews_score.pkl')
