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
  if i == 5: break

  disp.tempPrint(str(i))

  text = nltk.tokenize.word_tokenize(review['text'])
  score = float(review['stars'])
  
  reviews_feature[review['review_id']] = Counter(text)
  reviews_score[review['review_id']] = score
  alltoken.update(set(text))

#data.saveFile(alltoken, 'saved/alltoken.pkl')
#data.saveFile(reviews_score, 'saved/scores.pkl')
#data.saveFile(reviews_feature, 'saved/features_count.pkl')

# building dictionnary number <-> token
nTokens = len(alltoken)
tokens = sorted(list(alltoken))
positions = dict()
for i in xrange(len(tokens)):
  positions[tokens[i]] = i

columns = []
# normalizing vectors
for review_id in reviews_feature:
  review_vector = np.zeros(nTokens)
  for token in reviews_feature[review_id]:
    review_vector[positions[token]] = reviews_feature[review_id][token]

  columns.append(review_vector)

print "Number of token", nTokens

X = np.column_stack(columns)
Y = np.array(reviews_score.values())

print X.shape
print len(Y)
logit = sm.Logit(Y, X)
logit.fit()
print "r-squared:", r_value**2
