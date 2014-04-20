# Tools
from utils import disp, data

# ML
from collections import Counter
from math import log

print "Loading data"

alltoken = data.loadFile('../computed/alltoken.pkl')
reviews_feature = data.loadFile('../computed/reviews_feature.pkl')
reviews_score = data.loadFile('../computed/reviews_score.pkl')

n = len(reviews_score)

print "Total reviews:", n

# TF-IDF
print "Computing TF"
TF = dict()
i = 0
for review in reviews_feature:
  i += 1
  disp.tempPrint(str(i))
  TF[review] = Counter()
  for token in reviews_feature[review]:
    TF[review][token] = float(reviews_feature[review][token]) / float(max(reviews_feature[review].values()))

print "Computing IDF"
IDF = dict()
i = 0
for token in alltoken:
  i += 1
  disp.tempPrint(str(i))
  IDF[token] = log(float(n) / float(len(alltoken[token])))

print "Computing TFIDF"
TFIDF = dict()
i = 0
for review in reviews_feature:
  i += 1
  disp.tempPrint(str(i))
  TFIDF[review] = Counter()
  for token in reviews_feature[review]:
    TFIDF[review][token] = TF[review][token] * IDF[token]

# SGD linear regression

data.saveFile(TFIDF, '../computed/TFIDF.pkl')
