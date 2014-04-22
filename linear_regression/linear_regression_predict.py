# Tools 
from utils import data
from collections import Counter

def dot(csparse, c):
  r = 0
  for key in csparse:
    r += csparse[key] * c[key]
  return r

print "> Loading"
root = data.getParent(__file__)
print "loading TFIDF matrix"
TFIDF = data.loadFile(root + '/computed/TFIDF.pkl')
print "loading weights"
weights, bias = data.loadFile(root + "/computed/linear_regression_weights.pkl")

predict = dict()
for review in TFIDF:
  predict[review] = bias + dot(TFIDF[review], weights)

data.saveFile(predict, root + "/computed/linear_regression_predict.pkl")
