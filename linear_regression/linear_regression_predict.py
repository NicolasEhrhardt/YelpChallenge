# Tools 
from utils import data

def dot(csparse, c):
  r = 0
  for key in csparse:
    r += csparse[key] * c[key]
  return r

print "> Loading"
root = data.getParent(__file__)
print "loading review scores"
target = data.loadFile(root + '/computed/reviews_score.pkl')
print "loading TFIDF matrix"
TFIDF = data.loadFile(root + '/computed/TFIDF.pkl')

alphas = [0.001]#, 0.01, 0.1]

print "Optimizing with SGD"
for alpha in alphas: 
  print "Alpha = ", alpha
  weights, bias = data.loadFile(root + "/computed/linear_regression_weights_alpha_" + str(alpha) + ".pkl")

  predict = dict()
  for review in TFIDF:
    predict[review] = bias + dot(TFIDF[review], weights)

  data.saveFile(root + "/computed/linear_regression_predict.pkl")
