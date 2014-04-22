# Tools 
from utils import data
from linear_regression_sgd import sgd

print "> Loading"
root = data.getParent(__file__)
print "loading review scores"
target = data.loadFile(root + '/computed/reviews_score.pkl')
print "loading TFIDF matrix"
TFIDF = data.loadFile(root + '/computed/TFIDF.pkl')

nReviews = len(target)

print "> Optimizing with SGD"
RMSE, weights, bias = sgd(TFIDF, target, alpha=0.001, epsilon=0.001)
print "Alpha = ", alpha, " -- RMSE = ", RMSE
data.saveFile((weights, bias), root + "/computed/linear_regression_weights.pkl")
