# Tools 
from persutils import data
from linear_regression_sgd import sgd

print "loading review scores"
target = data.loadFile('data/reviews_score.pkl')
print "loading TFIDF matrix"
TFIDF = data.loadFile('data/TFIDF.pkl')

nReviews = len(target)

alphas = [0.001]#, 0.01, 0.1]

print "Optimizing with SGD"
for alpha in alphas: 
  RMSE, weights, bias = sgd(TFIDF, target, alpha, epsilon=0.001)
  print "Alpha = ", alpha, " -- RMSE = ", RMSE
  data.saveFile((weights, bias), "linear_regression_weights_alpha_" + str(alpha) + ".pkl")
