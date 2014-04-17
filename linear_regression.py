# Tools 
from persutils import data
from sgd import sgd;

print "loading review scores"
target = data.loadFile('data/reviews_score.pkl');
print "loading TFIDF matrix"
TFIDF = data.loadFile('data/TFIDF.pkl');

nReviews = len(target);

alphas = [0.001, 0.01, 0.1];

print "Optimizing with SGD"
for alpha in alphas : 
  results = sgd(TFIDF, target, alpha);
  print "Alpha = ", alpha, " -- RMSE = ", results['RMSE'];
