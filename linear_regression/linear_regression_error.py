# Tools 
from utils import data
from evaluation import plot

print "> Loading"
root = data.getParent(__file__)
print "loading review scores"
target = data.loadFile(root + '/computed/reviews_score.pkl')
print "loading predicted scores"
predict = data.loadFile(root + '/computed/linear_regression_predict.pkl')

RMSE = 0
for review in target:
  RMSE += (target[review] - predict[review])**2
RMSE /= len(target)

print "RMSE:", RMSE

plot.error_boxplot(target, predict)
