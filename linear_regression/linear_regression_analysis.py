# Tools 
from utils import data

root = data.getParent(__file__)
alphas = [0.001]#, 0.01, 0.1]

print "Optimizing with SGD"
for alpha in alphas: 
  print "Alpha = ", alpha
  weights, bias = data.loadFile(root + "/computed/linear_regression_weights_alpha_" + str(alpha) + ".pkl")
  print "\n".join(map(lambda t: str(t[1]) + " - " + t[0], weights.most_common(100)))
