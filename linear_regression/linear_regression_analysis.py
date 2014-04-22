# Tools 
from utils import data
import evaluation

root = data.getParent(__file__)

print "Biggest weights"
weights, bias = data.loadFile(root + "/computed/linear_regression_weights.pkl")
print "\n".join(map(lambda t: str(t[1]) + " - " + str(t[0]), weights.most_common(100)))
