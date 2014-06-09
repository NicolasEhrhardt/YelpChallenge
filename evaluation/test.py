from regression import error_boxplot
from classification import error_classification_matrix
from pylab import *

#################
# Box plot test #
#################

means = [1, 2, 3, 4, 5]
nreviews = [6, 5,2,3,4]
stds = [1, 0.5, 3, 0.3, 0.07]

target = {}
predict = {}

for imean in means:
  for irev in xrange(nreviews[imean-1]):
    key =  str(imean) + '_' + str(irev)
    target[ key ] = imean
    predict[ key ] = imean + rand() + randn()*stds[imean - 1]

print("> Boxplot test")
error_boxplot(target, predict, 5, ['Just Testings','yep','nope'])


##################
# Confusion test #
##################

target = [0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
result = [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4, 4, 4]

print("> Confusion matrix test")
error_classification_matrix(target, result)

