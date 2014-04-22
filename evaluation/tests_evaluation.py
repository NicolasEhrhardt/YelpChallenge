from results_evaluation import error_boxplot
from pylab import *

means = [1, 2, 3, 4, 5];
nreviews = [6, 5,2,3,4];
stds = [1,0.5,3,0.3,0.07];

target = {};
predict = {};

for imean in means:
  for irev in xrange(nreviews[imean-1]):
    key =  str(imean) + '_' + str(irev);
    target[ key ] = imean;
    predict[ key ] = imean + rand() + randn()*stds[imean - 1];

error_boxplot(target, predict,5,['Just Testings','yep','nope'],'tests.eps');
