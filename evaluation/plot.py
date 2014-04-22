from pylab import *
from matplotlib import pyplot as plt

"""
 Function : error_boxplot
 ------------------------
 This function plots a boxplot of the error between the result and the target. You can specify some parameters for the plot.
 @param : - target is a dictionnary of (ReviewID,Rating) pairs
          - results is a dictionnary of (ReviewID,PredictedRating) pairs
          - nclasses is the total number of different values taken by Ratings in the different entries of target ( DEFAULT = 5)
          - axis_argv is a list of parameters for the plot (axis_argv[0] -> title, axis_argv[1] -> xlabel, axis_argv[2] ->ylabel)
          - save_as specifies the file in which the figure should be saved. Do not provided this argument if you just want to plot the result.
"""
def error_boxplot(target, result, nclasses = 5, axis_argv = ['Error','Stars','Error'], save_as=''):

  error = [ [] for i in range(nclasses) ] 

  # Computing the error
  for key in target.keys():
    error[ target[key] - 1 ].append(target[key] - result[key])

  # Plotting the result
  fig = plt.figure()
  plot = fig.add_subplot(111)
  boxplot(error)
  fig.suptitle( axis_argv[0] , fontsize=20)
  plt.xlabel( axis_argv[1] , fontsize = 16)
  plt.ylabel( axis_argv[2] , fontsize = 16)
  plot.tick_params(axis='both', which='major', labelsize=14)
  plot.tick_params(axis='both', which='minor', labelsize=8)
  if save_as =='':
    plt.show()
  else :
    fig.savefig(save_as)


# Testing
if __name__ == "main":
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

  error_boxplot(target, predict,5,['Just Testings','yep','nope'],'tests.eps')
