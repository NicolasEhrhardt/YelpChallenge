import pylab
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np

"""
Function : error_boxplot
------------------------
    This function plots a boxplot of the error between the result and the target. You can specify some parameters for the plot.
    @param : 
    - target is a dictionnary of (ReviewID,Rating) pairs
    - result is a dictionnary of (ReviewID,PredictedRating) pairs
    - nclasses is the total number of different values taken by Ratings in the different entries of target ( DEFAULT = 5)
    - axis_argv is a list of parameters for the plot (axis_argv[0] -> title, axis_argv[1] -> xlabel, axis_argv[2] ->ylabel)
    - save_as specifies the file in which the figure should be saved. Do not provided this argument if you just want to plot the result.
"""
def error_boxplot(target, result, nclasses=5, axis_argv=['Error','Stars','Error'], save_as=''):

  error = [ [] for i in range(nclasses) ] 

  # Computing the error
  for key in target.keys():
    error[ target[key] - 1 ].append(target[key] - result[key])

  # Plotting the result
  fig = plt.figure()
  plot = fig.add_subplot(111)
  pylab.boxplot(error)
  fig.suptitle( axis_argv[0] , fontsize=20)
  plt.xlabel( axis_argv[1] , fontsize = 16)
  plt.ylabel( axis_argv[2] , fontsize = 16)
  plot.tick_params(axis='both', which='major', labelsize=14)
  plot.tick_params(axis='both', which='minor', labelsize=8)
  
  # Save options
  if save_as =='':
    plt.show()
  else :
    fig.savefig(save_as)


def error_classification_matrix(target, result, nclasses=5, save_as=''):
  assert(len(target) == len(result))
  values = zip(target, result)
  freq = Counter(values)
  freq_mat = np.zeros((nclasses,nclasses))
  for k1, k2 in freq:
    freq_mat[k1, k2] = freq[(k1, k2)]

  colsum = freq_mat.sum(axis=1)
  confusion_mat = np.divide(freq_mat, colsum[:, np.newaxis])


  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(confusion_mat, cmap=plt.cm.jet, interpolation='nearest')
  
  for x in xrange(nclasses):
    for y in xrange(nclasses):
      ax.annotate(
        str(round(confusion_mat[x, y], 1)), 
        xy=(y, x), 
        horizontalalignment='center',
        verticalalignment='center',
      )
 
  cb = fig.colorbar(res) 
  plt.xticks(range(nclasses), range(nclasses))
  plt.xlabel('Predicted')
  plt.ylabel('Target')
  plt.yticks(range(nclasses), range(nclasses))
  
  # Save options
  if save_as =='':
    plt.show()
  else :
    fig.savefig(save_as)


