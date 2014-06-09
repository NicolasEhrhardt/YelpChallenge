import pylab
from matplotlib import pyplot as plt

def error_boxplot(target, result, nclasses=5, axis_argv=['Error','Stars','Error'], save_as=''):
  """
  Plots a boxplot of the error between the result and the target.
  
  :type target: dict(ReviewID: Rating)
  :param target: is a dictionnary of (ReviewID,Rating) pairs

  :type result: dict(ReviewID: Rating)
  :param result: dictionnary of (ReviewID,PredictedRating) pairs
  
  :type nclasses: int
  :param nclasses: total number of different values taken by Ratings in the different entries of target ( DEFAULT = 5)
  
  :type axis_argv: list(String)
  :param axis_argv: list of parameters for the plot (axis_argv[0] -> title, axis_argv[1] -> xlabel, axis_argv[2] ->ylabel)
  
  :type save_as: string
  :param save_as: file in which the figure should be saved. Leave empty if you just want to plot the result.
  """
  error = [ [] for i in range(nclasses) ] 

  # Computing the error
  for key in xrange(len(target)):
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
