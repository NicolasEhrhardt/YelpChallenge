import pylab
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np

def error_classification(target, result, nclasses=5, save_as=''):
  error = [[] for i in range(nclasses)]
  for i in xrange(len(target)):
    err = (target[i] == result[i])
    error[target[i]].append(err)

  aggerror = {}
  for i in range(nclasses):
    aggerror[i+1] = float(sum(error[i])) / len(error[i])
  
  return aggerror

def error_classification_matrix(target, result, nclasses=5, save_as=''):
  """
  Plots a confustion matrix between the result and the target. Assumes that the labels are int starting from 0.

  :type target: list[int]
  :param target: list of correct labels

  :type result: list[int]
  :param result: list of predicted labels

  :type nclasses: int
  :param target: number of different labels (default 5)

  :type save_as: string
  :param save_as: file in which the figure should be saved. Leave empty if you just want to plot the result.
  """

  # Sanity checks 
  assert(len(target) == len(result))
  
  # Building confusiong matrix
  values = zip(target, result)
  freq = Counter(values)
  freq_mat = np.zeros((nclasses,nclasses))
  for k1, k2 in freq:
    freq_mat[k1, k2] = freq[(k1, k2)]

  colsum = freq_mat.sum(axis=1)
  confusion_mat = np.divide(freq_mat, colsum[:, np.newaxis])

  # Plot heat map
  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(confusion_mat, cmap=plt.cm.jet, interpolation='nearest')
  
  # Plot frequencies
  for x in xrange(nclasses):
    for y in xrange(nclasses):
      ax.annotate(
        str(round(confusion_mat[x, y], 1)), 
        xy=(y, x), 
        horizontalalignment='center',
        verticalalignment='center',
      )
 
  # Plot colorbar
  fig.colorbar(res) 
  plt.xticks(range(nclasses), range(nclasses))
  plt.xlabel('Predicted')
  plt.ylabel('Target')
  plt.yticks(range(nclasses), range(nclasses))
  
  # Save options
  if save_as =='':
    plt.show()
  else :
    fig.savefig(save_as)

def prob_dispersion(target, result, prob, nclasses=5, save_as=''):
  classprob = [[] for i in 2 * range(nclasses)]
  for i in xrange(len(result)):
    p = prob[i][result[i]]
    if result[i] == target[i]:
      classprob[2*result[i]].append(p)
    else:
      classprob[2*result[i] + 1].append(p)

  xlabels = [[str(i+1) + "-Good", str(i+1) + "-Bad"] for i in range(nclasses)]
  xlabels = reduce(list.__add__, xlabels, [])

  # Plotting the result
  fig = plt.figure()
  fig.suptitle('Probability distribution' , fontsize=20)
  plot = fig.add_subplot(111)
  pylab.boxplot(classprob)
  pylab.xticks(range(1, 2 * nclasses + 1), xlabels)
  plot.set_xlabel('Predicted' , fontsize = 16)
  plot.set_ylabel('Probabilities' , fontsize = 16)
  plot.tick_params(axis='both', which='major', labelsize=14)
  plot.tick_params(axis='both', which='minor', labelsize=8)
  
  # Save options
  if save_as =='':
    plt.show()
  else :
    fig.savefig(save_as)
