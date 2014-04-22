from utils import disp
from collections import Counter
from copy import copy
import math

def dot(csparse, c):
  r = 0
  for key in csparse:
    r += csparse[key] * c[key]
  return r

def sgd(TFIDF, target, alpha=0.04, epsilon=0.01, alapcoeff=.01, nIterations=1000):
  weights = Counter()

  # parameters 
  iIter = 0
  bias = 0

  # variables
  RMSE = 2 * epsilon
  first = True

  #delta_weights = Counter()
  #delta_bias = 0

  while iIter < nIterations:
    print "Iter", iIter

    # Update weights 
    i = 0
    for review in TFIDF:
      i += 1
      disp.tempPrint(str(i))

      coeff = target[review]
      coeff -= dot(TFIDF[review], weights)
      coeff -= bias

      #old_delta_weights = copy(weights)
      #old_delta_bias = coeff
 
      for token in TFIDF[review] :
        weights[token] += alpha * coeff * TFIDF[review][token]
        #delta_weights[token] = coeff * TFIDF[review][token]
        
      # Constant term
      bias += alpha * coeff
      #delta_bias = coeff

      #if alapcoeff:
        #alpha = alpha * (1 + alapcoeff * (delta_bias * old_delta_bias + dot(old_delta_weights, delta_weights))) 
    # Compute the approximation error

    print "Computing error"
    error = 0.
    
    i = 0
    for review in TFIDF:
      i += 1
      disp.tempPrint(str(i))
      errori = target[review]
      errori -= dot(TFIDF[review], weights)
      errori -= bias

      error += errori ** 2.0

    iIter += 1
    
    RMSE_old = RMSE
    RMSE = error / len(TFIDF)
    if first:
      first = False
    else:
      alpha = alpha * (1 - abs(RMSE - RMSE_old) / RMSE)

    print '[',iIter,']'," - Error = ", RMSE    
    
    if abs(RMSE_old - RMSE) < epsilon:
      break

  return (RMSE, weights, bias)
    
