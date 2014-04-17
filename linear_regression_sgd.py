from persutils import disp
from collections import Counter
import math

def dot(csparse, c):
  r = 0
  for key in csparse:
    r += csparse[key] * c[key]
  return r

def sgd(TFIDF,target, alpha = 0.04, epsilon = 0.01, nIterations = 1000):
  weights = Counter()

  # parameters 
  iIter = 0
  bias = 0

  # variables
  RMSE = 2 * epsilon
  
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

      for token in TFIDF[review] :
        weights[token] += alpha * coeff * TFIDF[review][token]

      # Constant term
      bias += alpha * coeff
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

    print '[',iIter,']'," - Error = ", RMSE    
    
    if abs(RMSE_old - RMSE) < epsilon:
      break

  return (RMSE, weights, bias)
    
