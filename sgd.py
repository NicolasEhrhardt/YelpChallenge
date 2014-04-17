
import numpy as np;
from collections import Counter;
import math;

def sgd(TFIDF,target, alpha = 0.04, epsilon = 0.01, nIterations = 1000):
  weights = Counter();

  # parameters 
  iIter = 0;
  bias = 0;

  # variables
  target_hat = Counter();  
  RMSE = 2 * epsilon;
  
  while iIter < nIterations and RMSE > epsilon:
    
    old_weights = weights.copy();

    # Update weights 
    for review in TFIDF:
      
      coeff = target[review];
      for token in TFIDF[review] :
        coeff -= weights[token] * TFIDF[review][token];
      coeff -= bias;

      for token in TFIDF[review] :
        weights[token] += alpha * coeff * TFIDF[review][token];
      
      # Constant term
      bias += alpha * coeff;
    # Compute the approximation error

    error = 0.;

    for review in TFIDF:
      errori = target[review];
      for token in weights:      
        errori -=   weights[token] * TFIDF[review][token];
      errori -= bias;

      error += errori ** 2.0;

    iIter += 1;
    
    RMSE = error / len(TFIDF);

    print '[',iIter,']'," - Error = ", RMSE;    
  
  return {"RMSE" : RMSE, "weights" : weights, "bias" : bias};
    
