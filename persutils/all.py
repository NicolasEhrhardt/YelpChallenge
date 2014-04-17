import pickle
import sys
import numpy as np
# Library

# normalize counter to have a distribution
def normalize(c):
  tot = float(sum(c.values()))
  for k in c:
    c[k] = float(c[k]) / tot

# give the average of counter
def mean(c):
  f = 0
  for v in c:
    f += v
  return f / float(len(c))

def dist(v, arr):
  dist = []
  for vi in arr:
    dist.append(np.linalg.norm(vi-v))
  return min(dist)

def argDist(v, arr):
  minimumDist = float("infinity")
  vec = None
  for vi in arr:
    distance = np.linalg.norm(vi-v)
    if distance < minimumDist:
        minimumDist = distance
        vec = vi
  return vec

# var to file
def saveFile(words, outputFile):
  print "Saving in file: " + outputFile
  with open(outputFile, 'wb') as output:
    pickle.dump(words, output, pickle.HIGHEST_PROTOCOL)

# return saved dictionary
def loadFile(inputfile):
  if inputfile is not None:
    with open(inputfile, 'r') as inputFile:
      return pickle.load(inputFile)

def tempPrint(s):
  sys.stdout.write(s + "\r")
  sys.stdout.flush()
