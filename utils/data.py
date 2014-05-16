import disp
import pickle
from os import path

# var to file
def saveFile(variable, outputFile):
  print "Saving in file: " + outputFile
  with open(outputFile, 'wb') as output:
    pickle.dump(variable, output, pickle.HIGHEST_PROTOCOL)

# file to var
def loadFile(inputfile):
  if inputfile is not None:
    with open(inputfile, 'r') as inputFile:
      return pickle.load(inputFile)

def getParent(filename):
  return path.dirname(path.dirname(path.abspath(filename)))

def generateLine(filename, verbose = True):
  num = 0
  with open(filename) as f:
    for line in f:
      disp.tempPrint(str(num))
      num += 1
      yield line
