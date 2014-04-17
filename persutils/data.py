import pickle

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
