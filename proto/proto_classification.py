# Tools
from utils import data
from math import isnan

# vectors libs
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer

import numpy as np

root = data.getParent(__file__)

training_filename = root + '/computed/prototypes_tfidf.pkl.gz'

train, valid, test = data.load(training_filename)

X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test

net = buildNetwork(50, 50, 5, bias=True, hiddenclass=TanhLayer, outclass=SoftmaxLayer, fast=True)
# fast requires arac which is a pain in the butt to install but doable

def createDataset(X, Y):
  ds = ClassificationDataSet(50, 1, nb_classes=5)
  ds.setField('input', X)
  ds.setField('target', np.asmatrix(Y).T)
  ds._convertToOneOfMany()
  return ds

trainingData = createDataset(X_train, Y_train)
validationData = createDataset(X_valid, Y_valid)
testData = createDataset(X_test, Y_test)

trainer = BackpropTrainer(net, trainingData) #, verbose=True)
#trainer.trainUntilConvergence(verbose=True, trainingData=trainingData, validationData=validationData)

maxEpochs = 100
continueEpochs = 10
convergence_threshold=10
trainingErrors = []
validationErrors = []
trainer.ds = trainingData
bestweights = trainer.module.params.copy()
bestverr = trainer.testOnData(validationData)
bestepoch = 0
trainingErrors = []
validationErrors = [bestverr]

print('> Training')

epochs = 0
while True:
  trainingError = trainer.train()
  validationError = trainer.testOnData(validationData)
  
  print('Validation error = %f - Training error = %f' % (validationError, trainingError))

  if isnan(trainingError) or isnan(validationError):
      raise Exception("Training produced NaN results")
  trainingErrors.append(trainingError)
  validationErrors.append(validationError)
  if epochs == 0 or validationErrors[-1] < bestverr:
      # one update is always done
      bestverr = validationErrors[-1]
      bestweights = trainer.module.params.copy()
      bestepoch = epochs

  if maxEpochs != None and epochs >= maxEpochs:
      trainer.module.params[:] = bestweights
      break
  epochs += 1

  if len(validationErrors) >= continueEpochs * 2:
      # have the validation errors started going up again?
      # compare the average of the last few to the previous few
      old = validationErrors[-continueEpochs * 2:-continueEpochs]
      new = validationErrors[-continueEpochs:]
      if min(new) > max(old):
          trainer.module.params[:] = bestweights
          break
      elif reduce(lambda x, y: x + (y - round(new[-1], convergence_threshold)), [round(y, convergence_threshold) for y in new]) == 0:
          trainer.module.params[:] = bestweights
          break

print('> Test on holdout set')
print(trainer.testOnData(testData))

# hit this command if you want to save the weights:
# data.save(bestweights, root + 'computed/bestweights.plk.gz')
predict = np.array([np.argmax(net.activate(x)) for x, _ in testData])
realerr = float(sum(np.equal(predict, Y_test)))/len(predict)
print('> Error on test set %f' % realerr)

from evaluation import error_classification_matrix
error_classification_matrix(
  Y_test,
  predict,
)
