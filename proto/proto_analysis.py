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
weights_filename = root + '/computed/bestweights_classification_noregul.pkl.gz'

train, valid, test = data.load(training_filename)
bestweights = data.load(weights_filename)

X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test

net = buildNetwork(50, 50, 5, bias=True, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
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
trainer.module.params[:] = bestweights

print('> Test on holdout set')
print(trainer.testOnData(testData))

# hit this command if you want to save the weights:
# data.save(bestweights, root + 'computed/bestweights.plk.gz')
predict = np.array([np.argmax(net.activate(x)) for x, _ in testData])
realerr = float(sum(np.equal(predict, Y_test)))/len(predict)
print('> Error on test set %f' % realerr)

from evaluation import error_classification_matrix, prob_dispersion
error_classification_matrix(
  Y_test,
  predict,
)

prob_dispersion(
  Y_test,
  predict,
  [net.activate(x) for x, _ in testData]
)
