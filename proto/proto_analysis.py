# Tools
from utils import data

# vectors libs
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer

import numpy as np

root = data.getParent(__file__)

#training_filename = root + '/computed/prototypes_tfidf.pkl.gz'
#weights_filename = root + '/computed/bestweights_classification_noregul.pkl.gz'
training_filename = root + '/computed/prototypes_sentence_regul_tfidf.pkl.gz'
bestweights_filename = root + '/computed/proto_final_sentence_regul_tfidf.pkl.gz'

train, valid, test = data.load(training_filename)
bestweights = data.load(bestweights_filename)
nunits = 50

X_train, Y_train = train
X_valid, Y_valid = valid
X_test, Y_test = test

net = buildNetwork(nunits, nunits, 5, bias=True, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
# fast requires arac which is a pain in the butt to install but doable

def createDataset(X, Y):
  ds = ClassificationDataSet(nunits, 1, nb_classes=5)
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
predict = np.array([np.argmax(net.activate(x)) for x, _ in testData])
realerr = float(sum(np.equal(predict, Y_test)))/len(predict)
print('> Error on test set %f' % realerr)

from evaluation.classification import error_classification, error_classification_matrix, prob_dispersion

err = (Y_test == predict)
print('> Global accuracy: %f' % (float(sum(err))/len(err) ))

print('> Accuracy by category')
derr = error_classification(
  Y_test,
  predict
)
print(derr)

error_classification_matrix(
  Y_test,
  predict,
)

prob_dispersion(
  Y_test,
  predict,
  [net.activate(x) for x, _ in testData]
)
