from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import  *
import numpy as np
from evaluation import plot
from utils import data
import gensim

def linear_regression(x_train,y_train,x_test,y_test,axis,figname):
    lin_reg_model = linear_model.LinearRegression();
    lin_reg_model.fit(x_train, y_train);
    y_pred = lin_reg_model.predict(x_test);

    RMSE = np.sqrt( np.mean( (y_test - y_pred) ** 2.0 ) );

    # Format required by the evaluation function
    dict_test = dict();
    dict_pred = dict();
    for ID in range( len(y_pred) ):
        dict_test[ID] = int(y_test[ID]);
        dict_pred[ID] = y_pred[ID];

    # Actually evaluating the results
    plot.error_boxplot( dict_test, dict_pred, 5, axis, figname);
    
    return RMSE
   


"""
 ##### sLDA Regression #####
"""

slda_x_train = np.genfromtxt("slda_x_train.txt", delimiter=" ");
slda_y_train = np.genfromtxt("slda_y_train.txt", delimiter=" ");
slda_x_test = np.genfromtxt("slda_x_test.txt", delimiter=" ");
slda_y_test = np.genfromtxt("slda_y_test.txt", delimiter=" ");

slda_RMSE = linear_regression(slda_x_train,
                                slda_y_train,
                                slda_x_test,
                                slda_y_test,
                                ["sLDA - Regression","Rating","Error"],
                                "slda_linReg_error.eps");

print "SLDA - RMSE : %1.4e" %slda_RMSE


"""
 ##### LDA Regression #####
"""

print "Loading LDA data"
lda_corpus_train = data.load("lda_corpus_train.pkl.gz");
lda_corpus_test = data.load("lda_corpus_test.pkl.gz");
word2id = data.load("slda_word2idx.pkl.gz");

print "Generating id2word"
id2word = dict();
for key in word2id:
    id2word[ word2id[key] ] = key;

print "lda training"
lda = gensim.models.ldamodel.LdaModel( lda_corpus_train, num_topics=20, id2word=id2word);

print "lda inference"
lda_x_train = lda.inference( lda_corpus_train );
lda_x_train = lda_x_train[0];
lda_y_train = slda_y_train;
lda_x_test = lda.inference( lda_corpus_test );
lda_x_test = lda_x_test[0]
lda_y_test = slda_y_test;

print "linear regression"
lda_RMSE =  linear_regression(lda_x_train,
                                lda_y_train,
                                lda_x_test,
                                lda_y_test,
                                ["LDA - Regression","Rating","Error"],
                                "lda_linReg_error.eps");

print "LDA - RMSE : %1.4e" %lda_RMSE;
