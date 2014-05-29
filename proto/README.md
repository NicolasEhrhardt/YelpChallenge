Order of execution
==================

        - ``bow_model`` tokenizes each review and creates a model + dictionary which returns a sparse vector of variable dimension for each review (token: frequency) or (token: tfidf score) or ..? pma? The token indexes are consistent with the indexes of a huang representation. (see huang folder)

        - ``bow_regression`` trains a linear regression using the output of the previous model. it outputs a weight vector.

        - ``proto_model`` for each review combines the Huang vector prototype using a weighted average. By default we are using the linear regression weight but the tfidf weights can be used as well.

        - ``proto_regression`` trains a single hidden layer neural network using the prototypes output by the previous script.
