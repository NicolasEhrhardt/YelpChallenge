Dataset and library install
===========================

Please see INSTALL.md


Dataset
=======

* Separate the dataset into a holdout and and training set.


        $ sort -R yelp_academic_dataset_review.json > yelp_academic_dataset_review_randomize.json
        $ head --lines=-50000 yelp_academic_dataset_review_randomize.json > yelp_academic_dataset_review_training.json
        $ mkdir -p holdout
        $ tail --lines=50000 yelp_academic_dataset_review_randomize.json > holdout/yelp_academic_dataset_review_holdout.json


Setup
=====


Before doing anything, update your python path:

        $ export PYTHONPATH=$PYTHONPATH:$(pwd)


Computing Features
==================


        $ python data_features.py

        $ pypy data_TFIDF.py


Machine learning
================

Linear regression
-----------------

        $ pypy linear_regression.py

        $ pypy linear_regression_analysis.py
