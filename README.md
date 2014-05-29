Dataset
=======

* Create data folders

        $ mkdir -p dataset computed saved

``dataset`` will contain the raw data, ``computed`` the values computed from this dataset, ``saved`` values computed for other models.

* Get the [data from yelp](http://www.yelp.com/dataset_challenge/) into `dataset/`

* Untar the data

        $ cd dataset
        $ tar xvf yelp_phoenix_academic_dataset.tgz

* Separate the dataset into a holdout and and training set.

        $ sort -R yelp_academic_dataset_review.json > yelp_academic_dataset_review_randomize.json
        $ head --lines=-50000 yelp_academic_dataset_review_randomize.json > yelp_academic_dataset_review_training.json
        $ head --lines=50000 yelp_academic_dataset_review_training.json > yelp_academic_dataset_review_training_small.json
        $ head --lines=5000 yelp_academic_dataset_review_training.json > yelp_academic_dataset_review_training_sample.json
        $ mkdir -p holdout
        $ tail --lines=50000 yelp_academic_dataset_review_randomize.json > holdout/yelp_academic_dataset_review_holdout.json
        $ tail --lines=5000 yelp_academic_dataset_review_randomize.json > holdout/yelp_academic_dataset_review_holdout_small.json

Setup
=====

Before doing anything, update your python path:

        $ export PYTHONPATH=$PYTHONPATH:$(pwd)

Folders used
============

- ``huang`` : train Huang's word vectors on dataset
- ``sLDA`` : assess accuracy of sLDA model
- ``proto`` : uses Huang's prototype to build a regression model
- ``utils`` : contains helpers used in the python scripts

Computed data
=============

We tried to implement it so that most computed data go into ``computed``. The utils script ``loadcomputed`` and ``savecomputed`` are to be used to save these data in separate folders depending on the type of analysis you are running. Save your work!
