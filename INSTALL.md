Install process for ubuntu.
===========================

Import data
-----------

* Get the [data from yelp](http://www.yelp.com/dataset_challenge/) into `data/`


* Untar the data

        $ tar xvf yelp_phoenix_academic_dataset.tgz


* Install [MongoDB](http://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/)


* Import the data into Mongo (in the database "yelp")

        $ mongoimport -d yelp yelp_academic_dataset_review.json

Machine learn things
--------------------

* Install scipy

        $ sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

        $ pip install patsy pandas statsmodel numpy
