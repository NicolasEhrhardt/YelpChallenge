# DB
from pymongo import MongoClient

client = MongoClient()

db = client.yelp

# Checking the reviews are here
print db.collection_names()

review_collection = db.yelp_academic_dataset_review

# Print structure of a review
print review_collection.find_one()

# Print kept info
review = review_collection.find_one()

print "Text", review['text']
print "Stars", review['stars']
