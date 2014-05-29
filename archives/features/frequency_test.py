import sys

# Tools
from utils import disp, data

# parsing
import json
from tokenizer import Tokenizer

# storing data
from collections import Counter

root = data.getParent(__file__)
#filename = root + "/dataset/yelp_academic_dataset_review_training_small.json"
filename = sys.argv[1]

# Variables

# number of reviews a token has to appear to be kept
hardthreshold = 2

print "> Loading data"
alltoken = data.loadFile(root + '/computed/alltoken.pkl')

print "> Scanning data"
print "Loading file", filename

reviews_feature = dict()
reviews_score = dict()

tok = Tokenizer(preserve_case=True)
# extracting tokens
for line in data.generateLine(filename):
  review = json.loads(line)
  reviewid = review['review_id']
  text = tok.ngrams(review['text'], 1, 3)
  score = int(review['stars'])
 
  # filtering tokens by the ones in the model
  text = filter(lambda k: k in alltoken, text)
  reviews_feature[reviewid] = Counter(text)
  reviews_score[reviewid] = score

print "> End of full scan"

print "> Saving"
data.saveFile(reviews_feature, root + "/computed/reviews_feature.pkl")
data.saveFile(reviews_score, root + "/computed/reviews_score.pkl")
