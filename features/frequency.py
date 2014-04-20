# Tools
from utils import disp, data

# parsing
import json
from tokenizer import Tokenizer

# storing data
from collections import Counter

filename = "../dataset/yelp_academic_dataset_review_training.json"
print "Loading data", filename

reviews_feature = dict()
reviews_score = dict()
alltoken = dict()

tok = Tokenizer(preserve_case=False)
# extracting tokens
for line in data.generateLine(filename):
  review = json.loads(line)
  text = tok.tokenize(review['text'])
  score = float(review['stars'])
  
  reviews_feature[review['review_id']] = Counter(text)
  reviews_score[review['review_id']] = score
  
  for token in text:
    if token not in alltoken:
      alltoken[token] = Counter()

    alltoken[token][review['review_id']] += 1

print "End of full scan"

data.saveFile(alltoken, "../computed/alltoken.pkl")
data.saveFile(reviews_feature, "../computed/reviews_feature.pkl")
data.saveFile(reviews_score, "../computed/reviews_score.pkl")
