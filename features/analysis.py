# Tools
from utils import disp, data

# parsing
import json
from tokenizer import Tokenizer

# storing data
from collections import Counter

# Display tools
from matplotlib import pyplot as plt

root = data.getParent(__file__)
filename = root + "/dataset/yelp_academic_dataset_review_training_sample.json"
print "Loading data", filename

alltoken = dict()
ntoken = list()
stars = list()
starsVote = dict()

tok = Tokenizer(preserve_case=False)
# extracting tokens
for line in data.generateLine(filename):
  review = json.loads(line)
  text = tok.tokenize(review['text'])
  score = float(review['stars'])
  stars.append(score)

  for vote in review['votes'].keys():
    if vote not in starsVote:
      starsVote[vote] = list()
    if review['votes'][vote] > 1:
      starsVote[vote].append(score)

  ntoken.append(len(text))

  for token in text:
    if token not in alltoken:
      alltoken[token] = Counter()

    alltoken[token][review['review_id']] += 1

print "End of full scan"

print "Stars distribution"
plt.figure(1)
plt.subplot(221)
plt.hist(stars, bins=5, normed=1, facecolor='k')
plt.title('Stars distribution')
plt.subplot(222)
plt.hist(starsVote['funny'], bins=5, normed=1, facecolor='r')
plt.title('Stars distribution (funny)')
plt.subplot(223)
plt.hist(starsVote['useful'], bins=5, normed=1, facecolor='g')
plt.title('Stars distribution (useful)')
plt.subplot(224)
plt.hist(starsVote['cool'], bins=5, normed=1, facecolor='b')
plt.title('Stars distribution (cool)')


print "Token per review"
plt.figure(2)
n, bins, patches = plt.hist(ntoken, 50, normed=1, facecolor='green', alpha=0.5)
plt.show()
