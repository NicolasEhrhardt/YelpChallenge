from utils import data
from utils.tokenizer import Tokenizer
import json

def generateYelpSentenceExample(filename):
  tok = Tokenizer(preserve_case=False)
  # extracting tokens
  for line in data.generateLine(filename):
    review = json.loads(line)
    tokens = tok.sentence_tokenize(review['text'])
    stars = int(review['stars'])
    yield tokens, stars

def generateYelpExample(filename):
  tok = Tokenizer(preserve_case=False)
  # extracting tokens
  for line in data.generateLine(filename):
    review = json.loads(line)
    tokens = tok.tokenize(review['text'])
    stars = int(review['stars'])
    yield tokens, stars

def generateYelpReview(filename):
  for tokens, stars in generateYelpExample(filename):
    yield tokens

class YelpCorpus(object):
  def __init__(self, dictionary_train, generator, filesource):
    self.dictionary_train = dictionary_train
    self.generator = generator
    self.filesource = filesource

  def __iter__(self):
    # using argument each time to rewind generator
    for tokens in self.generator(self.filesource):
      yield self.dictionary_train.doc2bow(tokens)
