import json
from utils import tokenizer, disp, data
from collections import Counter
import nltk
import numpy as np

""" Parameters """

root = '/home/umuhondo/Documents/Stanford/CS224U/yelp_challenge/YelpChallenge/dataset/'
filename = 'yelp_academic_dataset_review_training.json'
filepath = root + filename
n_reviews = 60000; #number of reviews to be considered 

""" A couple of useful initializations """
# Chris Potts tokenizer.
tok = tokenizer.Tokenizer( preserve_case=False );
# min and max ngram sizes
MIN_NGRAM = 1;
MAX_NGRAM = 1;
word_set = set(); # set of unique words
word2idx = dict(); # mapping from word to int representation of a word
ratings_list = [];
reviews_list = [];
reviews = []
data_list = [];
words_distr = dict();
words_counts = Counter();


""" PHASE 1 : Load file and get set of all words """
stopwords = nltk.corpus.stopwords.words('english');
print " PHASE 1 : Get all words "
loaded_file = open(filepath);
lines_file = loaded_file.readlines();
loaded_file.close();
i_review = 1;

# we randomly select n_reviews from the dataset 
permutation = np.random.permutation( len(lines_file) );
sample_reviews = permutation[0:n_reviews]; 

for idx_review in sample_reviews:
    line_json = lines_file[ idx_review ];
    review_dict = json.loads(line_json);
    tokens_list = tok.ngrams( review_dict['text'], MIN_NGRAM, MAX_NGRAM, string=True);
    rating      = review_dict['stars'];
    for token in tokens_list:
        if token not in stopwords:
          """
          if token not in words_distr:
            words_distr[token] = Counter({5:0, 4:0, 3:0, 2:0, 1:0}); 
          words_distr[token][rating] += 1;
          """
          words_counts[token] += 1;

    reviews_list.append( Counter(tokens_list) );
    ratings_list.append( review_dict['stars'] );
    word_set |=  set( tokens_list ) ;
    disp.tempPrint( str(i_review) );
    i_review+=1;


""" PHASE 2 : Word to int conversion """
filter_threshold = 0.00001 * (max( words_counts.values() ) * 1.0);
print " PHASE 2 : Word to int conversion "
i_word = 1;
for word in word_set:
    if ( words_counts[word] >= filter_threshold ):
        word2idx[word] = i_word;
        disp.tempPrint( str(i_word) );
        i_word+=1;
print "    Filtered. Before : %d words. After : %d" %(len(word_set), len(word2idx) );

""" PHASE 3 : Converting data to the right format """
print " PHASE 3 : Converting data to the right format "
i_review = 1;
for review in reviews_list:
    nwords = 0;
    data_line = "";
    for word in review:
        if word in word2idx:
            data_line += " " + str(word2idx[word]) + ":" + str( review[word]);
            nwords+=1;
    data_line += "\n";
    if (nwords != 0 ):
        data_line = str(nwords) + " " + data_line;
        data_list.append(data_line);
        disp.tempPrint( str(i_review) );
        i_review += 1

""" PHASE 4 : Save into right files """
print " PHASE 4 : Save into right files "
n_reviews = len( data_list );
idx_test = n_reviews* 5 / 6;

data_train = open('slda_data_train.txt','w');
label_train = open('slda_label_train.txt','w');

data_test = open('slda_data_test.txt','w');
label_test = open('slda_label_test.txt','w');

for i_review in range(idx_test):
    data_train.write( data_list[i_review] );
    label_train.write( str(ratings_list[i_review]) + "\n");

for i_review in range(idx_test, n_reviews):
    data_test.write( data_list[i_review] );
    label_test.write( str(ratings_list[i_review]) + "\n");

data_train.close();
data_test.close();
label_train.close();
label_test.close();

""" PHASE 5 : Save useful datastructures """
print " PHASE 5 : Save useful datastructures "
data.save(reviews_list,'slda_reviews.pkl.gz');
data.save(ratings_list,'slda_ratings.pkl.gz');
data.save(word2idx,'slda_word2idx.pkl.gz');



