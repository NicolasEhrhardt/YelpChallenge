"""
  Generates a training and test corpus to be used in gensim for 
  LDA stuff
"""

from utils import data

f_data_train = "slda_data_train.txt";
f_data_test = "slda_data_test.txt";

def get_corpus(filename):
    open_file = open(filename);
    lines = open_file.readlines();
    corpus = [];
    
    for line in lines:
        entries = line.split();
        doc_in_corpus = [];

        # First entry is ignored as it corresponds to the # of words
        for entry in entries[1:]:
            doc_in_corpus.append( tuple( [int(x) for x in entry.split(':') ] ) );
        corpus.append(doc_in_corpus);

    open_file.close();
    return corpus;


corpus_train = get_corpus( f_data_train );
corpus_test = get_corpus( f_data_test );

data.save(corpus_train, "lda_corpus_train.pkl.gz");
data.save(corpus_test, "lda_corpus_test.pkl.gz");

    
