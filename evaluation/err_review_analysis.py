from utils import data
from pybrain.tools.validation import ModuleValidator
import json
import copy


root = data.getParent(__file__)

"""
    Returns a dictionnary with the following properties
    res = get_regression_segregation();
    res.keys() = [0,1,2,3,4]
    res[key] = sorted list of tuples list_tuples
    list_tuples = ( error in this review, review idx in the holdout set of reviews)
"""
def get_regression_segregation():

    # Loading necessary data.
    file_path = "/computed/sentence_analysis_reg.pkl.gz"

    best_module, testData, Y_test = data.load( root + file_path )
    n_samples = len(Y_test)

    # Computing error and sorting and grouping errors by rating groups
    Y_pred = ModuleValidator.calculateModuleOutput(best_module, testData)
    
    
    error = [ Y_pred[i] - Y_test[i] for i in xrange(n_samples) ]
    err_and_revidx = zip( error, range(n_samples) )

    sorted_err = {0:[], 1:[], 2:[], 3:[], 4:[]}
    # for some reason the last n_samples/2 are corrupted and are not alligned to the reviews.
    for idx in range(n_samples/2):
        sorted_err[ Y_test[idx] ].append( err_and_revidx[idx] )
    for idx in range(5):
        sorted_err[idx] = sorted( sorted_err[idx] )
 
    return sorted_err


"""
    Returns the reviews in the holdout set. Each of the list has the following format
    {"stars": NUMBER, "review":TEXT} where NUMBER and TEXT are an integer and string respectively.
"""
def get_stars_and_review():
    file_test_reviews = "/dataset/holdout/yelp_academic_dataset_review_holdout.json"

    json_file = open( root + file_test_reviews )
    review_json_lines = json_file.readlines()
    json_file.close()

    star_and_review = []
    for review_json_line in review_json_lines:
        review = json.loads( review_json_line );
        star_and_review.append( {"stars": review["stars"], "review":review["text"] } );
    
    return star_and_review;

segregated_reviews = get_regression_segregation()
star_and_review = get_stars_and_review()

# We analyzed the reviews on which the model gives the biggest positive error, the biggest negative error and the smallest absolute error.
    reviews_analysis = { "pos": {0:[], 1:[], 2:[], 3:[], 4:[] },
                     "abs": {0:[], 1:[], 2:[], 3:[], 4:[] },
                     "neg": {0:[], 1:[], 2:[], 3:[], 4:[] } }

n_top = 10;
for i_star in range(5):
    n_elems = len( segregated_reviews[i_star] )
    top_neg = copy.deepcopy(segregated_reviews[i_star][0:n_top]);
    top_pos = copy.deepcopy(segregated_reviews[i_star][ -n_top::])
    abs_sorted = copy.deepcopy(segregated_reviews[i_star])
    for i in range( n_elems ):
        abs_sorted[i] = ( abs(abs_sorted[i][0]), abs_sorted[i][1] );
    abs_sorted = sorted(abs_sorted);
    top_abs = abs_sorted[0:n_top];

    for i in range(n_top):
        reviews_analysis["pos"][i_star].append( star_and_review[ top_pos[i][1] ]["review"] )
        reviews_analysis["neg"][i_star].append( star_and_review[ top_neg[i][1] ]["review"] )
        reviews_analysis["abs"][i_star].append( star_and_review[ top_abs[i][1] ]["review"] )    
