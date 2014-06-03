"""
    In this script we look at the distribution of the number of reviews by category to see which one to choose for the per category sLDA.
"""

from __future__ import print_function
import json
from utils import tokenizer, disp, data
from collections import Counter
import numpy as np

""" Files & Folders Parameters """
root = data.getParent("")

def categories_info():
    filepath = root + "/dataset/yelp_academic_dataset_business.json"

    """ Generate the count of reviews per category """
    business_file = open(filepath);
    lines_file = business_file.readlines();
    business_file.close();

    business_by_category = dict();
    categories_business_counts = Counter();
    categories_reviews_counts = Counter();

    for line_json in lines_file:
        business_dict = json.loads(line_json);
        business_id = business_dict["business_id"];
        categories_list = business_dict["categories"];

        for category in categories_list:
            if category not in business_by_category:
                business_by_category[category] = set();
            categories_business_counts[category] += 1
            categories_reviews_counts[category] += business_dict["review_count"];
            business_by_category[category].add(business_id);
        
    data.save(business_by_category, 'business_by_category.pkl.gz');
    data.save(categories_business_counts, 'categories_business_counts.pkl.gz');
    data.save(categories_reviews_counts, 'categories_reviews_counts.pkl.gz');

def extract_reviews_in_category(category="Pizza"):
    filepath = root + "/dataset/yelp_academic_dataset_review_randomize.json";
    savepath = root + "/dataset/yelp_academic_dataset_review_" +category.lower() + ".json";
    
    reviews_file = open(filepath);
    save_file = open(savepath, "w");
    lines_file = reviews_file.readlines();
    reviews_file.close();

    business_by_category = data.load("business_by_category.pkl.gz");
    #TODO check that the input category is present in the list!
    valid_businesses = business_by_category[category];

    i_review = 1;
    for line_json in lines_file:
        review_dict = json.loads(line_json);
        business_id = review_dict["business_id"];
        line_json = line_json.rstrip("\n")
        if business_id in valid_businesses:
            print(line_json, file=save_file);        
        i_review += 1;

    save_file.close();


