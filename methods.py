import json
from collections import Counter
import nltk
nltk.download('tagsets_json')
from nltk import word_tokenize

#
# Return list of part of speech tags for all reviews in file_path. 
#
def label_pos(file_path):
    rev_texts, _ = get_data(file_path, -1)
    pos_tags = dict()
    for text in rev_texts:
        tokens = word_tokenize(text)
        output = nltk.pos_tag(tokens)
        for _, tag in output:
            if tag not in pos_tags:
                pos_tags[tag] = 1
            else:
                pos_tags[tag] += 1
    return pos_tags

#
# Extract review text from file_path. Each review is labeled as helpful (1) if
# the number of helpful votes equals or exceeds vote_threshold. Skip duplicate
# lines. Return list of review text and list of corresponding labels to those
# reviews.
#
def get_data(file_path, vote_threshold):
    rev_texts = []
    rev_help_class = []
    prev_lines = set()
    with open(file_path, 'r') as f:
        for line in f:
            if line in prev_lines: # Don't process duplicate lines
                continue
            prev_lines.add(line)
            review = json.loads(line)
            text = review['reviewText']
            rev_texts.append(text)
            rev_help_class.append(int('vote' in review and int(review['vote'].replace(',','')) >= vote_threshold))
    return rev_texts, rev_help_class

#
# Split text and labels into training and testing sets. Split is float value
# determning what percent of data goes into training set and testing set gets
# the remaining data.
#
def split_data(text, labels, split):
    index = int(split * len(text))
    train_text = text[0:index]
    train_labels = labels[0:index]
    test_text = text[index:]
    test_labels = labels[index:]
    return train_text, train_labels, test_text, test_labels

#
# Return path to decimated data file corresponding to base name. Data in
# separate folder due to size of files.
#
def get_decimated_name(base):
    return '/opt/lin127-project-data/decimated/'+ base+ '.json'

#  
# Return list of base names. Excluded Appliances_5 because only around 200 unique
# reviews.
#
def basenames():
    return [
    "AMAZON_FASHION_5",
    "All_Beauty_5",
    "Arts_Crafts_and_Sewing_5",
    "Automotive_5",
    "Books_5",
    "CDs_and_Vinyl_5",
    "Cell_Phones_and_Accessories_5",
    "Clothing_Shoes_and_Jewelry_5",
    "Digital_Music_5",
    "Electronics_5",
    "Gift_Cards_5",
    "Grocery_and_Gourmet_Food_5",
    "Home_and_Kitchen_5",
    "Industrial_and_Scientific_5",
    "Kindle_Store_5",
    "Luxury_Beauty_5",
    "Magazine_Subscriptions_5",
    "Movies_and_TV_5",
    "Musical_Instruments_5",
    "Office_Products_5",
    "Patio_Lawn_and_Garden_5",
    "Pet_Supplies_5",
    "Prime_Pantry_5",
    "Software_5",
    "Sports_and_Outdoors_5",
    "Tools_and_Home_Improvement_5",
    "Toys_and_Games_5",
    "Video_Games_5",
]
