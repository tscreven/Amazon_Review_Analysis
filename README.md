# Linguistic Amazon Review Analysis
Linguistic analysis on a large dataset of Amazon reviews. Goal was twofold:
determine if text classification models can predict the likelihood of a review
being useful and identify linguistic differences in reviews across product
categories.  Amazon allows users to vote that a product review is helpful. A
straightforward technique to optimize the shopping experience is to show
reviews with the highest number of helpfulness votes first. However, this
heavily favors older reviews as newer reviews will not be seen by as many
people. The motivation of the first part of the project was to explore if text
classification models can pick out useful fresh reviews to show customers on
the front page. The motivation for the second part was to examine part of
speech usage to show differences and similarities between the way people write
reviews in different product categories.

Both models significantly outperformed random chance. Even in the worst case,
the Precision@5% metric is much higher than the helpfulness prior probability.
Naïve Bayes performs somewhat better than FastText. In the part of speech
analysis, there is a fascinating result: product categories which are similar
in function also are similar in POS usage in their product reviews.

Read the report document to get the full detailed results and explanations. All
the code used in this project is in the Github repository. Instructions on how
to obtain and process the data, and run the code are below.

## Dataset
Using the 2018 version of the Amazon Review Data compiled by the McAuley Lab at
UC San Diego. Dataset used in the following paper: 
* "Justifying recommendations using distantly-labeled reviews and fined-grained
aspects" authored by Jianmo Ni, Jiacheng Li, and Julian McAuley published in
the 2019 Conference on Empirical Methods in Natural Language Processing.
* Link to paper pdf: https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf
* Link to dataset: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

## Downloading and Decimating Data
Download and decimate 15 gigabytes of data.

**To Do**:<br>
* Choose a file location suitable to store 15 gigabytes of data.
* First, run `download.ipynb` in the chosen file location where the data is to
  be stored. 
* Second, run `decimate.ipynb` in the same file location as `download.ipynb`.
  Change value of variable `MAX_PER_CATEGORY` if you want to alter the maximum
  number of reviews randomly selected for each product category. 

In the directory you want to download the reviews dataset into, run every cell
in `download.ipynb`.  In the same directory, run every cell in `decimate.ipynb`.
<br>
`decimate.ipynb` reads each original category data file, and randomly chooses
at most `MAX_PER_CATEGORY` (currently 100,000) reviews in the file that
actually have review text. The notebook randomly shuffles reviews so one can
choose training and validation sets for any category just be splitting reviews.

## methods.py
Python file containing functions used in Naïve Bayes and `fastText` model
construction, and part of speech analysis. Read comments in file for detailed
explanations of each function.

**To Do**:<br>
* Change variable `PATH_TO_REVIEWS` to location containing json files of
  product reviews for each category. If json files are located in the
  same location you plan to use the data in, set string variable to: `"."`.
* Code removes duplicate reviews in every product category. If you want to keep
  duplicate reviews, change bool variable `remove_duplicates` to `False` in
  `get_data()`. `get_data()` used to gather data for both Naïve Bayes and
  `fastText` model.

## Naïve Bayes Model
For each product category, I built, trained, and tested a hand coded Naïve
Bayes model. It is a binary classifier which predicts the likelihood of a review
being helpful or unhelpful. The first 80% of the randomly sorted reviews in
each category was used for training and the last 20% for testing.  The Naïve Bayes
model used a bag of words technique with logarithmic add-one smoothing, and
normalization of the class probabilities. `nltk` is used for tokenization.
Examine `naive_bayes.py` and/or read the report for detailed explanation of
model construction and execution. Read report for results and analysis.

`python3 naive_bayes.py <excel_filename> <-print_all_results>`
* `<excel_filename>` is an optional argument for the path to the Excel file
  where results will be stored. The following statistics for each category
  model stored on Excel sheet: precision@5, prior helpfulness probability,
  difference between precision@5 and helpfulness probability, ratio between
  precision@5 and helpfulness probability, and number of reviews in the
  training set and testing set.
* `<-print_all_results>` is an optional argument to print each category model's
  precision@5, prior helpfulness probability, difference between precision@5
  and helpfulness probability, ratio between precision@5 and helpfulness
  probability, and number of reviews in the training set and testing set.

Mathematical explanation of normalization of class log probabilities: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

## fastText Model
For each product category, we built, trained, and tested a `fastText` model. It
is a binary classifier that predict the likelihood of a review being helpful or
unhelpful. The first 80% of the randomly sorted reviews in each category was
used for training and the rest for testing.  Special character strings are
removed (such as punctuation). After that, the model conducts supervised
training on the remaining review tokens. `nltk` is used for tokenization.
Examine `fasttext_model.py` and/or read the report for detailed explanation of
model construction and execution. Read report for results and analysis.

`python3 fasttext_model.py <excel_filename> <-print_all_results>`
* `<excel_filename>` is an optional argument for the path to the Excel file
  where results will be stored. The following statistics for each category
  model stored on Excel sheet: precision@5, prior helpfulness probability,
  difference between precision@5 and helpfulness probability, ratio between
  precision@5 and helpfulness probability, and number of reviews in the
  training set and testing set.
* `<-print_all_results>` is an optional argument to print each category model's
  precision@5, prior helpfulness probability, difference between precision@5
  and helpfulness probability, ratio between precision@5 and helpfulness
  probability, and number of reviews in the training set and testing set.

## Part of Speech Analysis
Run cells in `pos.ipynb` to conduct a part of speech analysis of the product
reviews in different categories. Reveals similarities and differences between
language use in the reviews between categories.  Uses a frequency distribution
table for all part of speech tags in all product categories. Then, graphs a
heatmap of this table showing the difference in part of speech usage in each
category’s reviews.  Then, uses a 2d embedding of the frequency distribution
vectors to visualize POS usage relationships between categories. Read report
for complete results and analysis.

**To Do**:<br>
* If first time running, set `GENERATE_POS_FILES` to `True` in the second cell.
  Afterwards, set it back to `False` because the computation to generate the
  part of speech tags are computationaly expensive.