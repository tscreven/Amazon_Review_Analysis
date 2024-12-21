from methods import *
import math
import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import argparse


BASE_NAMES = basenames()
VOTE_THRESHOLD = 1
TRAIN_TEST_SPLIT = 0.8

def get_parameters(base):
    prob_helpful, prob_unhelpful, train_text, test_text, train_labels, test_labels = process_category(base, VOTE_THRESHOLD, TRAIN_TEST_SPLIT)
    helpful_fd = nltk.FreqDist()
    unhelpful_fd = nltk.FreqDist()
    for review_text, helpful in zip(train_text, train_labels):
        tokens = word_tokenize(review_text)
        if helpful: helpful_fd.update(tokens)
        else: unhelpful_fd.update(tokens)
    return helpful_fd, unhelpful_fd, prob_helpful, prob_unhelpful, test_text, test_labels, len(train_text), len(test_text)


def class_log_probability(prior_prob, fd, class_num_words, review_tokens, total_vocab_size):
    terms = [math.log(prior_prob)]
    for token in review_tokens:
        prob = math.log((fd[token] + 1) / (class_num_words + total_vocab_size))
        terms.append(prob)
    return sum(terms) # Python sum() over a list is more accurate than incremently adding.


def normalize_log_probabilities(helpful_log_prob, unhelpful_log_prob):
    helpful_log_prob -= unhelpful_log_prob
    try:
        helpful_prob = math.exp(helpful_log_prob)
    except:
        return 1.0
    return helpful_prob / (helpful_prob + 1)


def run(base):
    helpful_fd, unhelpful_fd, prob_helpful, prob_unhelpful, test_text, test_labels, train_length, test_length = get_parameters(base)
    vocabulary = set(helpful_fd.keys()).union(set(unhelpful_fd.keys()))
    total_vocab_size = len(vocabulary)
    num_correct = 0
    results = []
    helpful_class_size = sum(helpful_fd.values())
    unhelpful_class_size = sum(unhelpful_fd.values())
    for rev_text, label in zip(test_text, test_labels):
        tokens = word_tokenize(rev_text)
        unnorm_helpful_prob = class_log_probability(prob_helpful, helpful_fd, helpful_class_size, tokens, total_vocab_size)
        unnorm_unhelpful_prob = class_log_probability(prob_unhelpful, unhelpful_fd, unhelpful_class_size, tokens, total_vocab_size)
        helpful_prob = normalize_log_probabilities(unnorm_helpful_prob, unnorm_unhelpful_prob)
        prediction = unnorm_helpful_prob >= unnorm_unhelpful_prob
        results.append((helpful_prob, prediction, label))
        if unnorm_helpful_prob >= unnorm_unhelpful_prob and label == 1:
            num_correct += 1
        elif unnorm_unhelpful_prob > unnorm_helpful_prob and label == 0:
            num_correct += 1

    results = sorted(results, key=lambda x: x[0], reverse=True)
    top_size = int(len(results) * 0.05)
    top_percent = results[:top_size]
    true_positives = sum(1 for _, _, label in top_percent if label == 1)
    precision_top_5_percent = true_positives / top_size
    return precision_top_5_percent, prob_helpful, train_length, test_length

def main(print_all_results, excel_filename=None):
    precisions = []
    prior_helpful_probs = []
    train_sizes = []
    test_sizes = []
    delta = []
    ratio = []

    for base in BASE_NAMES:
        prec, prob_helpful, train_len, test_len = run(base)
        diff_prec_prior = prec - prob_helpful
        ratio_prec_prior = prec / prob_helpful
        precisions.append(prec)
        prior_helpful_probs.append(prob_helpful)
        delta.append(diff_prec_prior)
        ratio.append(ratio_prec_prior)
        train_sizes.append(train_len)
        test_sizes.append(test_len)
        if print_all_results:
            print(f'{base.strip('_5')}: prec = {prec} | helpfulness probability = {prob_helpful} | delta = {diff_prec_prior} | ratio = {ratio_prec_prior}')
        else:
             print(f'{base.strip('_5')}: prec = {prec}')
    
    if excel_filename is not None:
        data = {'Category': BASE_NAMES, 'Precision at 5': precisions, 
                'Probability Review is Helpful': prior_helpful_probs,
                'Rank to Prior Prob Difference': delta, 
                'Rank to Prior Ratio': ratio, 'Train Size': train_sizes,
                'Test Size': test_sizes}
        df = pd.DataFrame(data)
        df.to_excel(excel_filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Naive Bayes model.')
    parser.add_argument('excel_filename',nargs='?',help='Name of Excel file. If set -generate_excel, have to input a filename.')
    parser.add_argument('-generate_excel',help='Create Excel sheet for results.',action='store_true')
    parser.add_argument('-print_all_results',help='Print extensive results to terminal.',action='store_true')
    args = parser.parse_args()

    if args.generate_excel:
        main(args.print_all_results, args.excel_filename)
    else: main(args.print_all_results)