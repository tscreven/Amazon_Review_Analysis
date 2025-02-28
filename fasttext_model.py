from methods import *
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
import fasttext
import re
import argparse
import pandas as pd

BASE_NAMES = basenames()
VOTE_THRESHOLD = 1
TRAIN_TEST_SPLIT = 0.8
WORD_RE = re.compile(r'.*\w')

def tokenize(all_text):
    all_tokens = []
    for rev_text in all_text:
        tokens = word_tokenize(rev_text)
        token_string = ' '.join(token for token in tokens if WORD_RE.match(token))
        all_tokens.append(token_string)
    return all_tokens

def get_features(filename):
    text, labels = get_data(filename, VOTE_THRESHOLD)
    train_text, train_labels, test_text, test_labels = split_data(text, labels, TRAIN_TEST_SPLIT)
    train_tokens = tokenize(train_text)
    test_tokens = tokenize(test_text)

    return train_tokens, train_labels, test_tokens, test_labels

def generate_file(base_name, folder):
    train_file = open(f'{folder}/{base_name}_train.ft', 'w')
    train_tokens, train_labels, test_tokens, test_labels = get_features(get_decimated_name(base_name))
    for i in range(len(train_tokens)):
        train_file.write(f'__label__{train_labels[i]} {train_tokens[i]}\n')

    train_file.close()
    return test_tokens, test_labels

def run(base, folder):
    test_tokens, test_labels = generate_file(base, folder)
    model = fasttext.train_supervised(f'{folder}/{base}_train.ft', epoch=12, dim=125, verbose=0)
    output = model.predict(test_tokens)
    predictions = output[0]
    scores = output[1]
    processed_predicitons = [int(x[0][-1]) for x in predictions]
    processed_scores = [s[0].item() if p==1 else 1-s[0].item() for s,p in zip(scores, processed_predicitons)]
    results = [(l, s) for l,s in zip(test_labels, processed_scores)]
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top_size = int(len(results) * 0.05)
    top_percent = results[:top_size]
    true_positives = sum(1 for label, _ in top_percent if label == 1)
    precision_top_5_percent = true_positives / top_size
    return precision_top_5_percent

def main(print_all_results, excel_filename=None):
    precisions = []
    prior_helpful_probs = []
    train_sizes = []
    test_sizes = []
    delta = []
    ratio = []

    for base in BASE_NAMES:
        prec = run(base, 'TrainFiles')
        prob_helpful, _, train_text, test_text, _, _ = process_category(base, VOTE_THRESHOLD, TRAIN_TEST_SPLIT)
        diff_prec_prior = prec - prob_helpful
        ratio_prec_prior = prec / prob_helpful
        precisions.append(prec)
        prior_helpful_probs.append(prob_helpful)
        delta.append(diff_prec_prior)
        ratio.append(ratio_prec_prior)
        train_sizes.append(len(train_text))
        test_sizes.append(len(test_text))

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
    parser.add_argument('excel_filename',nargs='?',help='Name of Excel file. Optional argument, do not give filename if do not want results written into Excel sheet.')
    parser.add_argument('-print_all_results',help='Print extensive results to terminal.',action='store_true')
    args = parser.parse_args()

    if args.excel_filename is not None:
        main(args.print_all_results, args.excel_filename)
    else: main(args.print_all_results)
