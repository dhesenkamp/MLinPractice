#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.stopword_remover import StopwordRemover
from code.preprocessing.lemmatizer import Lemmatizer
from code.util import COLUMN_TWEET, SUFFIX_TOKENIZED, SUFFIX_STOPWORDS, SUFFIX_LEMMATIZED, SUFFIX_PUNCTUATION


# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")

# <--- Preprocessing steps --->
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("--punctuation_input", help = "input column to remove punctuation from", default = COLUMN_TWEET)
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET)
parser.add_argument("-s", "--stopwords", action = "store_true", help = "remove common stopwords")
parser.add_argument("--stopwords_input", help = "input column to remove stopwords from", default = COLUMN_TWEET)
parser.add_argument("-l", "--lemmatize", action = "store_true", help = "change words into their lemma based on WordNet")
parser.add_argument("--lemmatize_input", help = "input column to lemmatize", default = COLUMN_TWEET)

# <--- File export --->
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []

if args.stopwords:
    preprocessors.append(StopwordRemover(args.stopwords_input, args.stopwords_input + SUFFIX_STOPWORDS))

if args.punctuation:
    preprocessors.append(PunctuationRemover(args.punctuation_input, args.punctuation_input + SUFFIX_PUNCTUATION))

if args.tokenize:
    preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))

if args.lemmatize:
    preprocessors.append(Lemmatizer(args.lemmatize_input, args.lemmatize_input + SUFFIX_LEMMATIZED))

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)    