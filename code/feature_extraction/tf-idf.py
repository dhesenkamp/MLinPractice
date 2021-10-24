# -*- coding: utf-8 -*-
"""
Extracts tf-idf scores

Created on Sun Oct 24 23:05:33 2021

@author: Yannik
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from code.feature_extraction.feature_extractor import FeatureExtractor

class TfIdf(FeatureExtractor):
    """Extracts tf-idf scores"""
    
    def __init__(self, input_column):
        super.__init__([input_column], "{0}_Tf-Idf".format(input_column))
        
    def _get_values(self, inputs):
        vectorizer = TfidfVectorizer(max_features = 20)
        result = vectorizer.transform(inputs[0])
        return np.array(result)
        # compute cosine similarity?
        # need to safe the # most occuring words as keywords with get_feature_names() (get_feature_names_out(input_features)???)
        # fit_transform???
