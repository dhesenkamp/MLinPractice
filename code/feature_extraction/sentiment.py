#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment analyser to extract compound sentiment from tweet.

Created on Wed Oct 13 11:12:51 2021

@author: dhesenkamp
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


class SentimentAnalyzer(FeatureExtractor):
    """Analyses input w.r.t. its sentiment using the VADER approach,
    taking into account punctuation, caps, emojis among others, making
    it a pristine choice for social media content."""
    
    
    def __init__(self, input_column):
        """Init sentiment analyser with given input column."""
        super().__init__([input_column], "{0}_sentiment".format(input_column))
    
    
    def _get_values(self, inputs):
        """Analyse sentiment and return compound value."""
        sia = SentimentIntensityAnalyzer()
        
        result = np.array([sia.polarity_scores(tweet)["compound"] + 1 for tweet in inputs[0]])
        result = result.reshape(-1, 1)
        return result
        