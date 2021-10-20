# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:01:56 2021

@author: Yannik
"""

class LikeExtractor(FeatureExtractor):
    """Collects the number of likes for a Tweet"""
    
    def __init__(self, input_column):
        super.__init__(input_column, "likes")
        
    def _get_values(self, inputs):
        