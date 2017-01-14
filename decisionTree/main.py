# -*- coding: utf-8 -*-

"""
Name: Anirudh K Muralidhar, anikamal@iu.edu
Partner: Arun Ram Sankaranarayanan, arunsank@iu.edu
Start date: 18th September, 2016
End date: 23th September, 2016
Objective: Implementation of a basic decision tree for machine learning
purpose.
"""

import tree
import testing
import evaluation
import numpy as np
import pandas as pd
from sys import argv

def mains(df, train_file, depth, test_file = None, ensemble = None):
    """
    This function helps to build the 
    decision tree and evaluates it.
    Arguments:
    1. train_file: The input train filename as string.
    2. depth: The depth of the decision tree.
    3. test_file: The test file as a string, but defaulted
    to None (Cross validation performed)
    4. weights: The weights with which the entropy is calculated, defaulted 
    to None.
    """
    
    dt_dict = tree.main(df, train_file, depth = depth, ensemble = ensemble)
    # function call for decision tree construction
    df_train = testing.test(train_file, dt_dict)
    # function call to evaluate the decision using train data
    df_test = testing.test(test_file, dt_dict)
    # func call to test on test data
    return df_train, df_test
    