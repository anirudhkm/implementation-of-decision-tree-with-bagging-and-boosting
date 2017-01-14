# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd

def preprocess(df):
    """
    This function helps to do some
    preprocessing of the data
    and returns them.
    """
    
    df.rename(columns = {"bruises?-bruises":"class"}, inplace = True)
    # change the column name
    df.drop("bruises?-no", axis = 1, inplace = True)
    # drop the unwanted column
    return df

def decision_tree_predict(dt_dict, test_data_pd_series):
    """
    This function helps to predict the class label for
    the given input.
    Arguments:
    1. dt_dict: The decision tree as a dictionary.
    2. test_data_pd_series: The test data as a pandas series object.
    
    Returns:
    1. The class label predicted by the decision tree.
    """
    
    if isinstance(dt_dict, (np.int64, int)):
        # check we have a class label
        return dt_dict
        
    for node in dt_dict:
        # iterate over each node of the tree
        if test_data_pd_series[node[0]] == node[1][0]:
            # check if the value matches
            dt_dict = dt_dict[node]['equal']
            # traversing the tree
        else:
            dt_dict = dt_dict[node]['not_equal']
            # traversing of tree
    return decision_tree_predict(dt_dict, test_data_pd_series)
    # recursive function call
    
def test(test_file, dt_dict):
    """
    This function helps to read the test data
    file and iterates over each row and predicts the
    class label.
    Arguments:
    1. test_file: The test file on which we would like to 
    make predictions.
    2. dt_dict: The decision tree as a dictionary.
    """
    
    test_df = preprocess(pd.read_csv(test_file))
    # get the pandas dataframe for the test file
    predicted_class = []
    # empty list to store predicted labels
    for index in test_df.index:
        # iterate over each index of the test data
        row = test_df.loc[index]
        # get the corresponding row for the index value
        predicted_class.append(decision_tree_predict(dt_dict, row))
        # get the predicted label and append that to the list
    test_df['predict'] = predicted_class
    # add the predicted labels to the main pandas dataframe
    return test_df
    
    
    
    
    
