# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import data_process

def entropy_calc(df, ensemble = None, weights = None):
    """
    This function helps to calculate the 
    entropy information of the features
    and returns them.
    
    Arguments:
    1. df: The data as a pandas dataframe.
    
    Returns:
    1. A sorted array of calculated entropy values.
    """
    
    entropy_dict = {}
    # dict to store entropy for each attribute
    df_group_class = df.groupby("class").count()
    df_group_class['weights'] = df.groupby("class").sum()['weights']
    # group the dataframe based on class and get its count and weight sum
    df_sum = df_group_class.sum()
    # get the sum of classes for each feature
    for col in df_group_class.columns:
        # iterate over each feature
        if col != 'weights':
            # check its the weight column
            entropy = 0.0
            # initial value of entropy
            for row in df_group_class.index:
                # iterate over each class
                prob = df_group_class.loc[row, 'weights']/df_sum[col]
                # probability of this class
                entropy += -prob*np.log2(prob)
                # calculate the entropy for each class and add
                entropy_dict[col] = round(entropy, 4)
                # round of the entropy value and add to a dict
        else:
            pass
        
    return pd.Series(entropy_dict).sort_values()
    
    
def information_gain_calc(df, feature, info_gain_dict):
    """
    This function helps to decide the best split for
    a given feature by calculating the information gain
    and updates the dictionary "info_gain_dict".
    Arguments:
    1. df: The input data as a pandas dataframe.
    2. feature: The feature for which we are evaulating.
    3. info_gain_dict: The dictionary which we will update.
    
    Returns:
    No returns.
    """
    
    feature_entropy_dict = {}
    # empty dictionary
    continous_bool = False
    # flag to check continous variable
    feature_values = df[feature].unique()
    # get the unique values of the feature
    if len(feature_values) == 0:
        # if we are out of data points
        return 0
    for threshold in feature_values:
        # iterate over each best feature values
        data_process.split_data_threshold(df, threshold, feature, 
                                          feature_entropy_dict, continous_bool)
        # function call to split data for this feature threshold
    best_threshold = round(min(feature_entropy_dict, key = feature_entropy_dict.get), 4)
    # get the best threshold value
    info_gain_dict[feature] = (best_threshold, feature_entropy_dict[best_threshold])
    # update the info_gain_dict which has the weighted entropy