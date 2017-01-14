# -*- coding: utf-8 -*-
from __future__ import division
import impurity 
    
def split_data_threshold(df, threshold, feature, feature_entropy_dict,
                         continous_bool):
    """
    This function helps to split the data for the 
    given threshold of the feature and calculates
    the entropy for the split threshold.
    
    Arguments:
    1. df: The input data as a pandas dataframe.
    2. threshold: The threshold value for which we want to test.
    3. feature: The attribute on which the data is being worked.
    4. feature_entropy_dict: The dictionary where we will update
    the entropy scores.
    
    Returns:
    No returns, since we just updating the dictionary, which is
    mutable, so changes are reflected back automatically where we called
    the function.
    """
    
    df_subset = df[[feature, "class", "weights"]]
    # get the subset of the main dataframe by taking only the desired feature
    # class labels.
    if not continous_bool:
        threshold_match_df = df_subset[df_subset[feature] == threshold]
        # if the threshold matches
        threshold_mismatch_df = df_subset[df_subset[feature] != threshold]
        # if the threshold does not match
    else:
        threshold_match_df = df_subset[df_subset[feature] <= threshold]
        # if the threshold matches
        threshold_mismatch_df = df_subset[df_subset[feature] > threshold]
        # if the threshold does not match
        
    threshold_match_entropy = impurity.entropy_calc(threshold_match_df)
    # entropy calculation for threshold match
    threshold_mismatch_entropy = impurity.entropy_calc(threshold_mismatch_df)
    # entropy calculation for threshold mismatch
    weighted_entropy = (sum(threshold_match_df['weights'])*threshold_match_entropy.get(feature, 0)/len(df_subset)
                        + sum(threshold_mismatch_df['weights'])*threshold_mismatch_entropy.get(feature, 0)/len(df_subset))
    # calculate the weighted entropy of the split
    feature_entropy_dict[threshold] = weighted_entropy
    # update the dictionary for the given threshold