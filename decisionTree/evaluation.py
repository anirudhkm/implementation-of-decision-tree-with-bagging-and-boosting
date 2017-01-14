# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

def print_confusion_matrix(tp, fp, fn, tn):
    """
    This function helps to print out the
    confusion matrix by taking in the
    required parameters.
    Arguments:
    1. tp: The true positive value.
    2. fp: The false positive value.
    3. fn: The false negative value.
    4. tn: The true negative value.
    Returns:
    None, just prints out the confusion matrix.
    """
    
    print("{0:^36}".format("CONFUSION MATRIX"))
    print("{0:^36}".format("================"))
    print("{0:^12}{1:^12}{2:^12}".format('', 'Predict:-', 'Predict:+'))
    print("{0:^12}{1:^12}{2:^12}".format('Actual:-', tn, fp))
    print("{0:^12}{1:^12}{2:^12}".format('Actual:+', fn, tp))

def calc_params_for_conf_matrix(df):
    """
    This function helps to calculate the
    following parameters.
    1. True Positive.
    2. True Negative.
    3. False Negative.
    4. False Positive.
    Arguments:
    1. df: The input data as a pandas dataframe.
    Returns:
    1. tp, tn, fn, fp: The parameters needed for
    confusion matrix calculation.
    """
    
    tp, tn, fn, fp = 0, 0, 0, 0
    # initial value for the parameters
    actual = df["class"].values
    # the class labels
    predict = df["predict"].values
    # the predicted class labels
    for i in range(len(actual)):
        # iterate over the class labels
        if actual[i] == 1 and predict[i] == 1:
            # if true positive
            tp += 1
        elif actual[i] == 1 and predict[i] != 1:
            # checking for false negative
            fn += 1
        elif actual[i] != 1 and predict[i] == 1:
            # check for false positive
            fp += 1
        elif actual[i] != 1 and predict[i] != 1:
            # check for true negative
            tn += 1
    return tp, fp, fn, tn
    
        
def model_evaluation(df, depth):
    """
    This function helps to calculate
    the accuracy of the decision tree 
    and prints them.
    Arguments:
    df: The data as a pandas dataframe.
    depth: The depth of the tree as an integer.
    Returns:
    1.This function does not return anything.
    """
    
    print("{0:<100}".format("DEPTH: " + str(depth)))
    # print kind of header
    accuracy = np.mean(df['class'] == df['predict'])*100
    # calculate the accuracy of the classifier
    tp, fp, fn, tn = calc_params_for_conf_matrix(df)
    # function call to calculate parameters
    misclassification_count = fp + fn
    # get the misclassification count
    print("Accuracy: {0}%".format(round(accuracy,4 )))
    # print out the accuracy of the classifier
    print("Misclassification count: " + str(misclassification_count))
    # print the misclassification count
    print_confusion_matrix(tp, fp, fn, tn)
    # function call to print out the confusion matrix    
    
    
    
    
    