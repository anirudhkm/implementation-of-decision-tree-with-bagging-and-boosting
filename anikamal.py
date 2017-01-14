#! /usr/bin/python

'''
This is a template outlining the functions we are expecting for us to be able to
interface with an call your code. This is not all of the functions you need. You
will also need to make sure to bring your decision tree learner in somehow
either by copying your code into a learn_decision_tree function or by importing
your decision tree code in from the file your wrote for PA#1. You will also need
some functions to handle creating your data bags, computing errors on your tree,
and handling the reweighting of data points.

For building your bags numpy's random module will be helpful.
'''

from __future__ import division
import numpy as np
import pandas as pd
import sys
sys.path.append("decisionTree")
import main
import evaluation

def get_weights(df, alpha_lst, initial = False):
    """
    This function helps to get the weights for the
    adaboost algorithm and returns them.
    Argument:
    1. df: The data as a pandas dataframe.
    2. initial: Flag check if its the initial weight calculation.
    """
    
    if initial:
        # check if initial weights
        weights = [1/len(df)]*len(df)
        # get the initial weights
        return weights
        # return weights
    else:
        df = df[['class', 'predict', 'weights']]
        # subset the data
        epsilon = df[df['class'] != df['predict']]['weights'].sum()
        # get the epsilon value
        if epsilon != 0:
            alpha = np.log((1 - epsilon)/epsilon)*0.5
            # calculate the alpha value
            alpha_lst.append(alpha)
            # append alpha values
            pos = np.e**(-alpha)
            neg = np.e**(alpha)
            # get weights for pos and neg classification
            norm = 2*((epsilon*(1 - epsilon))**(0.5))
            # find the normalization constant
            classified = df[df['class'] == df['predict']]
            misclassified = df[df['class'] != df['predict']]
            #df['weights'] = df['weights'][df['class'] == df['predict']]*pos/norm
            #df['weights'] = df['weights'][df['class'] != df['predict']]*neg/norm
            df.set_value(classified.index, "weights", df['weights']*pos/norm)
            df.set_value(misclassified.index, "weights", df['weights']*neg/norm)
            # perform normalization
            return df['weights']
        else:
            df['weights'] = 0
            return df['weights']
        
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


def learn_bagged(tdepth, numbags, datapath):
    '''
    Function: learn_bagged(tdepth, numbags, datapath)
    tdepth: (Integer) depths to which to grow the decision trees
    numbags: (Integer)the number of bags to use to learn the trees
    datapath: (String) the location in memory where the data set is stored

    This function will manage coordinating the learning of the bagged ensemble.

    Nothing is returned, but the accuracy of the learned ensemble model is printed
    to the screen.
    '''
    
    train_file = datapath + '//' + 'agaricuslepiotatrain1.csv'
    test_file = datapath + '//' + 'agaricuslepiotatest1.csv'
    # train and test file path
    main_df = preprocess(pd.read_csv(train_file))
    # read train file as a pandas dataframe
    main_df['weights'] = 1
    # add a column of weights
    bagging_df = pd.DataFrame()
    # empty dataframe
    for i in range(numbags):
        # iterate over each classifier
        np.random.seed(i+1)
        # set the seed value
        df = pd.DataFrame(main_df.values[np.random.randint(len(main_df), size = len(main_df))],
                          columns = main_df.columns)
        # generate random data with replacement
        df_train, df_test = main.mains(df, train_file, tdepth, test_file)
        # function call
        bagging_df["classifier_"+str(i+1)] = df_test['predict']
        # add predictor to the dataframe
    bagging_df['class'] = df_test['class']
    # add the actual class to the bagging dataframe
    bagging_predictor = []
    # empty list to store final prediction
    for index in bagging_df.index:
        # iterate over each index
        best_predictor = pd.value_counts(bagging_df.loc[index]).idxmax()
        # get the majority class
        bagging_predictor.append(best_predictor)
        # get the best predictor from all classifiers
    bagging_df['predict'] = bagging_predictor
    # add the final predictor to the dataframe
    print("BAGGING\n=======\nNUMBER OF TREES USED: " + str(numbags))
    # print the number of bags
    evaluation.model_evaluation(bagging_df[['class', 'predict']], tdepth)
    # evaluation the model

def learn_boosted(tdepth, numtrees, datapath):
    '''
    Function: learn_boosted(tdepth, numtrees, datapath)
    tdepth: (Integer) depths to which to grow the decision trees
    numtrees: (Integer) the number of boosted trees to learn
    datapath: (String) the location in memory where the data set is stored
    
    This function wil manage coordinating the learning of the boosted ensemble.
    
    Nothing is returned, but the accuracy of the learned ensemble model is printed
    to the screen.
    '''
    
    train_file = datapath + '//' + 'agaricuslepiotatrain1.csv'
    test_file = datapath + '//' + 'agaricuslepiotatest1.csv'
    # train and test file path
    main_df = preprocess(pd.read_csv(train_file))
    # read train file as a pandas dataframe
    adaboost_df = pd.DataFrame()
    # empty dataframe
    alpha_lst = []
    # empty list to store alpha
    weights = get_weights(main_df, alpha_lst, initial = True)
    # get the initial weights to work on
    for i in range(numtrees):
        # iterate over each classifier
        main_df['weights'] = weights
        # add weights to dataframe
        df_train, df_test = main.mains(main_df, train_file, tdepth, test_file)
        # function cald
        df_train['weights'] = weights
        # add weights column to the dataframe
        weights = get_weights(df_train, alpha_lst)
        # get the updated weights values
        if sum(weights) == 0:
            break
        else:
            pass
        df_test['predict'][df_test['predict'] == 0] = -1
        df_test['class'][df_test['class'] == 0] = -1
        adaboost_df['classifier_' + str(i+1)] = df_test['predict']*alpha_lst[i]
        # store the prediction with alpha multiplied
    adaboost_df['predict'] = adaboost_df.sum(axis = 1)
    adaboost_df['predict'][adaboost_df['predict'] >= 0] = 1
    adaboost_df['predict'][adaboost_df['predict'] < 0] = -1
    adaboost_df['class'] = df_test['class']
    print("ADABOOST\n========\nNUMBER OF TREES USED: " + str(numtrees))
    # print the number of bags
    evaluation.model_evaluation(adaboost_df, tdepth)


if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    # Get the depth of the trees
    tdepth = int(sys.argv[2]);
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    # Get the location of the data set
    datapath = sys.argv[4];

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);
