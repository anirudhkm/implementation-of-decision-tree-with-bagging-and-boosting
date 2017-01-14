# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import impurity

def get_class_count(df):
    """
    This function helps to get the 
    class count of each label
    in the class column.
    Arguments:
    1. df: The input data as a pandas dataframe.
    Returns
    1. The class label count of the data.
    """
    
    return df["class"].value_counts()

def display_tree(tree, inp_file = '', tab_space = 0, header = False, level = 0):

    if header:
        # print the header
        print(("{}".format(inp_file)))
    for i in iter(tree):
        # iterate through the tree
        print("level:" + str(level), tab_space*'\t', (i[0], i[1][0]))
        # print out nodes
        level += 1
        # increase in the level
        if isinstance(tree[i]["equal"], (int, np.int64)):
            print((tab_space+1)*'\t', (tree[i]["equal"]))
        else:
            display_tree(tree[i]["equal"], tab_space = tab_space + 1, level=level)

        if isinstance(tree[i]["not_equal"], (int, np.int64)):
            print((tab_space+1)*'\t', tree[i]["not_equal"])
        else:
            display_tree(tree[i]["not_equal"], tab_space = tab_space + 1,level=level)

def check_leaf_node(df):
    """
    This function checks if we have
    the leaf node.
    
    Arguments:
    1. df: Data as pandas dataframe.
    Returns:
    1. True or False boolean.
    """
    
    return len(set(df['class'])) == 1
    
def data_split(df, best_feature, info_gain_dict, dt_dict,
               curr_node, depth, continous = False):
    """
    This function helps to split the data after the 
    best feature has been found. Then finally calls
    the decision_tree function to develop the tree.
    
    Arguments:
    1. df: The input data as a pandas dataframe.
    2. best_feature: The best feature on which we will split.
    3. info_gain_dict: The information gain dictionary.
    4. dt_dict: The dictionary on which the decision tree is built.
    5. curr_node: The current node pointer.
    
    Returns:
    No returns. Calls the decision_tree function 
    and we have the stopping function there.
    """
    
    depth -= 1
    # decrease the depth count
    no_data = False
    # default flag for data check
    match_threshold_df = df[df[best_feature] == info_gain_dict[best_feature][0]]
    # subset the data if threshold is matched
    if not len(match_threshold_df):
        # no more data points
        no_data = True
        match_threshold_df = df
        # go back to prev dataframe
    else:
        pass
    
    mismatch_threshold_df = df[df[best_feature] != info_gain_dict[best_feature][0]]
    # subset the data if there is a mismatch
    if not len(mismatch_threshold_df):
        # if no more data points
        no_data = True
        mismatch_threshold_df = df
        # go back to prev dataframe
    else:
        pass
    decision_tree(match_threshold_df, dt_dict, curr_node, best_feature,
                  align_dir = "equal", depth=depth, no_data = no_data)
    # function call to grow tree on the left side
    decision_tree(mismatch_threshold_df, dt_dict, curr_node, best_feature,
                  align_dir = "not_equal", depth=depth, no_data = no_data)
    # function call to grow the tree on the right side

def decision_tree(df, dt_dict, curr_node,
                  prev_attr = None, align_dir = None,
                  depth = -1, no_data = False,
                  ensemble = None):
    """
    This function builds the decision tree
    using recursion.
    Arguments:
    1. df: The data as a pandas dataframe.
    2. dt_dict: The dictionary on which the decision
    tree will be built.
    3. curr_node: The current node of the decision 
    tree stored in a dictionary as well.
    
    Default Arguments:
    1. prev_attr: The previous attribute, defaulted to None.
    2. align_dir: The alignment of tree, left or right side.
    3. depth: The depth of the tree, which is defaulted to 12.
    
    Return:
    1. df_dict: The constructed decision tree as a dicitonary.
    """
    
    class_count = get_class_count(df)
    # get the class label counts for the given dataframe
    leaf_node_bool = check_leaf_node(df)
    # this function helps to check if we have a leaf node
    if leaf_node_bool:
        # if its leaf node
        curr_node[align_dir] = df['class'].values[0]
        # assign the leaf node value
    elif no_data:
        # if we are out of data points
        class_counts = df['class'].value_counts()
        # get the class counts
        curr_node[align_dir] = np.argmax(class_counts)
        # assign the majority class of prev node
    else:
        entropy_values_series = impurity.entropy_calc(df, ensemble = ensemble)
        # calculate the entropy values for each feature
        info_gain_dict = {}
        # empty dict for information gain
        for feature in entropy_values_series.index:
            # iterate over each features
            impurity.information_gain_calc(df, feature, info_gain_dict)
            # function call for information gain calculation
        for f in entropy_values_series.index:
            # iterate over each feature
            information_gain = entropy_values_series[f] - info_gain_dict[f][1]
            # calculation of information gain
            info_gain_dict[f] = (info_gain_dict[f][0], information_gain)
            # update the information gain dict
        best_feature = sorted(info_gain_dict, key = lambda x: info_gain_dict[x][1])[-1]
        # get the best feature on which to be splitted.
        #print(best_feature)
        node_value = (best_feature, info_gain_dict[best_feature], class_count[0],
                      class_count[1])
        # get the node value
        
        if not leaf_node_bool and align_dir:
            # growing the tree
            if depth == 0:
                if node_value[2] > node_value[3]:
                    node_value = 0
                else:
                    node_value = 1
                curr_node[align_dir] = node_value
                return 0
            else:
                curr_node[align_dir] = {node_value:{}}
                curr_node = curr_node[align_dir][node_value]
        else:
            dt_dict[node_value] = {}
            curr_node = dt_dict[node_value]
        
        data_split(df, best_feature, info_gain_dict, 
                                dt_dict, curr_node, depth)
        # function call for data split
    
def read_input_file(inp_file, sep = ","):
    """
    This function helps to read the input
    file as a pandas dataframe and returns them.
    Arguments:
    1. inp_file: The input file as a string.
    Return:
    1. df: returns the pandas dataframe.
    """
    
    df = pd.read_csv(inp_file, sep = sep)
    # read the input file as a pandas dataframe
    return df
    

def main(df, filename, depth = -1, ensemble = None):
    """
    This function serves as the heart of the 
    DT program and develops the tree from here.
    Arguments:
    1. df: The training data as a pandas dataframe.
    2. filename: The input filename as a string.
    3. weights: The weights with which entropy is calculated, 
    defaulted to None.
    """
    
    decision_tree_dict = {}
    # empty dictionary on which the decision tree will be built
    decision_tree(df, decision_tree_dict, {}, depth = depth, ensemble = ensemble)
    # function call to build the decision tree
    #display_tree(decision_tree_dict, filename, header=True)
    #print(decision_tree_dict)
    return decision_tree_dict


    