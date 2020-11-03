#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:13:43 2019

@author: em812
"""
import warnings
import numpy as np
import pandas as pd

def drop_ventrally_signed(feat):

    abs_feat = [ft for ft in feat.columns if 'abs' in ft]

    ventr_feat = [ft.replace('_abs','') for ft in abs_feat]

    feat = feat[feat.columns.difference(ventr_feat, sort=False)]

    return feat

def select_feat_set(features, set_name, align_bluelight=False):
    """
    Keep only features in a predefined tierpsy feature set
    feat_set options:
        - tierpsy_8
        - tierpsy_16
        - tierpsy_256
        - tierpsy_2k
    """
    from os.path import join
    from tierpsytools import AUX_FILES_DIR

    filenames = {
        'tierpsy_8': 'tierpsy_8.csv',
        'tierpsy_16': 'tierpsy_16.csv',
        'tierpsy_256': 'tierpsy_256.csv',
        'tierpsy_2k': 'tierpsy_2k.csv'
        }

    set_file = join(AUX_FILES_DIR,'feat_sets',filenames[set_name])

    ft_set = pd.read_csv(set_file,header=None).loc[:,0].to_list()

    if align_bluelight:
        bluelight_conditions = ['prestim', 'bluelight', 'poststim']
        ft_set = ['_'.join([ft, blue]) for ft in ft_set for blue in bluelight_conditions]

    check = [ft in features.columns for ft in ft_set]
    if not np.all(check):
        warnings.warn('The features dataframe does not contain all the features in the selected features set. \n'
              'Only {} of the {} features exist in the dataframe.'.format(np.sum(check), len(ft_set)))
        ft_set = [ft for ft in ft_set if ft in features.columns]

    return features[ft_set]

def select_feat(feat, files=[], featList=[]):
    """
    SELECT_FEAT: function to choose features based on i) a list of files with feature names
    and/or ii) a list of feature names.
    param:
        feat = a data frame containing all the features
        files = list with the full paths of the files containing feature names (default = [])
        featList = list of feature names (default = [])
    return:
        feat = data frame with the chosen summary features
    """
    if len(files)==0 and len(featList)==0:
        warnings.warn('No features to select. All features are kept in the feature dataframe.')
    else:
        featFromFile = []
        if len(files)!=0:
            for i in range(len(files)):
                featFromFile += pd.read_csv(files[i],header=None).values.flatten().tolist()

        featAll = featFromFile+featList

        feat = feat[feat.columns[np.isin(feat.columns,featAll)]]

    return feat


def filter_nan_inf(feat,threshold,axis):
    """
    FILTER_NAN_INF: function to remove features or samples based on the
    ratio of NaN+Inf values.
    param:
        feat = feature dataframe or np array (rows = samples, columns = features)
        threshold = max allowed ratio of NaN values within a feature or a sample
        axis = axis of filtering (0 --> filter features, 1 --> filter samples)
    return:
        feat = filtered feature matrix
    """
    import numpy as np

    nanRatio = np.sum(np.logical_or(np.isnan(feat),np.isinf(feat)),axis=axis)/np.size(feat,axis=axis)
    if axis==0:
        feat = feat.loc[:,nanRatio<threshold]
    else:
        feat = feat.loc[nanRatio<threshold,:]

    return feat


def cap_feat_values(feat,cutoff=1e15,remove_all_nan=True):
    """
    CAP_FEAT_VALUES: function to replace features
    with too large values (>cutoff) with the max value of the given
    feature in the remaining data points.
    param:
        feat = feature dataframe or np array containing features to be capped
               (rows = samples, columns = features)
        remove_all_nan = boolean (default=True). When remove_all_nan=True, a feature column
                that has only large values (>cutoff) will be removed from the feat matrix.
    return:
        feat = filtered feature matrix
    """
    isarray = False

    if isinstance(feat,np.ndarray):
        isarray = True
        feat = pd.DataFrame(feat)

    drop_cols = []
    for col in feat.columns:
        if np.all(feat[col].values>cutoff):
            drop_cols.append(col)
        else:
            maxvalid = feat.loc[feat[col]<1e15,col].max()
            feat.loc[feat[col]>1e15,col] = maxvalid

    if remove_all_nan:
        feat = feat[feat.columns.difference([drop_cols])]

    if isarray:
        feat = feat.values

    return feat

def impute_nan_inf(feat, groupby=None):
    """
    IMPUTE_NAN_INF: replace NaN and inf values with feature average
    param:
        feat : dataframe or np array
            Features matrix (rows = samples, columns = features)
        groupby : array or list of arrays
            Ids based on which the feat dataframe will be grouped. The nans
            will be imputed with the mean values of each group independently.
    return:
        feat = feature matrix without nan/inf
    """

    isarray = False
    if isinstance(feat, np.ndarray):
        isarray = True
        feat = pd.DataFrame(feat)

    feat = feat.replace([np.inf, -np.inf], np.nan)

    # fill in nans with mean values of cv features for each strain separately
    if groupby is None:
        feat = feat.fillna(feat.mean())
    else:
        feat = [x for _,x in feat.groupby(by=groupby, sort=True)]
        for i in range(len(feat)):
            feat[i] = feat[i].fillna(feat[i].mean())
        feat = pd.concat(feat, axis=0).sort_index()

    # Covert back to array
    if isarray:
        feat = feat.values

    return feat


def feat_filter_std(feat):
    """
    FEAT_FILTER_STD: remove features with zero std
    param:
        feat = data frame or np array with features (rows = samples, columns = features)
    return:
        feat = filtered feature matrix
    """

    isarray = False
    if isinstance(feat,np.ndarray):
        isarray = True
        feat = pd.DataFrame(feat)

    feat = feat.loc[:,feat.std()!=0]

    if isarray:
        feat = feat.values

    return feat

def feat_remove_byKeyword(feat,keywords):
    """
    FEAT_REMOVE_BYKEYWORD: remove features that contain a keyword
    param:
        feat = dataframe features (rows = samples, columns = features)
    return:
        feat = filtered feature matrix
    """
    import numpy as np

    if isinstance(keywords,(list,np.ndarray)):
        for key in keywords:
            feat = feat[feat.columns.drop(feat.filter(like=key))]
    elif isinstance(keywords,str):
        feat = feat[feat.columns.drop(feat.filter(like=keywords))]

    return feat
