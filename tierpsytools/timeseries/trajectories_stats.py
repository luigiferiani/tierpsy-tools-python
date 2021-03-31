#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:55:43 2021

@author: lferiani
"""
# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tierpsy.summary.filtering import filter_trajectories
from tierpsytools.read_data.get_timeseries import read_timeseries


def get_n_worms_estimate(frame_numbers, percentile=99):
    """
    get_n_worms_estimate Get an estimate of the number of worms using the
    frame_numbers array.
    If a frame number appears twice, then there are two worms, etc.
    Does not use the max value of the number of times a frame_number appears,
    but the 99th percentile, for noise reasons.

    This should be used on a per-well basis unless you want to count
    worms across all frames.

    Parameters
    ----------
    frame_numbers : Pandas Series
        array of frame numbers
    percentile : int, optional
        percentile to use in calculating n_worms, by default 99

    Returns
    -------
    n_worms_estimate : int
        estimate of how many worms in the timeseries the frame_numbers is
        a column of
    """
    n_per_frame = frame_numbers.value_counts()
    n_per_frame = n_per_frame.values
    if len(n_per_frame) > 0:
        n_worms_estimate = int(np.percentile(n_per_frame, percentile))
    else:
        n_worms_estimate = 0
    return n_worms_estimate


def count_skeletons(lengths):
    """
    count_skeletons really just an alias for sum of non nans.

    Parameters
    ----------
    lengths : arrays or pandas series
        an array that is NaN when the worm is not skeletonised.
        Length is a good example of one

    Returns
    -------
    n_skeletons : int
        sum of non nans in the input array...
    """
    n_skeletons = (~lengths.isna()).sum()
    return n_skeletons


def get_worm_level_stats(oneworm_timeseries):
    """
    get_worm_level_stats operate on the timeseries of a single worm,
    calculate:
    - trajectory length: how many frames exist for this worm
    - fraction skeletonised: fraction of frames in which this worms was
        skeletonised
    Can (and should) be used as input for groupby('worm_index').apply()


    Parameters
    ----------
    oneworm_timeseries : DataFrame
        timeseries_data that pertains to a single worm.
        Can be obtained by groupby on the full timeseries_data

    Returns
    -------
    pandas Series
        Containing the calculated stats. If function is used in groupby.apply,
        this will create a nice dataframe.
    """

    # get measurements
    stats_out = {
        'trajectory_length': oneworm_timeseries.shape[0],
        'fraction_skeletonised': 1-oneworm_timeseries['length'].isna().mean(),
    }

    # returning a pandas series yields a multicolumn dataframe when
    # using this function within a groupby.apply
    # messes up the dtype though

    return pd.Series(stats_out)


def get_well_level_stats(onewell_timeseries):
    """
    get_well_level_stats operate on the timeseries of a single well,
    calculate:
    - estimated number of worms (looking at repeated timestamp values)
    - number of skeletons
    Can (and should) be used as input for groupby('well_name').apply()

    Parameters
    ----------
    onewell_timeseries : DataFrame
        timeseries_data that pertains to a single well.
        Can be obtained by groupby on the full timeseries_data

    Returns
    -------
    pandas Series
        Containing the calculated stats. If function is used in groupby.apply,
        this will create a nice dataframe.
    """

    stats_out = {
        'number_of_worms': get_n_worms_estimate(
            onewell_timeseries['timestamp'], percentile=99),
        'number_of_skeletons': count_skeletons(
            onewell_timeseries['length'])
        }
    return pd.Series(stats_out)


def calculate_timeseries_stats(timeseries_data, filter_params=None):
    """
    calculate_timeseries_stats
    measure basic statistics of the timeseries data, such as
    trajectory length, fraction of skeletonised frames for each worm,
    estimated number of worms in a well, number of skeletons in a well.
    Allows filtering consistent with the filtering done in Tierpsy summaries.

    Example usage
    -------------

    filter_parameters = {
        'min_traj_length': None, 'time_units': None,
        'min_distance_traveled': None, 'distance_units': None,
        'timeseries_names': ['length', 'width_midbody'],
        'min_thresholds': [200, 20],
        'max_thresholds': [2000, 500],
        'units': ['microns', 'microns'],
    }
    ts = read_timeseries(fname)
    worm_stats, well_stats = calculate_timeseries_stats(
        ts, filter_params=filter_parameters)

    Parameters
    ----------
    timeseries_data : pandas DataFrame
        [description]
    filter_params : dict, optional
        parameters for filtering, same as the ones used by Tierpsy's summaries.
        It is passed as kwargs to Tierpsy's filter_trajectories.
        NO CHECKS are done on the units!

    Returns
    -------
    worm_stats : pandas DataFrame
        [description]
    well_stats : pandas DataFrame
    """

    # First let's do worm-level statistics:
    worm_stats = timeseries_data.groupby(['well_name', 'worm_index']).apply(
        get_worm_level_stats).reset_index()

    # now well level stats:
    well_stats = timeseries_data.groupby('well_name').apply(
        get_well_level_stats).reset_index()

    # and filter if any filtering
    if filter_params is not None:

        # find good trajectories
        filtered_worm_indices = filter_trajectories(
            timeseries_data, None, **filter_params)[0]['worm_index'].unique()

        # and add a filtering column to worms stats
        worm_stats['is_good_trajectory'] = (
            worm_stats['worm_index'].isin(filtered_worm_indices))

        # and do well stats on filtered timeseries
        idx = timeseries_data['worm_index'].isin(filtered_worm_indices)
        well_stats_filtered = timeseries_data[idx].groupby('well_name').apply(
            get_well_level_stats).reset_index()

        # and merge with unfiltered
        well_stats = pd.merge(
            well_stats, well_stats_filtered,
            on='well_name',
            how='outer',
            suffixes=('', '_filtered'),
            validate='1:1',
            )

    return worm_stats, well_stats


# %% Example usage

if __name__ == '__main__':

    # example file
    fname = (
        '/Volumes/Ashur Pro2/SyngentaScreen/Results/20191205/'
        + 'syngenta_screen_run2_prestim_20191205_160426.22956805/'
        + 'metadata_featuresN.hdf5'
    )

    filter_parameters = {
        'min_traj_length': None, 'time_units': None,
        'min_distance_traveled': None, 'distance_units': None,
        'timeseries_names': ['length', 'width_midbody'],
        'min_thresholds': [200, 20],
        'max_thresholds': [2000, 500],
        'units': ['microns', 'microns'],
    }

    ts = read_timeseries(fname)

    worm_stats, well_stats = calculate_timeseries_stats(
        ts, filter_params=filter_parameters)
