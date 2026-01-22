
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.interpolate import CubicSpline, interp1d

import os
import glob
from pathlib import Path

import ast

from .helperFunc import *
from .deBlink import *
from .readRaw import *


# Configuration
LOCAL_USERNAME = 'yud070'
PATH_EYETRACKER = Path("Z:/yueying/psychedeLights_YueyingDong/data/pdRaw_tracker/")
PATH_PSYCHOPY = Path("Z:/yueying/psychedeLights_YueyingDong/data/pdRaw_psychopy/")

"""
Pupil data cleaning pipeline with two-stage outlier removal and participant-level filtering.
"""

def apply_two_stage_cleaning(
    pupil_raw_df,
    first_pass_params=None,
    second_pass_params=None,
    smoothing_window=21
):
    """
    Apply two-stage blink detection and cleaning with intermediate smoothing.
    
    Stage 1: Cleaning to remove major artifacts
    Stage 2: Secnd cleaning on smoothed data to catch residual outliers
    
    Parameters
    ----------
    pupil_raw_df : pd.DataFrame
        Raw pupil data with TRIALID in first column, samples in remaining columns
    first_pass_params : dict, optional
        Parameters for first cleaning pass. Defaults to aggressive settings.
    second_pass_params : dict, optional
        Parameters for second cleaning pass. Defaults to conservative settings.
    smoothing_window : int, default=21
        Window size for rolling mean smoothing between cleaning stages
    
    Returns
    -------
    pd.DataFrame
        Cleaned pupil data with TRIALID column
    
    Notes
    -----
    Default parameters are optimized for blink removal in cognitive tasks.
    First pass uses aggressive MAD threshold (10) to catch major artifacts.
    Second pass uses conservative threshold (12) on smoothed data.
    """
    # Set default parameters
    if first_pass_params is None:
        first_pass_params = {
            'padding_before': 0.05,
            'padding_after': 0.2,
            'min_allowed_pupil': 2000,
            'cluster_tolerance': 0.05,
            'mad_threshold': 10
        }
    
    if second_pass_params is None:
        second_pass_params = {
            'min_allowed_pupil': 2000,
            'padding_before': 0.05,
            'padding_after': 0.05,
            'min_gap_duration': 0.002,
            'cluster_tolerance': 0.05,
            'mad_threshold': 12
        }
    
    print("Stage 1: Initial cleaning...")
    
    # First cleaning pass
    cleaned_once = pupil_raw_df.iloc[:, 1:].apply(
        lambda trial: detect_and_remove_blinks(
            raw_pupil=trial.astype(float),
            timestamp=np.linspace(0, int(len(trial)/1000), len(trial)),
            padding_before=first_pass_params['padding_before'],
            padding_after=first_pass_params['padding_after'],
            min_allowed_pupil=first_pass_params['min_allowed_pupil'],
            cluster_tolerance=first_pass_params['cluster_tolerance'],
            mad_threshold=first_pass_params['mad_threshold']
        ),
        axis=1,
        raw=True
    )
    
    print("Stage 2: Smoothing...")
    
    # Smooth with rolling window
    smoothed = cleaned_once.rolling(
        window=smoothing_window,
        min_periods=1,
        center=True,
        axis=1
    ).mean()
    
    print("Stage 3: Second cleaning...")
    
    # Second cleaning pass (on smoothed data)
    cleaned_twice = smoothed.apply(
        lambda trial: detect_and_remove_blinks(
            raw_pupil=trial,
            timestamp=np.linspace(0, int(len(trial)/1000), len(trial)),
            min_allowed_pupil=second_pass_params['min_allowed_pupil'],
            padding_before=second_pass_params['padding_before'],
            padding_after=second_pass_params['padding_after'],
            min_gap_duration=second_pass_params['min_gap_duration'],
            cluster_tolerance=second_pass_params['cluster_tolerance'],
            mad_threshold=second_pass_params['mad_threshold']
        ),
        axis=1,
        raw=True
    )
    
    # Add trial IDs back
    result = cleaned_twice.reset_index(drop=True)
    result.insert(0, 'TRIALID', pupil_raw_df['TRIALID'].values)
    
    return result


def calculate_median_variance(data):
    """
    Calculate robust variance estimate using median absolute deviation.
    
    Parameters
    ----------
    data : array-like
        Data to calculate variance for (can contain NaN values)
    
    Returns
    -------
    float
        Median-based variance estimate
    
    Notes
    -----
    More robust to outliers than standard deviation.
    """
    # Flatten if DataFrame
    if isinstance(data, pd.DataFrame):
        values = data.values.flatten()
    else:
        values = np.asarray(data).flatten()
    
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    
    return mad


def apply_participant_level_filtering(
    pupil_df,
    participant_info_df,
    sd_threshold=6,
    max_samples=8000
):
    """
    Apply participant-level outlier filtering based on individual distributions.
    
    For each participant, reject samples that exceed median ± (threshold × MAD).
    This accounts for individual differences in baseline pupil size.
    
    Parameters
    ----------
    pupil_df : pd.DataFrame
        Cleaned pupil data with TRIALID in first column
    participant_info_df : pd.DataFrame
        DataFrame containing 'participant' and 'TRIALID' columns
    sd_threshold : float, default=6
        Number of median-based SD units for rejection threshold
    max_samples : int, default=8000
        Maximum number of samples per trial to consider
    
    Returns
    -------
    pd.DataFrame
        Pupil data with participant-level outliers masked as NaN
    """
    print(f"Applying participant-level filtering (threshold={sd_threshold} MAD)...")
    
    # Merge to get participant labels
    pupil_with_id = pupil_df.merge(
        participant_info_df[['participant', 'TRIALID']],
        on='TRIALID',
        how='left'
    )
    
    # Store masks for each participant
    participant_masks = []
    
    for participant in pupil_with_id['participant'].unique():
        # Get this participant's data
        participant_data = pupil_with_id[pupil_with_id['participant'] == participant]
        pupil_values = participant_data.iloc[:, 1:max_samples + 1]
        
        # Calculate participant-specific bounds
        median = np.nanmedian(pupil_values)
        variance = calculate_median_variance(pupil_values)
        
        upper_bound = median + sd_threshold * variance
        lower_bound = median - sd_threshold * variance
        
        print(f"  Participant {participant}: "
              f"bounds = [{lower_bound:.0f}, {upper_bound:.0f}]")
        
        # Create outlier mask for this participant
        outlier_mask = (pupil_values > upper_bound) | (pupil_values < lower_bound)
        outlier_mask.insert(0, 'TRIALID', participant_data['TRIALID'].values)
        
        participant_masks.append(outlier_mask)
    
    # Combine all masks
    combined_mask = pd.concat(participant_masks, ignore_index=True)
    
    # Apply mask to original data
    result = pupil_df.copy()
    result.iloc[:, 1:] = result.iloc[:, 1:].mask(combined_mask.iloc[:, 1:], np.nan)
    
    return result


def filter_trials_by_data_quality(
    pupil_df,
    behavioral_df,
    time_window=(4001, 7501),
    max_missing_proportion=0.2,
    exclude_missing_responses=True
):
    """
    Filter out trials with insufficient valid data or missing responses.
    
    Parameters
    ----------
    pupil_df : pd.DataFrame
        Pupil data with TRIALID in first column
    behavioral_df : pd.DataFrame
        Behavioral data with TRIALID and missingRsp columns
    time_window : tuple of int, default=(4001, 7501)
        Column indices defining time window of interest for quality check
    max_missing_proportion : float, default=0.2
        Maximum proportion of missing data allowed in time window (0-1)
    exclude_missing_responses : bool, default=True
        If True, exclude trials where participant didn't respond
    
    Returns
    -------
    pd.DataFrame
        Filtered pupil data containing only high-quality trials
    
    Notes
    -----
    Default time_window (4001:7501) corresponds to 3.5 seconds of data,
    likely the critical task period for analysis.
    """
    print("\nFiltering trials by data quality...")
    
    start_idx, end_idx = time_window
    window_size = end_idx - start_idx
    max_missing_samples = int(max_missing_proportion * window_size)
    
    # Count missing data in time window of interest
    missing_counts = pupil_df.iloc[:, start_idx:end_idx].isnull().sum(axis=1)
    quality_mask = missing_counts < max_missing_samples
    
    print(f"  Trials with <{max_missing_proportion*100:.0f}% missing data: "
          f"{quality_mask.sum()}/{len(pupil_df)}")
    
    # Apply quality filter
    filtered = pupil_df[quality_mask].reset_index(drop=True)
    
    # Exclude trials with missing behavioral responses
    if exclude_missing_responses:
        missing_response_trials = behavioral_df[
            behavioral_df['missingRsp'] == True
        ]['TRIALID'].values
        
        response_mask = ~filtered['TRIALID'].isin(missing_response_trials)
        
        print(f"  Trials with valid responses: "
              f"{response_mask.sum()}/{len(filtered)}")
        
        filtered = filtered[response_mask].reset_index(drop=True)
    
    return filtered


def get_trial_counts_by_participant(pupil_df):
    """
    Count remaining trials per participant after filtering.
    
    Parameters
    ----------
    pupil_df : pd.DataFrame
        Pupil data with TRIALID column containing participant info
    
    Returns
    -------
    pd.Series
        Trial counts per participant, sorted by frequency
    """
    # Extract participant from TRIALID (assumes format like "['participant_id', ...]")
    participants = pd.Series([
        ast.literal_eval(trial_id)[0] 
        for trial_id in pupil_df['TRIALID']
    ])
    
    return participants.value_counts()

def process_pupil_data_pipeline(
    pupil_raw_df,
    behavioral_df,
    run_processing=True,

    # Cleaning parameters
    first_pass_params=None,
    second_pass_params=None,
    smoothing_window=21,

    # Participant filtering parameters
    sd_threshold=6,
    max_samples=8000,
    
    # Trial quality parameters
    time_window=(4001, 7501),
    max_missing_proportion=0.2,
    exclude_missing_responses=True
):
    """
    Complete pupil data processing pipeline.
    
    Pipeline stages:
    1. Two-stage blink detection and removal
    2. Participant-level outlier filtering
    3. Trial quality filtering
    
    Parameters
    ----------
    pupil_raw_df : pd.DataFrame
        Raw pupil data
    behavioral_df : pd.DataFrame
        Behavioral data with participant info and response flags
    run_processing : bool, default=True
        If False, skip processing and return empty DataFrame
    
    Cleaning Parameters
    -------------------
    first_pass_params : dict, optional
        Parameters for first cleaning pass. If None, uses defaults:
        {'padding_before': 0.05, 'padding_after': 0.2, 
         'min_allowed_pupil': 2000, 'cluster_tolerance': 0.05, 
         'mad_threshold': 10}
    second_pass_params : dict, optional
        Parameters for second cleaning pass. If None, uses defaults:
        {'min_allowed_pupil': 2000, 'padding_before': 0.05,
         'padding_after': 0.05, 'min_gap_duration': 0.002,
         'cluster_tolerance': 0.05, 'mad_threshold': 12}
    smoothing_window : int, default=21
        Rolling window size for smoothing between cleaning stages
    
    Participant Filtering Parameters
    ---------------------------------
    sd_threshold : float, default=6
        Number of MAD units for participant-level outlier detection
    max_samples : int, default=8000
        Maximum samples per trial to consider
    
    Trial Quality Parameters
    -------------------------
    time_window : tuple of int, default=(4001, 7501)
        (start_idx, end_idx) defining critical time window for quality check
    max_missing_proportion : float, default=0.2
        Maximum proportion of missing data allowed (0-1)
    exclude_missing_responses : bool, default=True
        Whether to exclude trials with missing behavioral responses
    
    Returns
    -------
    pd.DataFrame
        Fully processed and filtered pupil data
    
    Examples
    --------
    # Use default parameters
    >>> clean_data = process_pupil_data_pipeline(raw_pupil, behavior)
    
    # Adjust for more conservative cleaning
    >>> clean_data = process_pupil_data_pipeline(
    ...     raw_pupil, behavior,
    ...     first_pass_params={'mad_threshold': 8},  # Less aggressive
    ...     sd_threshold=8  # Wider participant bounds
    ... )
    
    # Adjust for different experiment timing
    >>> clean_data = process_pupil_data_pipeline(
    ...     raw_pupil, behavior,
    ...     time_window=(2000, 5000),  # Different critical period
    ...     max_missing_proportion=0.15  # Stricter quality requirement
    ... )
    """
    if not run_processing or pupil_raw_df.shape[0] == 0:
        print("Skipping processing (no data or processing disabled)")
        return pd.DataFrame()
    
    print("=" * 60)
    print("PUPIL DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Stage 1: Two-stage cleaning
    cleaned_pupil = apply_two_stage_cleaning(
        pupil_raw_df,
        first_pass_params=first_pass_params,
        second_pass_params=second_pass_params,
        smoothing_window=smoothing_window
    )
    
    # Stage 2: Participant-level filtering
    participant_filtered = apply_participant_level_filtering(
        cleaned_pupil,
        behavioral_df,
        sd_threshold=sd_threshold,
        max_samples=max_samples
    )
    
    # Stage 3: Trial quality filtering
    final_clean = filter_trials_by_data_quality(
        participant_filtered,
        behavioral_df,
        time_window=time_window,
        max_missing_proportion=max_missing_proportion,
        exclude_missing_responses=exclude_missing_responses
    )
    
    # Report final statistics
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total trials retained: {len(final_clean)}")
    print("\nTrials per participant:")
    print(get_trial_counts_by_participant(final_clean))
    
    return final_clean