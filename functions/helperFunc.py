import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.interpolate import CubicSpline, interp1d

import os
import glob
from pathlib import Path

import ast


"""
Pupil dilation and eye movement speed calculation functions.
"""

def _ensure_numeric_array(data):
    """
    Convert input data to numeric numpy array, handling NaN values.
    
    Parameters
    ----------
    data : array-like
        Input data (pandas Series, numpy array, or list)
    
    Returns
    -------
    np.ndarray
        Numeric array with NaN replaced by 0.0
    """
    try:
        # Handle pandas Series
        data = pd.to_numeric(data, errors='coerce').fillna(0.0)
    except AttributeError:
        # Handle numpy array
        data = np.where(np.isnan(data), 0, data)
    
    # Convert to numpy array if pandas Series
    try:
        return data.values
    except AttributeError:
        return data


def calculate_speed_unsmoothed(timepoint, pupil_size):
    """
    Calculate absolute dilation speed without smoothing.
    
    Uses two-point finite difference method with forward/backward differences
    at boundaries and maximum of forward/backward at interior points.
    
    Parameters
    ----------
    timepoint : array-like
        Time values corresponding to each measurement
    pupil_size : array-like
        Pupil size measurements
    
    Returns
    -------
    np.ndarray
        Absolute dilation speed at each timepoint
    """
    dilation_speed = np.zeros(len(pupil_size))
    
    ps = _ensure_numeric_array(pupil_size)
    tp = _ensure_numeric_array(timepoint)
    
    # Interior points: max of forward and backward differences
    prev = np.abs((ps[1:-1] - ps[0:-2]) / (tp[1:-1] - tp[0:-2]))
    post = np.abs((ps[1:-1] - ps[2:]) / (tp[1:-1] - tp[2:]))
    dilation_speed[1:-1] = np.maximum(prev, post)
    
    # Boundary points
    dilation_speed[0] = np.abs((ps[1] - ps[0]) / (tp[1] - tp[0]))
    dilation_speed[-1] = np.abs((ps[-1] - ps[-2]) / (tp[-1] - tp[-2]))
    
    return dilation_speed


def calculate_speed_smoothed_pupil(pupil_size):
    """
    Calculate absolute dilation speed with 5-point smoothing for pupil data.
    
    Uses centered finite difference with 2-point window on each side.
    Assumes uniform time sampling.
    
    Parameters
    ----------
    pupil_size : array-like
        Pupil size measurements
    
    Returns
    -------
    np.ndarray
        Smoothed absolute dilation speed at each timepoint
    """
    dilation_speed = np.zeros(len(pupil_size))
    
    ps = _ensure_numeric_array(pupil_size)
    
    # Extract values, handling both pandas and numpy
    try:
        nminus1 = ps.iloc[1:-3].values
        nplus1 = ps.iloc[3:-1].values
        nminus2 = ps.iloc[0:-4].values
        nplus2 = ps.iloc[4:].values
        
        # Boundary points
        dilation_speed[0] = np.abs(ps.iloc[1] - ps.iloc[0])
        dilation_speed[1] = np.abs(ps.iloc[2] - ps.iloc[1])
        dilation_speed[-1] = np.abs(ps.iloc[-1] - ps.iloc[-2])
        dilation_speed[-2] = np.abs(ps.iloc[-2] - ps.iloc[-3])
    except (AttributeError, TypeError):
        nminus1 = ps[1:-3]
        nplus1 = ps[3:-1]
        nminus2 = ps[0:-4]
        nplus2 = ps[4:]
        
        # Boundary points
        dilation_speed[0] = np.abs(ps[1] - ps[0])
        dilation_speed[1] = np.abs(ps[2] - ps[1])
        dilation_speed[-1] = np.abs(ps[-1] - ps[-2])
        dilation_speed[-2] = np.abs(ps[-2] - ps[-3])
    
    # Interior points: 5-point smoothed derivative
    dilation_speed[2:-2] = np.abs((nplus1 + nplus2 - nminus1 - nminus2) / 6)
    
    return dilation_speed


def calculate_velocity_saccade(position):
    """
    Calculate velocity for saccade data with 5-point smoothing.
    
    Uses centered finite difference with 2-point window on each side.
    Note: Returns signed velocity (not absolute value).
    
    Parameters
    ----------
    position : array-like
        Position measurements (e.g., eye position during saccade)
    
    Returns
    -------
    np.ndarray
        Smoothed velocity values (signed, not absolute)
    """
    vec = _ensure_numeric_array(position)
    
    # Extract values, handling both pandas and numpy
    try:
        nminus1 = vec.iloc[1:-3].values
        nplus1 = vec.iloc[3:-1].values
        nminus2 = vec.iloc[0:-4].values
        nplus2 = vec.iloc[4:].values
    except (AttributeError, TypeError):
        nminus1 = vec[1:-3]
        nplus1 = vec[3:-1]
        nminus2 = vec[0:-4]
        nplus2 = vec[4:]
    
    # Calculate derivative (5-point stencil)
    velocity = (nplus1 + nplus2 - nminus1 - nminus2) / 6
    
    return velocity



"""
General utility functions for eye tracking data processing.
"""

def find_consecutive_groups(data, stepsize=10, find_same=False):
    """
    Split array into groups of consecutive or nearby values.
    
    Useful for clustering timepoints (e.g., saccade detection, blink detection).
    
    Parameters
    ----------
    data : array-like
        1D array of values to group (typically timepoints or indices)
    stepsize : int, default=10
        Maximum difference between consecutive values to be considered 
        in the same group. Larger values allow more tolerance.
    find_same : bool, default=False
        If True, only group identical consecutive values (stepsize ignored).
        Useful for finding runs of 1s or 0s in binary masks.
    
    Returns
    -------
    list of np.ndarray
        List of arrays, each containing indices of one consecutive group
    
    Examples
    --------
    >>> find_consecutive_groups([1, 2, 3, 10, 11, 12], stepsize=5)
    [array([1, 2, 3]), array([10, 11, 12])]
    
    >>> find_consecutive_groups([1, 1, 1, 0, 0, 1], find_same=True)
    [array([1, 1, 1]), array([0, 0]), array([1])]
    """
    data = np.asarray(data)
    
    if find_same:
        # Find where values change
        split_indices = np.where(np.diff(data) != 0)[0] + 1
    else:
        # Find where difference exceeds stepsize
        split_indices = np.where(np.diff(data) > stepsize)[0] + 1
    
    return np.split(data, split_indices)


def smooth_signal(signal, window_len=11, window='hanning'):
    """
    Smooth a 1D signal using a window function.
    
    Parameters
    ----------
    signal : array-like
        Input signal to smooth (pandas Series or numpy array)
    window_len : int, default=11
        Length of smoothing window (must be odd number >= 3)
    window : str, default='hanning'
        Type of window function to use. Options:
        - 'flat': moving average (uniform weights)
        - 'hanning': Hann window (smooth, general purpose)
        - 'hamming': Hamming window
        - 'bartlett': Bartlett (triangular) window
        - 'blackman': Blackman window
    
    Returns
    -------
    np.ndarray
        Smoothed signal with same length as input
    
    Raises
    ------
    ValueError
        If signal is not 1D, signal is shorter than window, or 
        invalid window type specified
    
    Notes
    -----
    Uses reflection padding at boundaries to avoid edge effects.
    See https://numpy.org/doc/stable/reference/routines.window.html
    """
    # Convert pandas Series to numpy array
    try:
        signal = signal.values
    except AttributeError:
        signal = np.asarray(signal)
    
    # Validation
    if signal.ndim != 1:
        raise ValueError("smooth_signal only accepts 1D arrays.")
    
    if signal.size < window_len:
        raise ValueError("Input signal must be longer than window size.")
    
    if window_len < 3:
        return signal
    
    valid_windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if window not in valid_windows:
        raise ValueError(f"Window must be one of {valid_windows}")
    
    # Reflect signal at boundaries for padding
    padded = np.r_[signal[window_len-1:0:-1], 
                   signal, 
                   signal[-2:-window_len-1:-1]]
    
    # Create window weights
    if window == 'flat':
        weights = np.ones(window_len, dtype='d')
    else:
        weights = getattr(np, window)(window_len)
    
    # Apply convolution with normalized weights
    smoothed = np.convolve(weights / weights.sum(), padded, mode='valid')
    
    # Remove padding to match original signal length
    half_window = int(window_len / 2)
    return smoothed[half_window:-half_window]
def is_strictly_increasing(array):
    """
    Check if array values are strictly increasing.
    
    Parameters
    ----------
    array : array-like
        Input array to check
    
    Returns
    -------
    bool
        True if each element is strictly less than the next element
    
    Examples
    --------
    >>> is_strictly_increasing([1, 2, 3, 4])
    True
    >>> is_strictly_increasing([1, 2, 2, 3])
    False
    """
    return all(x < y for x, y in zip(array, array[1:]))


def fs(width, height):
    """
    Set the default figure size for matplotlib plots.
    
    Parameters
    ----------
    width : float
        Figure width in inches
    height : float
        Figure height in inches
    
    Examples
    --------
    >>> set_figure_size(12, 6)  # Wide figure for time series
    >>> set_figure_size(8, 8)   # Square figure
    """
    plt.rcParams['figure.figsize'] = (width, height)


"""
Outlier detection and rejection utilities for pupil data.
"""


def detect_outliers_mad(data, mad_threshold):
    """
    Detect outliers using Median Absolute Deviation (MAD) method.
    
    MAD is a robust measure of variability that is less sensitive to 
    extreme values than standard deviation.
    
    Parameters
    ----------
    data : array-like
        Data array to analyze (e.g., dilation speed values)
    mad_threshold : float
        Number of MAD units above the median to use as rejection threshold.
        Typical values: 2.5-3.5 (analogous to 2-3 standard deviations)
    
    Returns
    -------
    outlier_mask : np.ndarray
        Boolean array where True indicates outlier (reject)
    threshold_value : float
        The computed rejection threshold value
    
    Examples
    --------
    >>> data = np.array([1, 2, 2.5, 2.3, 100])  # 100 is outlier
    >>> mask, threshold = detect_outliers_mad(data, mad_threshold=3)
    >>> mask
    array([False, False, False, False, True])
    
    Notes
    -----
    Formula: threshold = median + (mad_threshold × MAD)
    where MAD = median(|data - median(data)|)
    """
    # Handle NaN and null values
    try:
        data = np.where(np.isnan(data), 0, data)
    except TypeError:
        # Handle pandas null values
        data = np.where(pd.isnull(data), 0, data)
    
    # Calculate Median Absolute Deviation
    median_val = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median_val))
    
    # Compute rejection threshold
    rejection_threshold = median_val + mad_threshold * mad
    
    # Create boolean mask (True = outlier/reject)
    outlier_mask = np.abs(data) >= rejection_threshold
    
    return outlier_mask, rejection_threshold


def pad_rejection_regions(
    timestamp,
    rejection_mask,
    min_gap_duration=0.03,
    padding_after=0.05,
    padding_before=0.0,
    sampling_freq=1000
):
    """
    Pad rejection regions with temporal margins and filter by minimum duration.
    
    Identifies continuous rejection periods, filters out brief gaps, and adds
    temporal padding before/after each rejection region to ensure clean data.
    
    Parameters
    ----------
    timestamp : array-like
        Timestamps for each data point
    rejection_mask : array-like
        Boolean array where True indicates rejected/invalid data
    min_gap_duration : float, default=0.03
        Minimum duration (seconds) for a gap to be considered valid rejection.
        Shorter gaps are ignored.
    padding_after : float, default=0.05
        Duration (seconds) to pad after each rejection region
    padding_before : float, default=0.0
        Duration (seconds) to pad before each rejection region
    sampling_freq : float, default=1000
        Sampling frequency in Hz
    
    Returns
    -------
    np.ndarray of arrays
        Array of index arrays, each containing indices for one padded rejection region.
        Empty array if no valid rejections found.
    
    Examples
    --------
    >>> timestamp = np.linspace(0, 1, 1000)
    >>> rejection_mask = np.zeros(1000, dtype=bool)
    >>> rejection_mask[400:450] = True  # 50ms rejection
    >>> padded = pad_rejection_regions(timestamp, rejection_mask, 
    ...                                min_gap_duration=0.03,
    ...                                padding_after=0.01)
    >>> # Returns indices 400-459 (original 400-449 + 10 samples padding)
    
    Notes
    -----
    This function is typically used to:
    1. Remove brief noise artifacts (via min_gap_duration)
    2. Add safety margins around blinks/artifacts (via padding)
    """
    # Convert durations to sample counts
    padding_samples_after = int(padding_after * sampling_freq)
    padding_samples_before = int(padding_before * sampling_freq)
    min_gap_samples = int(min_gap_duration * sampling_freq)
    
    # Find continuous runs of rejected data
    # Create a NaN-based representation for finding runs
    marked_array = np.where(rejection_mask, np.nan, rejection_mask)
    
    # Find start and end indices of NaN runs
    # Add True at boundaries to catch runs at start/end
    is_nan = np.isnan(marked_array)
    boundaries = np.flatnonzero(np.r_[True, np.diff(is_nan) != 0, True])
    run_lengths = np.diff(boundaries)
    run_starts = boundaries[:-1]
    
    # Filter for NaN runs only
    is_nan_run = is_nan[run_starts]
    rejection_starts = run_starts[is_nan_run]
    rejection_lengths = run_lengths[is_nan_run]
    
    # Get end indices
    rejection_ends = rejection_starts + rejection_lengths - 1
    rejection_regions = list(zip(rejection_starts, rejection_ends))
    
    # Return empty if no rejections found
    if len(rejection_regions) == 0:
        return np.array([], dtype=object)
    
    # Filter by minimum gap duration
    meets_duration = np.array([
        (end - start) > min_gap_samples 
        for start, end in rejection_regions
    ])
    valid_regions = np.array(rejection_regions)[meets_duration]
    
    # Add padding to each valid rejection region
    padded_regions = []
    n_samples = len(timestamp)
    
    for start, end in valid_regions:
        # Apply padding with boundary checks
        padded_start = max(0, start - padding_samples_before)
        padded_end = min(n_samples - 1, end + padding_samples_after)
        
        # Create array of indices for this padded region
        padded_indices = np.arange(padded_start, padded_end + 1)
        padded_regions.append(padded_indices)
    
    return np.array(padded_regions, dtype=object)