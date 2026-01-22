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


"""
Blink detection and pupil signal cleaning functions.
"""
def detect_and_remove_blinks(
    raw_pupil,
    timestamp,
    return_mask=False,
    max_gap_duration=2.0,
    min_gap_duration=0.003,
    sampling_freq=1000,
    padding_before=0.001,
    padding_after=0.01,
    cluster_tolerance=0.15,
    plot_steps=False,
    mad_threshold=3.5,
    min_allowed_pupil=2000
):
    """
    Detect blinks and clean pupil size data through interpolation.
    
    Blink detection uses velocity-based outlier detection (MAD threshold on 
    smoothed dilation speed) combined with minimum pupil size filtering.
    Detected blinks are padded, clustered, and interpolated using cubic splines.
    
    Parameters
    ----------
    raw_pupil : array-like
        Raw pupil size measurements
    timestamp : array-like
        Timestamps corresponding to each pupil measurement
    return_mask : bool, default=False
        If True, return only the boolean blink mask (no cleaning/interpolation)
    max_gap_duration : float, default=2.0
        Maximum blink duration (seconds) to attempt interpolation.
        Longer blinks remain as NaN.
    min_gap_duration : float, default=0.003
        Minimum gap duration (seconds) for rejection padding
    sampling_freq : float, default=1000
        Sampling frequency in Hz
    padding_before : float, default=0.001
        Time padding (seconds) to add before each detected blink
    padding_after : float, default=0.01
        Time padding (seconds) to add after each detected blink
    cluster_tolerance : float, default=0.15
        Time tolerance (seconds) for clustering nearby blinks.
        Blinks closer than this are treated as one continuous blink.
    plot_steps : bool, default=False
        If True, generate diagnostic plots showing cleaning steps
    mad_threshold : float, default=3.5
        MAD threshold for velocity-based outlier detection
    min_allowed_pupil : float, default=2000
        Minimum valid pupil size (arbitrary units)
    
    Returns
    -------
    np.ndarray
        If return_mask=False: cleaned pupil data with blinks interpolated
        If return_mask=True: boolean array (1=blink, 0=valid data)
    
    Notes
    -----
    Interpolation strategy:
    1. Try cubic spline using 4 reference points around blink
    2. If cubic fails, fall back to linear interpolation
    3. If linear fails, leave as NaN
    
    ------
    Requires: calculate_speed_smoothed_pupil, detect_outliers_mad, pad_rejection_regions, 
              find_consecutive_groups
    """
    # Detect initial blinks using velocity threshold and minimum pupil size
    velocity_outliers = detect_outliers_mad(calculate_speed_smoothed_pupil(raw_pupil), 
                                  mad_threshold)[0]
    size_outliers = raw_pupil < min_allowed_pupil
    rejection_array = velocity_outliers | size_outliers
    
    # Initialize mask arrays
    padded_mask = np.zeros(len(rejection_array))
    final_blink_mask = np.zeros(len(rejection_array))
    
    # Pad detected blinks with temporal margins
    try:
        padded_indices = np.concatenate(
            pad_rejection_regions(
                timestamp, 
                rejection_array,
                min_gap_duration=min_gap_duration,
                padding_before=padding_before,
                padding_after=padding_after
            )
        )
        padded_mask[padded_indices.astype(int)] = 1
    except ValueError:
        padded_mask = rejection_array
    
    # Cluster nearby blinks (merge if closer than cluster_tolerance)
    try:
        blink_indices = np.where(padded_mask)[0]
        blink_clusters = find_consecutive_groups(
            blink_indices,
            stepsize=cluster_tolerance * sampling_freq
        )
        
        # Expand each cluster to fill gaps between nearby blinks
        merged_clusters = np.array([
            np.arange(cluster[0], cluster[-1] + 1) 
            for cluster in blink_clusters
        ])
        
        np.put(final_blink_mask, np.concatenate(merged_clusters), 1, mode='clip')
        
    except ValueError:
        final_blink_mask = padded_mask
    except IndexError:
        # No outliers detected
        if return_mask:
            return np.zeros(len(rejection_array))
        return raw_pupil
    
    blink_mask = final_blink_mask.copy()
    
    # Return just the mask if requested
    if return_mask:
        return blink_mask
    
    # Mask pupil data at blink locations
    masked_pupil_for_plot = np.where(final_blink_mask, np.nan, raw_pupil)
    masked_pupil = masked_pupil_for_plot.copy()
    
    # Prepare data for interpolation
    blink_clusters = find_consecutive_groups(np.where(final_blink_mask)[0], 
                                              stepsize=1)
    
    # Only interpolate blinks shorter than max_gap_duration
    interpolable_clusters = np.array([
        cluster for cluster in blink_clusters 
        if len(cluster) < max_gap_duration * sampling_freq
    ], dtype=object)
    
    # Get reference indices for interpolation (2 points before, 2 after each blink)
    interp_indices = np.asarray([
        np.array([
            max(0, cluster[0] - len(cluster)),           # far before
            max(0, cluster[0] - 1),                      # just before
            min(cluster[-1] + 1, len(timestamp) - 1),    # just after
            min(len(timestamp) - 1, cluster[-1] + len(cluster))  # far after
        ]) 
        for cluster in interpolable_clusters
    ], dtype=object).astype(int)
    
    # Convert to numpy arrays for indexing
    try:
        timestamp = timestamp.values
        raw_pupil = raw_pupil.values
    except AttributeError:
        timestamp = np.array(timestamp)
        raw_pupil = np.array(raw_pupil)
    
    # Interpolate each blink cluster
    try:
        ts_reference = timestamp[interp_indices]
        pupil_reference = masked_pupil[interp_indices]
        
        # Timestamps where we need interpolated values
        ts_targets = [timestamp[cluster.astype(int)] 
                     for cluster in interpolable_clusters]
        
        for i in range(len(interpolable_clusters)):
            try:
                # Try cubic spline (smooth, continuous derivative)
                interpolated = CubicSpline(
                    ts_reference[i], 
                    pupil_reference[i]
                )(ts_targets[i].astype(float))
                
            except ValueError:
                try:
                    # Fall back to linear interpolation
                    interpolated = interp1d(
                        ts_reference[i][1:3],  # just use adjacent points
                        pupil_reference[i][1:3]
                    )(ts_targets[i].astype(float))
                    
                except ValueError:
                    # Can't interpolate (e.g., at trial boundaries)
                    interpolated = np.nan
            
            # Fill blink with interpolated values
            masked_pupil[interpolable_clusters[i].astype(int)] = interpolated
            
    except IndexError:
        pass
    
    # Diagnostic plots
    if plot_steps:
        fig, axes = plt.subplots(nrows=5, figsize=(12, 15))
        
        # 1. Velocity and rejection threshold
        velocity = calculate_speed_smoothed_pupil(raw_pupil)
        threshold = detect_outliers_mad(velocity, mad_threshold)[1]
        sns.scatterplot(x=timestamp, y=velocity, ax=axes[0])
        axes[0].axhline(threshold, color='r', linestyle='--', 
                       label=f'MAD threshold={mad_threshold}')
        axes[0].set_ylabel('Dilation Speed')
        axes[0].set_title('1. Velocity-based Outlier Detection')
        axes[0].legend()
        
        # 2. Initial detection (unfiltered mask)
        sns.scatterplot(x=timestamp, y=raw_pupil, hue=rejection_array, 
                       ax=axes[1], palette=['blue', 'red'])
        axes[1].set_ylabel('Pupil Size')
        axes[1].set_title('2. Initial Blink Detection (Before Padding)')
        
        # 3. After padding and clustering
        sns.scatterplot(x=timestamp, y=raw_pupil, hue=final_blink_mask,
                       ax=axes[2], palette=['blue', 'red'])
        axes[2].set_ylabel('Pupil Size')
        axes[2].set_title('3. After Padding & Clustering')
        
        # 4. Masked data (NaN during blinks)
        sns.scatterplot(x=timestamp, y=masked_pupil_for_plot, ax=axes[3])
        axes[3].set_ylabel('Pupil Size')
        axes[3].set_title('4. Masked Data (Blinks → NaN)')
        
        # 5. Final interpolated data
        sns.scatterplot(x=timestamp, y=masked_pupil, ax=axes[4])
        axes[4].set_ylabel('Pupil Size')
        axes[4].set_title('5. Final Cleaned Data (Interpolated)')
        axes[4].set_xlabel('Time (s)')
        
        # Link x-axes
        for ax in axes[1:]:
            ax.sharex(axes[0])
        
        plt.tight_layout()
    
    return masked_pupil