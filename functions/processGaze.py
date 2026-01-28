import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.interpolate import CubicSpline, interp1d

import os
import glob
from pathlib import Path


from .helperFunc import *
from .deBlink import *
from .readRaw import *
from .processPup import *

def process_gaze_data_pipeline(
    pupil_raw_df,
    gaze_x_raw_df,
    gaze_y_raw_df,
    pupil_clean_df,
    pupil_std_mask_df,
    max_samples=8000,
    blink_params=None,
    smoothing_window=11,
    interpolation_method='nearest'
):
    """
    Process gaze data with blink masking, artifact removal, and interpolation.
    
    Generates a gaze mask from blink detection, combines it with pupil size
    artifact mask, applies to X/Y gaze coordinates, then interpolates.
    
    Parameters
    ----------
    pupil_raw_df : pd.DataFrame
        Raw pupil data with TRIALID in first column
    gaze_x_raw_df : pd.DataFrame
        Raw X-coordinate gaze data with TRIALID
    gaze_y_raw_df : pd.DataFrame
        Raw Y-coordinate gaze data with TRIALID
    pupil_clean_df : pd.DataFrame
        Cleaned pupil data (used to filter which trials to keep)
    pupil_std_mask_df : pd.DataFrame
        Pupil size outlier mask from participant-level filtering
    max_samples : int, default=8000
        Maximum samples per trial
    blink_params : dict, optional
        Parameters for blink detection. Defaults to conservative settings.
    smoothing_window : int, default=11
        Window size for anti-aliasing before interpolation
    interpolation_method : str, default='nearest'
        Interpolation method ('nearest', 'linear', 'cubic')
    
    Returns
    -------
    pd.DataFrame
        Cleaned and interpolated gaze data with MultiIndex (axis, trial)
        where axis is 'x' or 'y'
    
    Notes
    -----
    Pipeline:
    1. Generate blink mask from pupil data (conservative MAD=12)
    2. Combine with pupil size artifact mask
    3. Apply mask to gaze X/Y coordinates
    4. Filter to only trials that passed pupil quality checks
    5. Smooth with rolling window (anti-aliasing)
    6. Interpolate missing segments
    """
    print("=" * 60)
    print("GAZE DATA PROCESSING")
    print("=" * 60)
    
    # Set default blink detection parameters (conservative to reduce false positives)
    if blink_params is None:
        blink_params = {
            'padding_before': 0.05,
            'padding_after': 0.05,
            'min_allowed_pupil': 2000,
            'cluster_tolerance': 0.03,
            'mad_threshold': 12  # Higher threshold = fewer false positives
        }
    
    # Step 1: Generate gaze mask from blink detection
    print("\nStep 1: Generating blink mask for gaze filtering...")
    gaze_mask = pupil_raw_df.iloc[:, 1:].apply(
        lambda trial: detect_and_remove_blinks(
            raw_pupil=trial.astype(float),
            timestamp=np.linspace(0, max_samples / 1000, len(trial)),
            padding_before=blink_params['padding_before'],
            padding_after=blink_params['padding_after'],
            min_allowed_pupil=blink_params['min_allowed_pupil'],
            cluster_tolerance=blink_params['cluster_tolerance'],
            mad_threshold=blink_params['mad_threshold'],
            return_mask=True
        ),
        axis=1,
        raw=True
    )
    gaze_mask.insert(0, 'TRIALID', pupil_raw_df['TRIALID'])
    
    # Step 2: Find trials present in all datasets
    print("Step 2: Aligning trials across datasets...")
    shared_trials = list(
        set(gaze_x_raw_df['TRIALID']) & 
        set(gaze_mask['TRIALID'])
    )
    print(f"  Found {len(shared_trials)} shared trials")
    
    # Filter and sort all datasets by shared trials
    gaze_mask_aligned = gaze_mask[
        gaze_mask['TRIALID'].isin(shared_trials)
    ].sort_values('TRIALID').reset_index(drop=True)
    
    pupil_std_mask_aligned = pupil_std_mask_df[
        pupil_std_mask_df['TRIALID'].isin(shared_trials)
    ].sort_values('TRIALID').reset_index(drop=True)
    
    gaze_x_aligned = gaze_x_raw_df[
        gaze_x_raw_df['TRIALID'].isin(shared_trials)
    ].sort_values('TRIALID').reset_index(drop=True)
    
    gaze_y_aligned = gaze_y_raw_df[
        gaze_y_raw_df['TRIALID'].isin(shared_trials)
    ].sort_values('TRIALID').reset_index(drop=True)
    
    # Step 3: Combine blink mask with pupil artifact mask
    print("Step 3: Combining blink mask with pupil artifact mask...")
    combined_mask = gaze_mask_aligned.copy()
    combined_mask.iloc[:, 1:] = (
        gaze_mask_aligned.iloc[:, 1:].values.astype(bool) |
        pupil_std_mask_aligned.iloc[:, 1:].values.astype(bool)
    ).astype(float)
    
    # Step 4: Verify trial alignment
    print("Step 4: Verifying trial alignment...")
    trial_mismatch_x = (combined_mask['TRIALID'] != gaze_x_aligned['TRIALID']).sum()
    trial_mismatch_y = (combined_mask['TRIALID'] != gaze_y_aligned['TRIALID']).sum()
    
    if trial_mismatch_x > 0 or trial_mismatch_y > 0:
        raise ValueError(
            f"TRIALID mismatch: X has {trial_mismatch_x} mismatches, "
            f"Y has {trial_mismatch_y} mismatches"
        )
    print("  ✓ All trials aligned correctly")
    
    # Step 5: Apply mask to gaze data
    print("Step 5: Applying mask to gaze coordinates...")
    gaze_x_masked = gaze_x_aligned.copy()
    gaze_y_masked = gaze_y_aligned.copy()
    
    gaze_x_masked.iloc[:, 1:] = gaze_x_aligned.iloc[:, 1:].mask(
        combined_mask.iloc[:, 1:].astype(bool),
        np.nan
    )
    gaze_y_masked.iloc[:, 1:] = gaze_y_aligned.iloc[:, 1:].mask(
        combined_mask.iloc[:, 1:].astype(bool),
        np.nan
    )
    
    # Add axis labels and combine
    gaze_x_masked['axis'] = 'x'
    gaze_y_masked['axis'] = 'y'
    gaze_combined = pd.concat([gaze_x_masked, gaze_y_masked], ignore_index=True)
    
    # Step 6: Filter to trials that passed pupil quality checks
    print("Step 6: Filtering to high-quality trials...")
    trials_to_keep = pupil_clean_df['TRIALID'].values
    gaze_filtered = gaze_combined[
        gaze_combined['TRIALID'].isin(trials_to_keep)
    ].reset_index(drop=True)
    
    n_removed = len(gaze_combined) - len(gaze_filtered)
    print(f"  Removed {n_removed // 2} trials (kept {len(gaze_filtered) // 2} trials)")
    
    # Step 7: Anti-aliasing smoothing
    print(f"Step 7: Smoothing with window size {smoothing_window}...")
    gaze_smoothed = gaze_filtered.iloc[:, 1:].rolling(
        window=smoothing_window,
        min_periods=1,
        center=True,
        axis=1
    ).mean()
    
    # Step 8: Interpolation
    print(f"Step 8: Interpolating missing segments (method={interpolation_method})...")
    timepoints = np.linspace(0, 5, max_samples)
    
    gaze_interpolated = gaze_smoothed.iloc[:, :max_samples].apply(
        lambda trial: linear_interpolate(timepoints, trial, method=interpolation_method),
        axis=1,
        raw=True
    )
    
    # Add back metadata
    gaze_interpolated.insert(0, 'TRIALID', gaze_filtered['TRIALID'])
    gaze_interpolated.insert(0, 'axis', gaze_filtered['axis'])
    
    # Set axis as index for easy access
    gaze_clean = gaze_interpolated.set_index('axis')
    
    print("\n" + "=" * 60)
    print("GAZE PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final dataset: {len(gaze_clean) // 2} trials × {max_samples} timepoints")
  
    return gaze_clean


def detect_gaze_shifts_2d(
    gaze_x,
    gaze_y,
    threshold=6,
    cluster_tolerance=50,
    baseline=None,
    window_length=50,
    min_shift_amplitude=0.2,
    baseline_padding=0,
    return_mask_only=False
):
    """
    Detect gaze shifts using 2D Euclidean velocity thresholding.
    
    Computes 2D gaze velocity, identifies threshold crossings, clusters nearby
    shifts, and calculates shift amplitude by comparing pre/post positions.
    
    Parameters
    ----------
    gaze_x : array-like
        Horizontal gaze position (degrees or pixels)
    gaze_y : array-like
        Vertical gaze position (degrees or pixels)
    threshold : float, default=6
        Velocity threshold in MAD units (median + threshold × MAD)
    cluster_tolerance : int, default=50
        Maximum temporal gap (samples) to merge nearby shifts into one cluster
    baseline : None, list, or float, optional
        Baseline period for shift calculation:
        - None: Use relative baseline (position before shift)
        - [start, end]: Use specific time window as baseline
        - float: Use fixed baseline value
    window_length : int, default=50
        Window size (samples) for calculating mean position before/after shift
    min_shift_amplitude : float, default=0.2
        Minimum shift amplitude (degrees/pixels) to count as real shift
    baseline_padding : int, default=0
        Additional padding (samples) before baseline window.
        Only used when baseline=None.
    return_mask_only : bool, default=False
        If True, return binary mask (1=shift onset, 0=no shift).
        If False, return shift amplitudes at onset times.
    
    Returns
    -------
    np.ndarray
        If return_mask_only=True: Boolean array (1 at shift onsets, 0 elsewhere)
        If return_mask_only=False: Array with shift amplitudes at onset times,
                                   0 elsewhere
    
    Notes
    -----
    - Uses 5-point smoothed velocity calculation (see calculate_speed_smoothed_pupil)
    - Velocity array is 4 samples shorter than input due to smoothing
    - First and last (3×window_length) samples are excluded from detection
    - Shift amplitude = mean_position_after - mean_position_before
    
    Examples
    --------
    >>> # Detect shifts with relative baseline
    >>> shifts = detect_gaze_shifts_2d(gaze_x, gaze_y, threshold=6)
    
    >>> # Use fixed baseline period
    >>> shifts = detect_gaze_shifts_2d(gaze_x, gaze_y, baseline=[500, 1000])
    
    >>> # Get only shift locations (not amplitudes)
    >>> shift_mask = detect_gaze_shifts_2d(gaze_x, gaze_y, return_mask_only=True)
    """
    # Calculate 2D velocity (Euclidean distance in velocity space)
    velocity_x = calculate_speed_smoothed_pupil(gaze_x).astype(float)
    velocity_y = calculate_speed_smoothed_pupil(gaze_y).astype(float)
    velocity_2d = _euclidean_distance(velocity_x, velocity_y)
    
    # Calculate velocity threshold using MAD
    median_velocity = np.nanmedian(velocity_2d)
    mad_velocity = calculate_median_variance(velocity_2d)
    velocity_threshold = median_velocity + mad_velocity * threshold
    
    # Identify threshold crossings
    threshold_crossings = velocity_2d > velocity_threshold
    
    # Create mask to exclude edges (can't compute baseline/aftermath at edges)
    total_edge_padding = window_length + baseline_padding
    valid_region_mask = np.r_[
        np.zeros(total_edge_padding),
        np.ones(len(threshold_crossings) - 3 * window_length - 2 * baseline_padding),
        np.zeros(2 * window_length + baseline_padding)
    ].astype(bool)
    
    # Apply mask and find clusters
    valid_crossings = np.where(threshold_crossings & valid_region_mask)[0]
    shift_clusters = find_consecutive_groups(valid_crossings, stepsize=cluster_tolerance)
    
    # Initialize output array
    gaze_shifts = np.zeros(len(gaze_x))
    
    # If only need shift locations, return binary mask
    if return_mask_only:
        try:
            shift_onsets = np.array([cluster[0] for cluster in shift_clusters])
            gaze_shifts[shift_onsets] = 1
        except IndexError:
            pass  # No shifts detected
        return gaze_shifts
    
    # Calculate shift amplitudes
    try:
        for cluster in shift_clusters:
            shift_onset = cluster[0]
            shift_offset = cluster[-1]
            
            # Determine baseline window
            if baseline is None:
                # Use relative baseline (position before shift)
                baseline_start = shift_onset - window_length - baseline_padding
                baseline_end = shift_onset - baseline_padding
            else:
                baseline_start, baseline_end = baseline
            
            # Calculate pre-shift position
            if isinstance(baseline, (int, float)):
                # User provided fixed baseline value
                position_before = baseline
            else:
                # Calculate mean position in baseline window
                try:
                    position_before = np.nanmean(
                        gaze_x.iloc[baseline_start:baseline_end]
                    )
                except AttributeError:
                    # Handle numpy arrays (not pandas Series)
                    position_before = np.nanmean(
                        gaze_x[baseline_start:baseline_end]
                    )
            
            # Calculate post-shift position
            try:
                position_after = np.nanmean(
                    gaze_x.iloc[shift_offset:shift_offset + window_length]
                )
            except AttributeError:
                position_after = np.nanmean(
                    gaze_x[shift_offset:shift_offset + window_length]
                )
            
            # Calculate shift amplitude
            shift_amplitude = position_after - position_before
            
            # Only record shift if amplitude exceeds minimum
            if abs(shift_amplitude) > min_shift_amplitude:
                gaze_shifts[shift_onset] = shift_amplitude
                
    except IndexError:
        # No valid clusters found
        pass
    
    return gaze_shifts


def calculate_gaze_shifts(
    gaze_clean_df,
    x_shift_threshold=8,
    y_shift_threshold=8,
    smoothing_window=51,
    shift_detection_threshold=6
):
    """
    Calculate gaze shifts (saccades) from cleaned gaze data.
    
    Detects saccadic eye movements by smoothing gaze coordinates and
    identifying shifts that exceed minimum thresholds in either direction.
    
    Parameters
    ----------
    gaze_clean_df : pd.DataFrame
        Cleaned gaze data with MultiIndex (axis='x'/'y')
        from process_gaze_data()
    x_shift_threshold : float, default=8
        Minimum horizontal shift (pixels) to count as saccade
    y_shift_threshold : float, default=8
        Minimum vertical shift (pixels) to count as saccade
    smoothing_window : int, default=51
        Window size for smoothing before shift detection.
        Larger values reduce noise but may miss small saccades.
    shift_detection_threshold : float, default=6
        Threshold for shift detection algorithm (pixels)
    
    Returns
    -------
    pd.DataFrame
        Gaze shift data with columns:
        - 'axis': 'x' or 'y'
        - 'TRIALID': trial identifier
        - Timepoint columns: relative shift magnitude at each timepoint
    
    Notes
    -----
    - Y-axis typically more noisy, hence same smoothing window for both
    - Shifts filtered to require minimum magnitude in either X or Y direction
    - Positive/negative values indicate shift direction
    """
    print("=" * 60)
    print("GAZE SHIFT CALCULATION")
    print("=" * 60)
    
    # Step 1: Smooth gaze coordinates
    print(f"\nStep 1: Smoothing gaze data (window={smoothing_window})...")
    
    smoothed_x = gaze_clean_df.loc['x'].iloc[:, 1:].apply(
        lambda trial: smooth_signal(trial, window_len=smoothing_window),
        axis=1,
        raw=True
    )
    
    smoothed_y = gaze_clean_df.loc['y'].iloc[:, 1:].apply(
        lambda trial: smooth_signal(trial, window_len=smoothing_window),
        axis=1,
        raw=True
    )
    
    print(f"  Smoothed {len(smoothed_x)} trials")
    
    # Step 2: Calculate gaze shifts in both dimensions
    print(f"Step 2: Detecting gaze shifts (threshold={shift_detection_threshold}px)...")
    
    # Initialize shift arrays
    shift_x = np.zeros(shape=smoothed_x.shape)
    shift_y = np.zeros(shape=smoothed_y.shape)
    
    # Calculate shifts for each trial
    for i, (x_coords, y_coords) in enumerate(zip(smoothed_x.values, smoothed_y.values)):
        # X shifts (considering Y position for 2D detection)
        shift_x[i] = detect_gaze_shifts_2d(
            x_coords,
            y_coords,
            min_shift_amplitude=0,
            threshold=shift_detection_threshold
        )
        
        # Y shifts (considering X position for 2D detection)
        shift_y[i] = detect_gaze_shifts_2d(
            y_coords,
            x_coords,
            min_shift_amplitude=0,
            threshold=shift_detection_threshold
        )
    
    # Step 3: Filter out small drifts
    print(f"Step 3: Filtering shifts (X≥{x_shift_threshold}px or Y≥{y_shift_threshold}px)...")
    
    # Create combined magnitude mask
    magnitude_mask = (
        (np.abs(shift_x) > x_shift_threshold) |
        (np.abs(shift_y) > y_shift_threshold)
    )
    
    # Apply filter - set small shifts to zero
    shift_x_filtered = np.where(magnitude_mask, shift_x, 0)
    shift_y_filtered = np.where(magnitude_mask, shift_y, 0)
    
    # Count significant shifts
    n_shifts_x = (shift_x_filtered != 0).sum()
    n_shifts_y = (shift_y_filtered != 0).sum()
    print(f"  Detected {n_shifts_x:,} X-shifts and {n_shifts_y:,} Y-shifts")
    
    # Step 4: Create output DataFrames
    print("Step 4: Formatting output...")
    
    # X-axis shifts
    shift_df_x = pd.DataFrame(shift_x_filtered)
    shift_df_x.insert(0, 'TRIALID', gaze_clean_df.loc['x']['TRIALID'].values)
    shift_df_x.insert(0, 'axis', 'x')
    
    # Y-axis shifts
    shift_df_y = pd.DataFrame(shift_y_filtered)
    shift_df_y.insert(0, 'TRIALID', gaze_clean_df.loc['y']['TRIALID'].values)
    shift_df_y.insert(0, 'axis', 'y')
    
    # Combine
    shift_df_combined = pd.concat([shift_df_x, shift_df_y], axis=0, ignore_index=True)
    
    print("\n" + "=" * 60)
    print("GAZE SHIFT CALCULATION COMPLETE")
    print("=" * 60)
    print(f"Output shape: {shift_df_combined.shape}")
    print(f"Trials with shifts: {(shift_df_combined.iloc[:, 2:] != 0).any(axis=1).sum() // 2}")
    
    return shift_df_combined


def _euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two vectors.
    
    Parameters
    ----------
    a, b : array-like
        Input vectors
    
    Returns
    -------
    np.ndarray
        Euclidean distance sqrt(a² + b²)
    """
    return np.sqrt(a**2 + b**2)