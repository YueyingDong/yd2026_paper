import math
import numpy as np
import pywt
import pandas as pd

def calculate_ipa(pupil_data, sampling_freq=1000):
    """
    Calculate Index of Pupillary Activity (IPA) using wavelet decomposition.
    
    IPA quantifies the rate of pupillary oscillations by detecting modulus maxima
    in the wavelet domain and counting events above a threshold.
    
    Parameters
    ----------
    pupil_data : array-like
        Pupil diameter time series. NaN values will be excluded.
        This is the data after rejecting blinks, but before interpolation

    sampling_freq : float, optional
        Sampling frequency in Hz (default: 1000)
    
    Returns
    -------
    float
        IPA value (events per second), or np.nan if decomposition fails
    
    References
    ----------
    Based on wavelet-based pupillary activity analysis methods.
    """
    # Remove NaN values
    pupil_clean = pupil_data[~np.isnan(pupil_data)]
    
    # Perform 2-level discrete wavelet transform
    try:
        cA2, cD2, cD1 = pywt.wavedec(pupil_clean, 'sym16', 'antireflect', level=2)
    except Exception:
        return np.nan
    
    # Get signal duration in seconds
    duration = len(pupil_clean) / sampling_freq
    
    # Normalize coefficients by 1/sqrt(2^j) where j is the decomposition level
    cA2 = cA2 / math.sqrt(4)  # j=2
    cD1 = cD1 / math.sqrt(2)  # j=1
    cD2 = cD2 / math.sqrt(4)  # j=2
    
    # Find modulus maxima in detail coefficients
    cD2_modmax = _compute_modulus_maxima(cD2)
    
    # Apply universal threshold
    threshold = np.std(cD2_modmax) * math.sqrt(2.0 * np.log2(len(cD2_modmax)))
    cD2_thresholded = pywt.threshold(cD2_modmax, threshold, mode='hard')
    
    # Calculate IPA: count of suprathreshold events per second
    ipa = np.sum(np.fabs(cD2_thresholded) > 0) / duration
    
    return ipa


def _compute_modulus_maxima(coefficients):
    """
    Identify local maxima in wavelet coefficient magnitudes.
    
    A point is a modulus maximum if its magnitude is >= both neighbors
    and strictly > at least one neighbor.
    
    Parameters
    ----------
    coefficients : array-like
        Wavelet detail coefficients
    
    Returns
    -------
    ndarray
        Array with magnitudes at local maxima, zeros elsewhere
    """
    magnitudes = np.fabs(coefficients)
    modmax = np.zeros(len(coefficients))
    
    for i in range(len(coefficients)):
        # Handle boundary conditions
        left = magnitudes[i-1] if i > 1 else magnitudes[i]
        center = magnitudes[i]
        right = magnitudes[i+1] if i < len(coefficients) - 2 else magnitudes[i]
        
        # Check if local maximum
        is_peak = (left <= center and center >= right) and (left < center or center > right)
        
        if is_peak:
            modmax[i] = math.sqrt(coefficients[i]**2)
    
    return modmax

def calculate_rolling_ipa(pupil_data, trial_index, window_size=10, 
                          start_sample=4000, end_sample=7500, 
                          downsample_factor=10, sampling_freq=100):
    """
    Calculate rolling IPA over a sliding window on downsampled pupil data.
    
    Parameters
    ----------
    pupil_data : ndarray
        Pupil data array (trials x samples)
    trial_index : array
        Trial IDs corresponding to pupil_data rows
    window_size : int, optional
        Size of sliding window in downsampled bins (default: 10)
    start_sample : int, optional
        Start sample index before downsampling (default: 4001)
    end_sample : int, optional
        End sample index before downsampling (default: 7501)
    downsample_factor : int, optional
        Factor to downsample by (default: 10)
    sampling_freq : float, optional
        Sampling frequency after downsampling in Hz (default: 100)
    
    Returns
    -------
    ndarray
        Rolling IPA values (trials x windows)
    
    Notes
    -----
    IMPORTANT: Make sure sampling_freq matches the downsampled rate.
    For original 1000 Hz downsampled by factor 10 → use sampling_freq=100
    """
    # Extract time window and downsample
    pupil_window = pupil_data[:, start_sample:end_sample]
    n_samples = end_sample - start_sample
    n_bins = np.arange(n_samples) // downsample_factor
    
    
    # Downsample by averaging bins
    pupil_downsampled = pd.DataFrame(pupil_window).groupby(
        n_bins, 
        axis=1
    ).mean(numeric_only=True)
    
    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(
        pupil_downsampled.values, 
        window_shape=(window_size,), 
        axis=1
    )
    
    # Calculate IPA for each window
    ipa_rolling = np.apply_along_axis(
        lambda x: calculate_ipa(x, sampling_freq=int(sampling_freq/downsample_factor)),
        axis=2, 
        arr=windows
    )
    
    ipa_rolling = pd.concat([pd.DataFrame(trial_index,columns=['TRIALID']),
                       pd.DataFrame(ipa_rolling)],axis = 1)

    
    return ipa_rolling