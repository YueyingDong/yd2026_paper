
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

# Configuration
LOCAL_USERNAME = 'yud070'
PATH_EYETRACKER = Path("Z:/yueying/psychedeLights_YueyingDong/data/pdRaw_tracker/")
PATH_PSYCHOPY = Path("Z:/yueying/psychedeLights_YueyingDong/data/pdRaw_psychopy/")



# Column definitions
RELEVANT_COLUMNS = [
    'RECORDING_SESSION_LABEL', 'AVERAGE_GAZE_X', 'AVERAGE_GAZE_Y',
    'AVERAGE_PUPIL_SIZE', 'EYE_TRACKED', 'IP_LABEL', 'IP_START_TIME',
    'LEFT_PUPIL_SIZE', 'RIGHT_PUPIL_SIZE', 'SAMPLE_MESSAGE',
    'TIMESTAMP', 'TRIALID'
]

PUPIL_COLUMNS = [
    'EYE_TRACKED', 'LEFT_PUPIL_SIZE', 'RIGHT_PUPIL_SIZE',
    'AVERAGE_PUPIL_SIZE', 'IP_LABEL', 'TIMESTAMP', 'TRIALID'
]

# Analysis parameters
SELECTED_EYE = 'AVERAGE_PUPIL_SIZE'
MAX_SAMPLES_PER_TRIAL = 8000


"""
Script to process eye tracking data from psychedeLights experiment.
Reads raw tracker data and organizes into trial × timestamp structure.
"""

def load_eyetracker_file(filepath, relevant_cols):
    """
    Load and preprocess eye tracker data file.
    
    Parameters
    ----------
    filepath : Path or str
        Path to eye tracker .txt file
    relevant_cols : list
        Column names to load
    
    Returns
    -------
    pd.DataFrame
        Preprocessed eye tracking data
    """
    # Read eye tracker data
    df = pd.read_csv(
        filepath,
        delimiter="\t",
        usecols=relevant_cols
    )
    
    # Clean trial IDs
    df['TRIALID'] = df['TRIALID'].replace('UNDEFINED', np.nan)
    df = df[~df['TRIALID'].isnull()]
    
    # Forward fill sample messages
    df['SAMPLE_MESSAGE'] = df['SAMPLE_MESSAGE'].replace('.', np.nan).ffill()
    
    return df


def prepare_trial_data(df, eye_column, max_samples):
    """
    Transform long-format data to trial × timestamp structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw eye tracking data
    eye_column : str
        Column name for pupil size measurement to use
    max_samples : int
        Maximum number of samples to include per trial
    
    Returns
    -------
    dict
        Dictionary with keys: 'pupil', 'gaze_x', 'gaze_y', 'events'
        Each value is a DataFrame with trials as rows, timestamps as columns
    """
    # Rename and convert columns; I had this naming convention
    # because in the past I had been dealing with pupil lab data, which had
    # this convention
    df = df.rename(columns={
        eye_column: 'diameter_3d',
        'TIMESTAMP': 'pupil_timestamp'
    })
    
    # Convert to numeric, handling missing values
    df['diameter_3d'] = pd.to_numeric(df['diameter_3d'].replace('.', 0))
    df['AVERAGE_GAZE_X'] = pd.to_numeric(df['AVERAGE_GAZE_X'].replace('.', np.nan))
    df['AVERAGE_GAZE_Y'] = pd.to_numeric(df['AVERAGE_GAZE_Y'].replace('.', np.nan))
    
    # Create event onset column and sample index within trial
    df['eventOnset'] = df['SAMPLE_MESSAGE'].astype(str)
    df['sample_idx'] = df.groupby('TRIALID').cumcount()
    
    # Pivot to trial × timestamp structure (more efficient than multiple pivots)
    pivot_kwargs = {
        'index': 'TRIALID',
        'columns': 'sample_idx'
    }
    
    # Create all pivoted dataframes at once
    pupil_wide = df.pivot_table(
        values='diameter_3d',
        **pivot_kwargs
    ).reset_index().iloc[:, :max_samples + 1]
    
    gaze_x_wide = df.pivot_table(
        values='AVERAGE_GAZE_X',
        **pivot_kwargs
    ).reset_index().iloc[:, :max_samples + 1]
    
    gaze_y_wide = df.pivot_table(
        values='AVERAGE_GAZE_Y',
        **pivot_kwargs
    ).reset_index().iloc[:, :max_samples + 1]
    
    events_wide = df.pivot_table(
        values='eventOnset',
        **pivot_kwargs,
        aggfunc='first'  # Use first value if multiple
    ).reset_index().iloc[:, :max_samples + 1]
    
    return {
        'pupil': pupil_wide,
        'gaze_x': gaze_x_wide,
        'gaze_y': gaze_y_wide,
        'events': events_wide
    }


def process_all_sessions(
    file_list,
    tracker_path,
    eye_column=SELECTED_EYE,
    max_samples=MAX_SAMPLES_PER_TRIAL
):
    """
    Process multiple eye tracking sessions and concatenate results.
    
    Parameters
    ----------
    file_list : list
        List of session folder names to process
    tracker_path : Path
        Base path to eye tracker data
    eye_column : str
        Column name for pupil measurement
    max_samples : int
        Maximum samples per trial
    
    Returns
    -------
    dict
        Dictionary with concatenated dataframes:
        'pupil_raw', 'gaze_x_raw', 'gaze_y_raw', 'events'
    """
    # Initialize result containers
    all_pupil = []
    all_gaze_x = []
    all_gaze_y = []
    all_events = []
    
    # Process each session
    for session_name in file_list:
        print(f'Processing: {session_name}')
        
        # Find eye tracker file
        search_pattern = str(tracker_path / session_name / "*.txt")
        tracker_files = glob.glob(search_pattern)
        
        if not tracker_files:
            print(f'  Warning: No tracker file found for {session_name}')
            continue
        
        # Load and preprocess
        raw_data = load_eyetracker_file(tracker_files[0], RELEVANT_COLUMNS)
        
        # Transform to trial × timestamp structure
        trial_data = prepare_trial_data(raw_data, eye_column, max_samples)
        
        # Collect results
        all_pupil.append(trial_data['pupil'])
        all_gaze_x.append(trial_data['gaze_x'])
        all_gaze_y.append(trial_data['gaze_y'])
        all_events.append(trial_data['events'])
        
        print(f'  ✓ Completed: {session_name}')
    
    # Concatenate all sessions
    print('\nConcatenating all sessions...')
    results = {
        'pupil_raw': pd.concat(all_pupil, ignore_index=True),
        'gaze_x_raw': pd.concat(all_gaze_x, ignore_index=True),
        'gaze_y_raw': pd.concat(all_gaze_y, ignore_index=True),
        'events': pd.concat(all_events, ignore_index=True)
    }
    
    print(f'Total trials: {len(results["pupil_raw"])}')
    
    return results

def read_psychopy(readIn,PATH_PSYCHOPY):   
    psyFull = pd.DataFrame()
    for f in readIn:
        f = str(f)

        trlInfo = pd.read_csv(glob.glob(PATH_PSYCHOPY+f+ '/*.csv')[0])
        cols = trlInfo[['probe.started','probeStart','probe.stopped','probeEnd','probeMouse.x','probeMouse.y',
        'probeMouse.leftButton','probeMouse.midButton','probeMouse.rightButton','probeMouse.time','probeMouse.mouseOnProbes',
        'probeLocationArr','rspIndex','rsp','rspPath','rt','timeEachClick','itemEachClick','missingRsp','ITI.started','itiStart','ITI.stopped','itiEnd']]
        newDf = trlInfo[~trlInfo.cuedItem.isnull()].reset_index(drop = True)

        newDf[['probe.started','probeStart','probe.stopped','probeEnd','probeMouse.x','probeMouse.y',
        'probeMouse.leftButton','probeMouse.midButton','probeMouse.rightButton','probeMouse.time','probeMouse.mouseOnProbes',
        'probeLocationArr','rspIndex','rsp','rspPath','rt','timeEachClick','itemEachClick','missingRsp','ITI.started','itiStart','ITI.stopped','itiEnd']] = cols.dropna(axis = 0, how ='all').reset_index(drop=True)


        psyFull = pd.concat([psyFull,newDf])


    psyFull = psyFull.drop_duplicates()



# Main execution
if __name__ == "__main__":
    # Select files to process
    all_files = sorted(os.listdir(PATH_PSYCHOPY))
    
    #subjects to handle
    readIn = sorted([f for f in os.listdir(PATH_PSYCHOPY)])
    files_to_process = readIn
    
    print(f"Processing {len(files_to_process)} sessions:")
    for f in files_to_process:
        print(f"  - {f}")
    print()
    
    # Process all sessions
    processed_data = process_all_sessions(
        file_list=readIn,
        tracker_path=PATH_EYETRACKER,
        eye_column=SELECTED_EYE,
        max_samples=MAX_SAMPLES_PER_TRIAL
    )
    
    # Access results
    pupil_full_trial_raw = processed_data['pupil_raw']
    sacc_full_trial_x_raw = processed_data['gaze_x_raw']
    sacc_full_trial_y_raw = processed_data['gaze_y_raw']
    event_df = processed_data['events']
    
    print("\n✓ All processing complete!")