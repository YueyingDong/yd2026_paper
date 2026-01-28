# functions/__init__.py

"""
Eye tracking data processing utilities.

Contains functions for:
- Blink detection and removal
- Gaze processing and saccade identification
- Speed/velocity calculations  
- Signal smoothing and filtering
- Trial quality assessment
- IPA (index of pupillary activity) calculation
"""

__version__ = "0.1.0"

# Import main pipeline function
from .processPup import *
from .processGaze import *

# Import cleaning function
from .deBlink import *

# Import read in function
from .readRaw import *

# Import permutation tests
from .permutationFunc import *

# Import utilities
from .helperFunc import *

# Import IPA
from .ipa import *


# Define what's available with "from functions import *"
__all__ = [

    #process
    'process_pupil_data_pipeline',
    'process_gaze_data_pipeline',
    'detect_and_remove_blinks',
    'calculate_gaze_shifts',

    # Commonly used utilities
    'find_consecutive_groups',
    'smooth_signal',
    'fs', # Set figure size
    'linear_interpolate',

    # Permutation
    'cluster_permutation_test',
    'plot_permutation_results',

    # Statistical functions
    'calculate_median_variance',
    'cohenD'
]