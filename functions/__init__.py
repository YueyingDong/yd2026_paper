# functions/__init__.py

"""
Eye tracking data processing utilities.

Contains functions for:
- Blink detection and removal
- Speed/velocity calculations  
- Signal smoothing and filtering
- Trial quality assessment
"""

__version__ = "0.1.0"

# Import main pipeline function
from .processPup import *

# Import cleaning function
from .deBlink import *

# Import read in function
from .readRaw import *

# Import permutation tests
from .permutationFunc import *

# Import utilities
from .helperFunc import *


# Define what's available with "from functions import *"
__all__ = [

    #process
    'process_pupil_data_pipeline',

    # Commonly used utilities
    'find_consecutive_groups',
    'smooth_signal',
    'fs', # Set figure size

    # Permutation
    'cluster_permutation_test',
    'plot_permutation_results',

    # Statistical functions
    'calculate_median_variance',
]