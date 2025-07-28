from .main import *
from .logo import *

import os
import sys
import builtins

# Detailed Documentation

detailed_doc = """                                                               
The `model_tuner` library is a versatile and powerful tool designed to 
facilitate the training, tuning, and evaluation of machine learning models. 
It supports various functionalities such as handling imbalanced data, applying 
different scaling and imputation techniques, calibrating models, and conducting 
cross-validation. This library is particularly useful for hyperparameter tuning
and ensuring optimal performance across different metrics.

PyPI: https://pypi.org/project/model-tuner/
Documentation: https://uclamii.github.io/model_tuner/


Version: 0.0.34b1

"""

# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc


__version__ = "0.0.34b1"
__author__ = "Arthur Funnell, Leonid Shpaner, Panayiotis Petousis"
__email__ = "lshpaner@ucla.edu; alafunnell@gmail.com; pp89@ucla.edu"


# Define the custom help function
def custom_help(obj=None):
    """
    Custom help function to dynamically include ASCII art in help() output.
    """
    if (
        obj is None or obj is sys.modules[__name__]
    ):  # When `help()` is called for this module
        print(model_tuner_logo)  # Print ASCII art first
        print(detailed_doc)  # Print the detailed documentation
    else:
        original_help(obj)  # Use the original help for other objects


# Backup the original help function
original_help = builtins.help

# Override the global help function in builtins
builtins.help = custom_help
