"""  

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$      __  __           _      _   _____                          $ 
$     |  \/  | ___   __| | ___| | |_   _|   _ _ __   ___ _ __     $
$     | |\/| |/ _ \ / _` |/ _ \ |   | || | | | '_ \ / _ \ '__|    $
$     | |  | | (_) | (_| |  __/ |   | || |_| | | | |  __/ |       $
$     |_|  |_|\___/ \__,_|\___|_|   |_| \__,_|_| |_|\___|_|       $
$                                                                 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                                               
The `model_tuner` library is a versatile and powerful tool designed to 
facilitate the training, evaluation, and tuning of machine learning models. 
It supports various functionalities such as handling imbalanced data, applying 
different scaling and imputation techniques, calibrating models, and conducting 
cross-validation. This library is particularly useful for model selection, 
hyperparameter tuning, and ensuring optimal performance across different metrics.

Version: 0.0.17a

"""

__version__ = "0.0.17a"
__author__ = "Arthur Funnell, Leonid Shpaner, Panayiotis Petousis"
__email__ = "lshpaner@ucla.edu; alafunnell@gmail.com; pp89@ucla.edu"

from .main import *
