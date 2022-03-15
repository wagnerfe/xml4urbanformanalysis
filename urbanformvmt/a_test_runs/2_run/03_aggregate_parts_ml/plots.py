"""
Module that applies
- Linear Regression
- XGBoost
on pre calculated dataset

In this case, only street features are calculated

@authors: Nikola, Felix W
last modified: 25.02.2021

"""

# imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
#import seaborn as sns
from collections import Counter
import xgboost
import os 
import sys 
import time 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Set Options
pd.options.display.max_rows = 999

# Set path of file
#Path to files
path_file = '/p/projects/vwproject/felix_files/data/output/ML/model_in_lb_ub.csv' 


#Add time
start_overall = time.time()


