## Project 0: Titanic Survival Exploration
#In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank,
#resulting in the deaths of most of its passengers and crew.

import numpy as np
import pandas as pd

# RMS Titanic data visualization code
#from titanic_visualizations import survival_stats
from IPython.display import display
#matplotlib inline

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)
# Print the first few entries of the RMS Titanic data
display(full_data.head())
