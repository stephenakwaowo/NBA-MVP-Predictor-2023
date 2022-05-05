# MVP-Predictor
This is a model that uses three different machine learning regression tools (Linear Regression, XGBoost and Random Forest) to help predict the 2022 NBA MVP, and to see which past MVPs the models would have chosen.

These models, and the inspiration of the project is based on the work by David Yoo.

The link to the TDS article by Yoo, is [here](https://towardsdatascience.com/predicting-the-next-nba-mvp-using-machine-learning-62615bfcff75), and this is the link to his [github repository](https://github.com/DavidYoo912/nba_mvp_project).

### Installation and Packages


```Python
import os
import numpy as np
import pandas as pd
import dataframe_image as dfi
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
pd.set_option('display.max_columns', None)
```
