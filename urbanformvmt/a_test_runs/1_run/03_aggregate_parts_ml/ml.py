"""
Module that aggregates all feature parts in one large dataframe, called df_all_parts_ft

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
path_file = '/data/metab/urbanformvmt/data/output/ml_tests/model_input.csv' 
path_output = '/data/metab/urbanformvmt/data/output/ml_tests/model_output.csv' 

#Add time
start_overall = time.time()

# feature groups
features_groups = {

'dist_based_bu_100': ['std_block_orientation_within_buffer_100', 'av_block_orientation_within_buffer_100', 'av_block_av_footprint_area_within_buffer_100', 
'std_block_footprint_area_within_buffer_100', 'av_block_footprint_area_within_buffer_100', 'std_block_length_within_buffer_100', 
'av_block_length_within_buffer_100', 'blocks_within_buffer_100', 'std_orientation_within_buffer_100', 'av_orientation_within_buffer_100', 
'std_convexity_within_buffer_100', 'av_convexity_within_buffer_100', 'std_elongation_within_buffer_100', 'av_elongation_within_buffer_100', 
'std_footprint_area_within_buffer_100', 'av_footprint_area_within_buffer_100', 'total_ft_area_within_buffer_100', 'buildings_within_buffer_100'], 

'dist_based_bu_500': ['std_block_orientation_within_buffer_500', 'av_block_orientation_within_buffer_500', 'av_block_av_footprint_area_within_buffer_500', 
'std_block_footprint_area_within_buffer_500', 'av_block_footprint_area_within_buffer_500', 'std_block_length_within_buffer_500', 
'av_block_length_within_buffer_500', 'blocks_within_buffer_500', 'std_orientation_within_buffer_500', 'av_orientation_within_buffer_500', 
'std_convexity_within_buffer_500', 'av_convexity_within_buffer_500', 'std_elongation_within_buffer_500', 'av_elongation_within_buffer_500', 
'std_footprint_area_within_buffer_500', 'av_footprint_area_within_buffer_500', 'total_ft_area_within_buffer_500', 'buildings_within_buffer_500'], 

'str_int_closest':  ['street_closeness_500_closest_road', 'street_betweeness_global_closest_road', 'street_closeness_global_closest_road', 'street_openness_closest_road', 
'street_width_std_closest_road', 'street_width_av_closest_road', 'street_length_closest_road', 'distance_to_closest_road', 'distance_to_closest_intersection'], 

'str_int_100':  ['street_closeness_500_av_inter_buffer_100', 'street_closeness_500_max_inter_buffer_100', 'street_betweeness_global_av_inter_buffer_100', 
'street_betweeness_global_max_inter_buffer_100', 'street_width_std_inter_buffer_100', 'street_width_av_inter_buffer_100', 
'street_length_total_inter_buffer_100', 'street_length_std_within_buffer_100', 'street_length_av_within_buffer_100', 
'street_length_total_within_buffer_100', 'intersection_count_within_buffer_100'], 

'str_int_500':  ['street_closeness_500_av_inter_buffer_500', 
'street_closeness_500_max_inter_buffer_500', 'street_betweeness_global_av_inter_buffer_500', 'street_betweeness_global_max_inter_buffer_500', 
'street_width_std_inter_buffer_500', 'street_width_av_inter_buffer_500', 'street_length_total_inter_buffer_500', 'street_length_std_within_buffer_500', 
'street_length_av_within_buffer_500', 'street_length_total_within_buffer_500', 'intersection_count_within_buffer_500'], 

}

## preprocessing
# read data
df = pd.read_csv(path_file)

# for test
#df = df.sample(n=100)
#df.reset_index(drop=True)

# split
#df_copy = df.copy()
df_train = df.sample(frac=0.8,random_state=0)
df_results = df.drop(df_train.index)
del df

X_train = df_train.drop(['tripdistancemeters','tripid','lengthoftrip','geometry'], axis=1)
X_valid = df_results.drop(['tripdistancemeters','tripid','lengthoftrip','geometry'],axis=1)

y_train = df_train[['tripdistancemeters']]
y_valid = df_results[['tripdistancemeters']]
del df_train

df_results = df_results[['tripdistancemeters','tripid','geometry']]
df_results.rename(columns={'tripdistancemeters':'y_valid'},inplace=True) 

print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))
print('X_valid: {}'.format(X_valid.shape))
print('y_valid: {}'.format(y_valid.shape))


### linear regression
# get time
start = time.time()
# Initialise
model = LinearRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_valid)

df_results ['y_predict_ols'] = y_predict
df_results ['error_ols'] = df_results ['y_predict_ols'] - df_results ['y_valid']
df_results ['error_abs_ols'] = abs(df_results ['error_ols'])

# print time
end = time.time()
last = divmod(end - start, 60)
print('LinReg in {} minutes {} seconds'.format(last[0],last[1])) 

## metrics
print('Overall metrics...')
print('R2: {}'.format(metrics.r2_score(df_results['y_valid'],df_results['y_predict_ols'])))
print('MAE: {} m'.format(metrics.mean_absolute_error(df_results['y_valid'],df_results['y_predict_ols'])))
print('RMSE: {} m'.format(np.sqrt(metrics.mean_squared_error(df_results['y_valid'],df_results['y_predict_ols']))))



### XGBoost
# get time
start = time.time()

model = xgboost.XGBRegressor()
model.fit(X_train,y_train)
y_predict_xgb = model.predict(X_valid)

df_results['y_predict_xgb'] = y_predict_xgb
df_results['error_xgb'] = df_results['y_predict_xgb'] - df_results['y_valid']
df_results['error_abs_xgb'] = abs(df_results['error_xgb'])

# print time
end = time.time()
last = divmod(end - start, 60)
print('XGBoost done in {} minutes {} seconds'.format(last[0],last[1])) 

## metrics
print('Overall metrics...')
print('R2: {}'.format(metrics.r2_score(df_results['y_valid'],df_results['y_predict_xgb'])))
print('MAE: {} m'.format(metrics.mean_absolute_error(df_results['y_valid'],df_results['y_predict_xgb'])))
print('RMSE: {} m'.format(np.sqrt(metrics.mean_squared_error(df_results['y_valid'],df_results['y_predict_xgb']))))


# individual feature importance
ft_importance = pd.DataFrame.from_dict(model.get_booster().get_score(importance_type="gain"), columns=['importance'],orient='index')
ft_importance = ft_importance.sort_values(by=['importance'], ascending=False)
ft_importance['feature'] = ft_importance.index
print(ft_importance)

# feature group importance
metrics_df = pd.DataFrame()
for ft_group in features_groups:

    print(ft_group)
    fts = ft_importance[ft_importance['feature'].isin(features_groups[ft_group])]
    ft_grp_av_imp = fts['importance'].sum() / len(features_groups[ft_group])
    print(ft_grp_av_imp)
    metrics_df[ft_group] = ft_grp_av_imp

end_overall = time.time()
last = divmod(end_overall - start_overall, 60)
print('Overall ML done in {} minutes {} seconds'.format(last[0],last[1])) 


## save df_results
df_results.to_csv(path_output,index=False)
