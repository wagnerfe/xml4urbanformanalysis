"""
File to runs trees with shap function

Author: felix
Date: 26.08.2021
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
#from time import time
import time


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import plot_partial_dependence
from sklearn import metrics
from e_utils.utils import make_pizza, pizza_oppo_split
from PyALE import ale
import xgboost
import shap
import pickle
import random

import warnings
warnings.filterwarnings('ignore')

def xgb_spatial_cv_shap_save_all(df,
                path,
                target="mean",
                kfold = 5,
                optimize_hype = False,
                calc_ale = False,
                print_split=False,
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates Shap values in each folds and saves them as dump in pickle!

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)

    # Do spatial cross validation
    out=[]
    df_x = pd.DataFrame()
    
    if calc_ale:
        ale_out = pd.DataFrame()
    
    for i in range(kfold):
        print('cross validation in sector ',i)

        #########################
        ## 2. Get train & test data
        #########################
        # get train & test part via make_pizza func
        df_pizza = make_pizza(df,do_plot=print_split,share = 1/kfold,angle=i*2*np.pi/kfold)        
        df_train = df_pizza[df_pizza.pizza.str.match('train')]
        df_train = df_train.drop(columns='pizza')
        df_test = df_pizza[df_pizza.pizza.str.match('test')]
        df_test = df_test.drop(columns='pizza')

        # assign features to X
        X_train = df_train[[col for col in df_train.columns if "feature_" in col]]
        X_train["feature_noise"] = np.random.normal(size=len(df_train))
        X_test = df_test[[col for col in df_test.columns if "feature_" in col]]
        X_test["feature_noise"] = np.random.normal(size=len(df_test))
        
        # assign target to y
        y_train = df_train[target]
        y_test = df_test[target]

        # create results df
        df_test = df_test[['tripdistancemeters','hex_id','geometry']]
        df_test.rename(columns={'tripdistancemeters':'y_test'},inplace=True) 
        
        #########################
        ## 3. Tune Hyperparameters
        #########################
        if optimize_hype:
            print("Optimizing Hyperparameters..")
            #start = time()
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}             
            tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            #end = time()
            print("Best Parameters found: ", tuning.best_params_)
            #print("After {} s".format(end - start))

            n_parameter = tuning.best_params_["n_estimators"]
            lr_parameter = tuning.best_params_["learning_rate"] 
            md_parameter = tuning.best_params_["max_depth"] 
        else:
            n_parameter = 100
            lr_parameter = 0.05
            md_parameter = 2

        #########################
        ## 4. Training on kfold
        #########################
        tree = xgboost.XGBRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter,
            importance_type = 'total_gain')

        # Train the model
        model = tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)

        #########################
        ## 5. Results
        #########################
        # Append results to df_results
        df_test['y_predict_trees'] = y_predict
        df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
        df_test['error_abs_trees'] = abs(df_test['error_trees'])

        # Metric of model
        r2_model = tree.score(X_train,y_train)
        # Metrics of prediction
        r2_pred = tree.score(X_test,y_test)
        mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
        rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))

        #Plot metrics each round!
        print('Metrics of prediction in sector',i)
        print('R2 Model: {}'.format(r2_model))
        print('R2 Predict: {}'.format(r2_pred))
        print('MAE: {} m'.format(mae_pred))
        print('RMSE: {} m'.format(rmse_pred))        


        #########################
        ## 6. Feature Importance
        #########################
        # Calculate feature importances
        feature_importance = model.feature_importances_

        #########################
        ## 7. Partial Dependence
        #########################
        if calc_ale:
            for j,col in enumerate(X_train):
                ale_eff = ale(X=X_train, model = model, feature = [X_train.columns[j]], grid_size=50, include_CI=False, plot=False).reset_index()
                ale_eff = ale_eff.rename(columns={'eff':col+'_eff','size':col+'_size'})
                ale_out = pd.concat([ale_out,ale_eff],axis=1)
        
        if path is not None:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_train)
            path_shap = path + '_shap_val_fold_'+str(i)+'.pkl'
            pickle.dump(shap_values, open(path_shap,"wb"))
            time.sleep(2)
            print('save fold finish --> ',i)
            
        #########################
        ## 8. Append 
        #########################
        # Append stats in output df
        val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
        val_list.extend(feature_importance.tolist())
        out.append(val_list)

        # Append x_train
        x_train_sorted = pd.DataFrame()
        # Sort column values to plot average feature distribution
        for col in X_train.columns:
            x_train_sorted[col] = X_train[col].sort_values().reset_index(drop=True)
        # Append x_train in df_x to save for mean feat distribution
        df_x = pd.concat([df_x,x_train_sorted],axis=1)

    #########################
    ## 9. Mean over k folds 
    #########################    
    # Get Mean of stats and feature importance
    col = ['n_parameter','lr_parameter','md_parameter','r2_model','r2_pred','mae_pred','rmse_pred']
    col.extend(X_train.columns.values.tolist())
    df_cv = pd.DataFrame(out,columns=tuple(col))
    df_out = pd.DataFrame(df_cv.mean()).transpose()

    # Get Mean of sorted x_train 
    df_x_mean = pd.DataFrame()
    for col in df_x.columns.unique():
        df_x_mean[col] = df_x[col].mean(axis=1)

    if calc_ale:
        # Get Mean of ALEs
        df_ale_mean = pd.DataFrame()
        for idx, colx in enumerate(X_train):
            df_ale_mean[colx] = ale_out[colx].mean(axis=1)
            df_ale_mean[colx+'_eff'] = ale_out[colx+'_eff'].mean(axis=1)
            df_ale_mean[colx+'_size'] = ale_out[colx+'_size'].mean(axis=1)

    #########################
    ## 10. Print Metrics 
    #########################
    print('Best parameters found: learning rate: {}, max_depth {}, n_estimators: {}'
                                                    .format((round(df_cv.iloc[0][1],3)),
                                                    (round(df_out.iloc[0][2],3)),
                                                    (round(df_out.iloc[0][0],3))))
    print('--------------------')
    print('Metrics of model after cv')
    print('R2 Model: {}'.format(round(df_out.iloc[0][3],3)))
    print('--------------------')
    print('Metrics of prediction after cv')
    print('R2 Pred.: {}'.format(round(df_out.iloc[0][4],3)))
    print('MAE Pred.: {} m'.format(round(df_out.iloc[0][5],3)))
    print('RMSE Pred.: {} m'.format(round(df_out.iloc[0][6],3)))
    print('--------------------')
    print('Feature importance')
    print('--------------------')
    
    # print importance sorted 
    feat_imp = df_out[df_x_mean.columns]
    sorted_idx = np.argsort(feat_imp.iloc[0])[::-1]
    for feature, p in zip(feat_imp.columns[sorted_idx], sorted(feat_imp.iloc[0],reverse=True)):
        print("{}: {}".format(feature,round(p,3)))

    return (df_out, df_x_mean, df_ale_mean) if calc_ale else (df_out, df_x_mean)  


def xgb_spatial_cv_shap(df,
                path,
                target="mean",
                kfold = 5,
                optimize_hype = False,
                calc_ale = False,
                print_split=False,
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates ALE Shap values on whole dataset!

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)

    # Do spatial cross validation
    out=[]
    df_x = pd.DataFrame()
    
    if calc_ale:
        ale_out = pd.DataFrame()
    
    for i in range(kfold):
        print('cross validation in sector ',i)

        #########################
        ## 2. Get train & test data
        #########################
        # get train & test part via make_pizza func
        df_pizza = make_pizza(df,do_plot=print_split,share = 1/kfold,angle=i*2*np.pi/kfold)        
        df_train = df_pizza[df_pizza.pizza.str.match('train')]
        df_train = df_train.drop(columns='pizza')
        df_test = df_pizza[df_pizza.pizza.str.match('test')]
        df_test = df_test.drop(columns='pizza')

        # assign features to X
        X_train = df_train[[col for col in df_train.columns if "feature_" in col]]
        X_train["feature_noise"] = np.random.normal(size=len(df_train))
        X_test = df_test[[col for col in df_test.columns if "feature_" in col]]
        X_test["feature_noise"] = np.random.normal(size=len(df_test))
        
        # assign target to y
        y_train = df_train[target]
        y_test = df_test[target]

        # create results df
        df_test = df_test[['tripdistancemeters','hex_id','geometry']]
        df_test.rename(columns={'tripdistancemeters':'y_test'},inplace=True) 
        
        #########################
        ## 3. Tune Hyperparameters
        #########################
        if optimize_hype:
            print("Optimizing Hyperparameters..")
            #start = time()
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}             
            tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            #end = time()
            print("Best Parameters found: ", tuning.best_params_)
            #print("After {} s".format(end - start))

            n_parameter = tuning.best_params_["n_estimators"]
            lr_parameter = tuning.best_params_["learning_rate"] 
            md_parameter = tuning.best_params_["max_depth"] 
        else:
            n_parameter = 100
            lr_parameter = 0.05
            md_parameter = 2

        #########################
        ## 4. Training on kfold
        #########################
        tree = xgboost.XGBRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter,
            importance_type = 'total_gain')

        # Train the model
        model = tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)

        #########################
        ## 5. Results
        #########################
        # Append results to df_results
        df_test['y_predict_trees'] = y_predict
        df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
        df_test['error_abs_trees'] = abs(df_test['error_trees'])

        # Metric of model
        r2_model = tree.score(X_train,y_train)
        # Metrics of prediction
        r2_pred = tree.score(X_test,y_test)
        mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
        rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))

        #Plot metrics each round!
        print('Metrics of prediction in sector',i)
        print('R2 Model: {}'.format(r2_model))
        print('R2 Predict: {}'.format(r2_pred))
        print('MAE: {} m'.format(mae_pred))
        print('RMSE: {} m'.format(rmse_pred))        

        #########################
        ## 8. Append 
        #########################
        # Append stats in output df
        val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
        out.append(val_list)

    # -------------- End k fold CROSSVALIDATION -----------------------#

    #########################
    ## 9. Mean over k folds 
    #########################    
    # Get Mean of stats 
    col = ['n_parameter','lr_parameter','md_parameter','r2_model','r2_pred','mae_pred','rmse_pred']
    #col.extend(X_train.columns.values.tolist())
    # collect in one df
    df_cv = pd.DataFrame(data = out, columns = col)
    # calc average 
    df_out = pd.DataFrame(df_cv.mean()).transpose()

    print('------------------------')
    print('Average parameters found: learning rate: {}, max_depth {}, n_estimators: {}'
                                                    .format((round(df_cv.iloc[0][1],3)),
                                                    (round(df_out.iloc[0][2],3)),
                                                    (round(df_out.iloc[0][0],3))))
    print('------------------------')
    print('R2 Model: {} '.format(round(df_out.iloc[0][3],3)))
    print('R2 Prediction: {} '.format(round(df_out.iloc[0][4],3)))
    print('MAE Prediction: {} '.format(round(df_out.iloc[0][5],3)))
    print('RMSE Prediction: {} '.format(round(df_out.iloc[0][6],3)))
    print('------------------------')
    
    ##################################################
    ## 10 Take best Hyperparameters from kfolds and assign data
    ##################################################
    # assign features to X
    X_train_ws = df[[col for col in df.columns if "feature_" in col]]
    X_train_ws["feature_noise"] = np.random.normal(size=len(df))
    
    # assign target to y
    y_train_ws = df[target]

    # assign mean values from folds 
    md_ave = int(round(df_out.iloc[0][2],0))
    n_ave = int(round(df_out.iloc[0][0],0))
    lr_ave = df_cv.iloc[0][1]
    
    ##################################################
    ## 11. Training whole sample
    ##################################################
    print('------------------------')
    print('Final training of model with hyperparameter from CV')
    tree = xgboost.XGBRegressor(
        max_depth=md_ave, 
        n_estimators=n_ave, 
        learning_rate=lr_ave,
        importance_type = 'total_gain')

    # Train the model on ws
    model = tree.fit(X_train_ws, y_train_ws)

    # Metric of model on ws
    r2_model = tree.score(X_train_ws,y_train_ws)
    
    #########################
    ## 9. Stats on whole sample
    #########################
    # Calculate feature importances
    feature_importance = model.feature_importances_
    
    if calc_ale:
            ale_out = pd.DataFrame()
            for j,col in enumerate(X_train_ws):
                ale_eff = ale(X=X_train_ws, model = model, feature = [X_train_ws.columns[j]], grid_size=50, include_CI=False, plot=False).reset_index()
                ale_eff = ale_eff.rename(columns={'eff':col+'_eff','size':col+'_size'})
                ale_out = pd.concat([ale_out,ale_eff],axis=1)
        
    if path is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_train_ws)
        path_shap = path + '_shap_val_ws.pkl'
        pickle.dump(shap_values, open(path_shap,"wb"))


    print('--------------------')
    print('Metrics of model on ws')
    print('R2 Model: {}'.format(round(r2_model,3)))

   # print importance sorted 
    #feat_imp = df_out[X_train_raw.columns]
    feat_imp = feature_importance
    sorted_idx = np.argsort(feat_imp)[::-1]
    for feature, p in zip(X_train_ws.columns[sorted_idx], sorted(feat_imp,reverse=True)):
        print("{}: {}".format(
            feature,
            round(p, 3)
            ))

    return (df_out, ale_out) if calc_ale else (df_out)  


def xgb_spatial_oppo_cv_shap_save_all(df,
                path,
                target="mean",
                kfold = 5,
                optimize_hype = False,
                calc_ale = False,
                print_split=False,
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates Shap values in each folds and saves them as dump in pickle!
    divides k-folds 180° oppo

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)

    # Do spatial cross validation
    out=[]
    df_x = pd.DataFrame()
    
    if calc_ale:
        ale_out = pd.DataFrame()
    
    for i in range(kfold):
        print('cross validation in sector ',i)
        """
        #########################
        ## 2. Get train & test data
        #########################
        # get train & test part via make_pizza func
        df_pizza = make_pizza(df,do_plot=print_split,share = 1/(2*kfold),angle=i*2*np.pi/(2*kfold))
        # take the fold from the oppo site
        df_pizza_oppo = make_pizza(df,do_plot=print_split,share = 1/(2*kfold),angle=(i+0.5*kfold)*2*np.pi/(2*kfold))                
        # put the training in the train and the test in the test dataset
        df_train = df[(df_pizza.pizza=='train') & (df_pizza_oppo.pizza=='train')]    
        df_test = df[(df_pizza.pizza=='test') | (df_pizza_oppo.pizza=='test')]
        """
        df_train = df.sample(frac=0.8,random_state=i)
        df_test = df.drop(df_train.index)

        # assign features to X
        X_train = df_train[[col for col in df_train.columns if "feature_" in col]]
        X_train["feature_noise"] = np.random.normal(size=len(df_train))
        X_test = df_test[[col for col in df_test.columns if "feature_" in col]]
        X_test["feature_noise"] = np.random.normal(size=len(df_test))
        
        # assign target to y
        y_train = df_train[target]
        y_test = df_test[target]

        # create results df
        df_test = df_test[['tripdistancemeters','hex_id','geometry']]
        df_test.rename(columns={'tripdistancemeters':'y_test'},inplace=True) 
        
        #########################
        ## 3. Tune Hyperparameters
        #########################
        if optimize_hype:
            print("Optimizing Hyperparameters..")
            #start = time()
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}             
            tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            #end = time()
            print("Best Parameters found: ", tuning.best_params_)
            #print("After {} s".format(end - start))

            n_parameter = tuning.best_params_["n_estimators"]
            lr_parameter = tuning.best_params_["learning_rate"] 
            md_parameter = tuning.best_params_["max_depth"] 
        else:
            n_parameter = 100
            lr_parameter = 0.05
            md_parameter = 2

        #########################
        ## 4. Training on kfold
        #########################
        tree = xgboost.XGBRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter,
            importance_type = 'total_gain')

        # Train the model
        model = tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)

        #########################
        ## 5. Results
        #########################
        # Append results to df_results
        df_test['y_predict_trees'] = y_predict
        df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
        df_test['error_abs_trees'] = abs(df_test['error_trees'])

        # Metric of model
        r2_model = tree.score(X_train,y_train)
        # Metrics of prediction
        r2_pred = tree.score(X_test,y_test)
        mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
        rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))

        #Plot metrics each round!
        print('Metrics of prediction in sector',i)
        print('R2 Model: {}'.format(r2_model))
        print('R2 Predict: {}'.format(r2_pred))
        print('MAE: {} m'.format(mae_pred))
        print('RMSE: {} m'.format(rmse_pred))        


        #########################
        ## 6. Feature Importance
        #########################
        # Calculate feature importances
        feature_importance = model.feature_importances_

        #########################
        ## 7. Partial Dependence
        #########################
        if calc_ale:
            for j,col in enumerate(X_train):
                ale_eff = ale(X=X_train, model = model, feature = [X_train.columns[j]], grid_size=50, include_CI=False, plot=False).reset_index()
                ale_eff = ale_eff.rename(columns={'eff':col+'_eff','size':col+'_size'})
                ale_out = pd.concat([ale_out,ale_eff],axis=1)
        
        if path is not None:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_train)
            path_shap = path + '_shap_val_fold_'+str(i)+'.pkl'
            pickle.dump(shap_values, open(path_shap,"wb"))
            
        #########################
        ## 8. Append 
        #########################
        # Append stats in output df
        val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
        val_list.extend(feature_importance.tolist())
        out.append(val_list)

        # Append x_train
        x_train_sorted = pd.DataFrame()
        # Sort column values to plot average feature distribution
        for col in X_train.columns:
            x_train_sorted[col] = X_train[col].sort_values().reset_index(drop=True)
        # Append x_train in df_x to save for mean feat distribution
        df_x = pd.concat([df_x,x_train_sorted],axis=1)

    #########################
    ## 9. Mean over k folds 
    #########################    
    # Get Mean of stats and feature importance
    col = ['n_parameter','lr_parameter','md_parameter','r2_model','r2_pred','mae_pred','rmse_pred']
    col.extend(X_train.columns.values.tolist())
    df_cv = pd.DataFrame(out,columns=tuple(col))
    df_out = pd.DataFrame(df_cv.mean()).transpose()

    # Get Mean of sorted x_train 
    df_x_mean = pd.DataFrame()
    for col in df_x.columns.unique():
        df_x_mean[col] = df_x[col].mean(axis=1)

    if calc_ale:
        # Get Mean of ALEs
        df_ale_mean = pd.DataFrame()
        for idx, colx in enumerate(X_train):
            df_ale_mean[colx] = ale_out[colx].mean(axis=1)
            df_ale_mean[colx+'_eff'] = ale_out[colx+'_eff'].mean(axis=1)
            df_ale_mean[colx+'_size'] = ale_out[colx+'_size'].mean(axis=1)

    #########################
    ## 10. Print Metrics 
    #########################
    print('Best parameters found: learning rate: {}, max_depth {}, n_estimators: {}'
                                                    .format((round(df_cv.iloc[0][1],3)),
                                                    (round(df_out.iloc[0][2],3)),
                                                    (round(df_out.iloc[0][0],3))))
    print('--------------------')
    print('Metrics of model after cv')
    print('R2 Model: {}'.format(round(df_out.iloc[0][3],3)))
    print('--------------------')
    print('Metrics of prediction after cv')
    print('R2 Pred.: {}'.format(round(df_out.iloc[0][4],3)))
    print('MAE Pred.: {} m'.format(round(df_out.iloc[0][5],3)))
    print('RMSE Pred.: {} m'.format(round(df_out.iloc[0][6],3)))
    print('--------------------')
    print('Feature importance')
    print('--------------------')
    
    # print importance sorted 
    feat_imp = df_out[df_x_mean.columns]
    sorted_idx = np.argsort(feat_imp.iloc[0])[::-1]
    for feature, p in zip(feat_imp.columns[sorted_idx], sorted(feat_imp.iloc[0],reverse=True)):
        print("{}: {}".format(
            feature,
            round(p, 4)
            ))

    return (df_out, df_x_mean, df_ale_mean) if calc_ale else (df_out, df_x_mean) 


def xgb_spatial_cv_shap_combine_all(df,
                path,
                target="mean",
                kfold = 5,
                optimize_hype = False,
                calc_ale = False,
                print_split=False,
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates Shap values per fold and saves them in one output which can then be plotted as one figure.

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)

    # Do spatial cross validation
    out=[]
    df_x = pd.DataFrame()
    
    if calc_ale:
        ale_out = pd.DataFrame()

    if path is not None:
        list_shap_values = list()
        list_test_sets = list()  
    
    for i in range(kfold):
        print('cross validation in sector ',i)

        #########################
        ## 2. Get train & test data
        #########################
        # get train & test part via make_pizza func
        df_pizza = make_pizza(df,do_plot=print_split,share = 1/kfold,angle=i*2*np.pi/kfold)        
        df_train = df_pizza[df_pizza.pizza.str.match('train')]
        df_train = df_train.drop(columns='pizza')
        df_test = df_pizza[df_pizza.pizza.str.match('test')]
        df_test = df_test.drop(columns='pizza')

        # assign features to X
        X_train = df_train[[col for col in df_train.columns if "feature_" in col]]
        X_train["feature_noise"] = np.random.normal(size=len(df_train))
        X_test = df_test[[col for col in df_test.columns if "feature_" in col]]
        X_test["feature_noise"] = np.random.normal(size=len(df_test))
        
        # assign target to y
        y_train = df_train[target]
        y_test = df_test[target]

        # create results df
        df_test = df_test[['tripdistancemeters','hex_id','geometry']]
        df_test.rename(columns={'tripdistancemeters':'y_test'},inplace=True) 
        
        #########################
        ## 3. Tune Hyperparameters
        #########################
        if optimize_hype:
            print("Optimizing Hyperparameters..")
            #start = time()
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}             
            tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            #end = time()
            print("Best Parameters found: ", tuning.best_params_)
            #print("After {} s".format(end - start))

            n_parameter = tuning.best_params_["n_estimators"]
            lr_parameter = tuning.best_params_["learning_rate"] 
            md_parameter = tuning.best_params_["max_depth"] 
        else:
            n_parameter = 100
            lr_parameter = 0.05
            md_parameter = 2

        #########################
        ## 4. Training on kfold
        #########################
        tree = xgboost.XGBRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter,
            importance_type = 'total_gain')

        # Train the model
        model = tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)

        #########################
        ## 5. Results
        #########################
        # Append results to df_results
        df_test['y_predict_trees'] = y_predict
        df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
        df_test['error_abs_trees'] = abs(df_test['error_trees'])

        # Metric of model
        r2_model = tree.score(X_train,y_train)
        # Metrics of prediction
        r2_pred = tree.score(X_test,y_test)
        mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
        rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))

        #Plot metrics each round!
        print('Metrics of prediction in sector',i)
        print('R2 Model: {}'.format(r2_model))
        print('R2 Predict: {}'.format(r2_pred))
        print('MAE: {} m'.format(mae_pred))
        print('RMSE: {} m'.format(rmse_pred))        


        #########################
        ## 6. Feature Importance
        #########################
        # Calculate feature importances
        feature_importance = model.feature_importances_

        #########################
        ## 7. Partial Dependence
        #########################
        if calc_ale:
            for j,col in enumerate(X_train):
                ale_eff = ale(X=X_train, model = model, feature = [X_train.columns[j]], grid_size=50, include_CI=False, plot=False).reset_index()
                ale_eff = ale_eff.rename(columns={'eff':col+'_eff','size':col+'_size'})
                ale_out = pd.concat([ale_out,ale_eff],axis=1)
        

        if path is not None:
            # calc shap values with TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            #for each iteration we save the test_set index and the shap_values
            list_shap_values.append(shap_values)
            # in our case test_index = df_pizza[df_pizza.pizza == 'test'].index.values
            list_test_sets.append(df_pizza[df_pizza.pizza == 'test'].index.values)

            
        #########################
        ## 8. Append 
        #########################
        # Append stats in output df
        val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
        val_list.extend(feature_importance.tolist())
        out.append(val_list)

        # Append x_train
        x_train_sorted = pd.DataFrame()
        # Sort column values to plot average feature distribution
        for col in X_train.columns:
            x_train_sorted[col] = X_train[col].sort_values().reset_index(drop=True)
        # Append x_train in df_x to save for mean feat distribution
        df_x = pd.concat([df_x,x_train_sorted],axis=1)

    #########################
    ## 9. Mean over k folds 
    #########################    
    # Get Mean of stats and feature importance
    col = ['n_parameter','lr_parameter','md_parameter','r2_model','r2_pred','mae_pred','rmse_pred']
    col.extend(X_train.columns.values.tolist())
    df_cv = pd.DataFrame(out,columns=tuple(col))
    df_out = pd.DataFrame(df_cv.mean()).transpose()

    # Get Mean of sorted x_train 
    df_x_mean = pd.DataFrame()
    for col in df_x.columns.unique():
        df_x_mean[col] = df_x[col].mean(axis=1)

    if calc_ale:
        # Get Mean of ALEs
        df_ale_mean = pd.DataFrame()
        for idx, colx in enumerate(X_train):
            df_ale_mean[colx] = ale_out[colx].mean(axis=1)
            df_ale_mean[colx+'_eff'] = ale_out[colx+'_eff'].mean(axis=1)
            df_ale_mean[colx+'_size'] = ale_out[colx+'_size'].mean(axis=1)
    
    
    #########################
    # 10. Concatenate shap values from each fold!
    #########################
    if path is not None:
        test_set = list_test_sets[0]
        shap_values = np.array(list_shap_values[0])
        for k in range(1,len(list_test_sets)):
            test_set = np.concatenate((test_set,list_test_sets[k]),axis=0)
            shap_values = np.concatenate((shap_values,np.array(list_shap_values[k])),axis=0)
        
        #bringing back variable names - without noise feature
        X_shap = df.iloc[test_set][X_train.columns[:-1]]   
        # save as pickle (dirty for now but time is money)   
        path_shap = path + '_shap_vals.pkl'
        pickle.dump(shap_values, open(path_shap,"wb"))
        path_xshap = path + '_Xshap_vals.pkl'
        pickle.dump(X_shap, open(path_xshap,"wb"))           

    
    #########################
    ## 11. Print Metrics 
    #########################
    print('Best parameters found: learning rate: {}, max_depth {}, n_estimators: {}'
                                                    .format((round(df_cv.iloc[0][1],3)),
                                                    (round(df_out.iloc[0][2],3)),
                                                    (round(df_out.iloc[0][0],3))))
    print('--------------------')
    print('Metrics of model after cv')
    print('R2 Model: {}'.format(round(df_out.iloc[0][3],3)))
    print('--------------------')
    print('Metrics of prediction after cv')
    print('R2 Pred.: {}'.format(round(df_out.iloc[0][4],3)))
    print('MAE Pred.: {} m'.format(round(df_out.iloc[0][5],3)))
    print('RMSE Pred.: {} m'.format(round(df_out.iloc[0][6],3)))
    print('--------------------')
    print('Feature importance')
    print('--------------------')
    
    # print importance sorted 
    feat_imp = df_out[df_x_mean.columns]
    sorted_idx = np.argsort(feat_imp.iloc[0])[::-1]
    for feature, p in zip(feat_imp.columns[sorted_idx], sorted(feat_imp.iloc[0],reverse=True)):
        print("{}: {}".format(
            feature,
            round(p, 4)
            ))

    return (df_out, df_x_mean, df_ale_mean) if calc_ale else (df_out, df_x_mean)  

    
def xgb_shap_test_val_train(df,
                target="mean",
                kfold = 5,
                optimize_hype = False,
                calc_ale = False,
                calc_shap = False,
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    returns results based on prediction on hold out test set!

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)
    
    
    ##########################
    # 2. Get Hold out Test Set
    ##########################
    fold_pos = range(kfold)
    # choose random fold out of kfolds for hold out test set
    k = random.choice(fold_pos)
    # get hold out test set in pizza oppo form
    df_train_raw, df_test = pizza_oppo_split(df,num_folds=5, pos =k) 
    # print which gold is hold out test set 
    print('Hold out test set in fold {}'.format(k))
    
    
    # -------------- START k fold CROSSVALIDATION -----------------------#
    out=[]
    # loop over all slices apart from the one that was used for hold out test set
    for i in [x for x in range(kfold) if x != k]:
        #print('cross validation in sector ',i)

        ##################################################
        ## 3. Get train & test data
        ##################################################
        # get data based on oppo pizza split
        df_train, df_val = pizza_oppo_split(df,num_folds=5, pos =i)
        # remove rows that are in hold out test set
        train_duplicates = pd.merge(df_train, df_test, how='inner',left_index=True,right_index=True)
        val_duplicates = pd.merge(df_val, df_test, how='inner',left_index=True,right_index=True)
        df_train = df_train.drop(train_duplicates.index)
        df_val = df_val.drop(val_duplicates.index)
        

        # assign features to X
        X_train = df_train[[col for col in df_train.columns if "feature_" in col]]
        X_train["feature_noise"] = np.random.normal(size=len(df_train))
        X_val = df_val[[col for col in df_val.columns if "feature_" in col]]
        X_val["feature_noise"] = np.random.normal(size=len(df_val))
        
        # assign target to y
        y_train = df_train[target]
        y_val = df_val[target]

        # create results df
        df_val = df_val[['tripdistancemeters','hex_id','geometry']]
        df_val.rename(columns={'tripdistancemeters':'y_val'},inplace=True) 
        
        ##################################################
        ## 4. Tune Hyperparameters in Fold
        ##################################################
        if optimize_hype:
            print("Optimizing Hyperparameters..")
            #start = time()
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}             
            tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            #end = time()
            print("Best Parameters found: ", tuning.best_params_)

            n_parameter = tuning.best_params_["n_estimators"]
            lr_parameter = tuning.best_params_["learning_rate"] 
            md_parameter = tuning.best_params_["max_depth"] 
        else:
            n_parameter = 100
            lr_parameter = 0.05
            md_parameter = 2

        ##################################################
        ## 5. Training on kfold
        ##################################################
        tree = xgboost.XGBRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter,
            importance_type = 'total_gain')

        # Train the model
        model = tree.fit(X_train, y_train)
        y_val_predict = tree.predict(X_val)

        ##################################################
        ## 6. Results per Fold
        ##################################################
        # Append results to df_results
        df_val['y_predict_trees'] = y_val_predict
        df_val['error_trees'] = df_val['y_predict_trees'] - df_val['y_val']
        df_val['error_abs_trees'] = abs(df_val['error_trees'])

        # Metric of model in k fold
        r2_model = tree.score(X_train,y_train)
        # Metrics of prediction in k fold
        r2_pred = tree.score(X_val,y_val)
        mae_pred = metrics.mean_absolute_error(df_val['y_val'],df_val['y_predict_trees'])
        rmse_pred = np.sqrt(metrics.mean_squared_error(df_val['y_val'],df_val['y_predict_trees']))

        #Plot metrics each round!
        print('Metrics of prediction in sector',i)
        print('R2 Model: {}'.format(r2_model))
        print('R2 Predict: {}'.format(r2_pred))
        #print('MAE: {} m'.format(mae_pred))
        #print('RMSE: {} m'.format(rmse_pred))

        #########################
        ## 8. Append 
        #########################
        # Append stats in output df
        val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
        out.append(val_list)


    # -------------- End k fold CROSSVALIDATION -----------------------#

    #########################
    ## 9. Mean over k folds 
    #########################    
    # Get Mean of stats 
    col = ['n_parameter','lr_parameter','md_parameter','r2_model','r2_pred','mae_pred','rmse_pred']
    #col.extend(X_train.columns.values.tolist())
    # collect in one df
    df_cv = pd.DataFrame(data = out, columns = col)
    # calc average 
    df_out = pd.DataFrame(df_cv.mean()).transpose()

    print('Average parameters found: learning rate: {}, max_depth {}, n_estimators: {}'
                                                    .format((round(df_cv.iloc[0][1],3)),
                                                    (round(df_out.iloc[0][2],3)),
                                                    (round(df_out.iloc[0][0],3))))

    ##################################################
    ## 10 Take best Hyperparameters from kfolds and assign data
    ##################################################
    # assign features to X
    X_train_raw = df_train_raw[[col for col in df_train_raw.columns if "feature_" in col]]
    X_train_raw["feature_noise"] = np.random.normal(size=len(df_train_raw))
    X_test = df_test[[col for col in df_test.columns if "feature_" in col]]
    X_test["feature_noise"] = np.random.normal(size=len(df_test))
    
    # assign target to y
    y_train_raw = df_train_raw[target]
    y_test = df_test[target]

    # create results df
    df_test = df_test[['tripdistancemeters','hex_id','geometry']]
    df_test.rename(columns={'tripdistancemeters':'y_test'},inplace=True) 

    # assign mean values from folds 
    md_ave = int(round(df_out.iloc[0][2],0))
    n_ave = int(round(df_out.iloc[0][0],0))
    lr_ave = df_cv.iloc[0][1]
    

    print('max depth: ',md_ave)
    print('num estimators: ', n_ave)
    print('learning rate: ',lr_ave)


    ##################################################
    ## 8. Training and Prediction on Hold Out Test Set
    ##################################################
    print('------------------------')
    print('Final training of model with hyperparameter from CV')
    tree = xgboost.XGBRegressor(
        max_depth=md_ave, 
        n_estimators=n_ave, 
        learning_rate=lr_ave,
        importance_type = 'total_gain')

    # Train the model
    model = tree.fit(X_train_raw, y_train_raw)
    y_predict = tree.predict(X_test)

    ##################################################
    ## 8. Results
    ##################################################
    # Append results to df_results
    df_test['y_predict_trees'] = y_predict
    df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
    df_test['error_abs_trees'] = abs(df_test['error_trees'])

    # Metric of model
    r2_model = tree.score(X_train_raw,y_train_raw)
    # Metrics of prediction
    r2_pred = tree.score(X_test,y_test)
    mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
    rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))


    #########################
    ## 9. Feature Importance
    #########################
    # Calculate feature importances
    feature_importance = model.feature_importances_
    
    #########################
    ## 10. Partial Dependence
    #########################
    if calc_ale:
        ale_out = pd.DataFrame()
        for j,col in enumerate(X_train_raw):
            ale_eff = ale(X=X_train_raw, model = model, feature = [X_train_raw.columns[j]], grid_size=50, include_CI=False, plot=False).reset_index()
            ale_eff = ale_eff.rename(columns={'eff':col+'_eff','size':col+'_size'})
            ale_out = pd.concat([ale_out,ale_eff],axis=1)
    
    if calc_shap:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)           
        

    #########################
    ## 10. Print Metrics 
    #########################
        #Plot metrics each round!
    #print('Metrics of prediction in sector',i)
    #print('R2 Model: {}'.format(r2_model))
    #print('R2 Predict: {}'.format(r2_pred))
    #print('MAE: {} m'.format(mae_pred))
    #print('RMSE: {} m'.format(rmse_pred))    


    print('--------------------')
    print('Metrics of model after cv')
    print('R2 Model: {}'.format(r2_model))
    print('--------------------')
    print('Metrics of prediction after cv')
    print('R2 Predict: {}'.format(r2_pred))
    print('MAE: {} m'.format(mae_pred))
    print('RMSE: {} m'.format(rmse_pred)) 
    print('--------------------')
    print('Feature importance')
    print('--------------------')
    
    
    # print importance sorted 
    #feat_imp = df_out[X_train_raw.columns]
    feat_imp = feature_importance
    sorted_idx = np.argsort(feat_imp)[::-1]
    for feature, p in zip(X_train_raw.columns[sorted_idx], sorted(feat_imp,reverse=True)):
        print("{}: {}".format(
            feature,
            round(p, 4)
            ))
    
    
    return (df_out, X_train_raw, ale_out, shap_values) if calc_ale else (df_out, X_train_raw, feature_importance)              
    

def xgb_spatial_cv_shap_test_perc(df,
                    test_perc,
                    path,
                    target="mean",
                    kfold = 5,
                    optimize_hype = False,
                    calc_ale = False,
                    print_split=False,
                    cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates ALE Shap values on whole dataset!

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)

    # Do spatial cross validation
    out=[]
    df_x = pd.DataFrame()
    
    if calc_ale:
        ale_out = pd.DataFrame()
    
    for i in range(kfold):

        #########################
        ## 2. Get train & test data
        #########################
        # get train & test part via make_pizza func
        df_pizza = make_pizza(df,do_plot=print_split,share = 1/kfold,angle=i*2*np.pi/kfold)        
        df_train = df_pizza[df_pizza.pizza.str.match('train')]
        df_train = df_train.drop(columns='pizza')
        df_test = df_pizza[df_pizza.pizza.str.match('test')]
        df_test = df_test.drop(columns='pizza')

        # cut training dataset and only use test_perc % of it
        df_train = df_train.sample(frac=test_perc/20)
        # assign features to X
        X_train = df_train[[col for col in df_train.columns if "feature_" in col]]
        X_train["feature_noise"] = np.random.normal(size=len(df_train))
        X_test = df_test[[col for col in df_test.columns if "feature_" in col]]
        X_test["feature_noise"] = np.random.normal(size=len(df_test))
        
        # assign target to y
        y_train = df_train[target]
        y_test = df_test[target]

        # create results df
        df_test = df_test[['tripdistancemeters','hex_id','geometry']]
        df_test.rename(columns={'tripdistancemeters':'y_test'},inplace=True) 
        
        #########################
        ## 3. Tune Hyperparameters
        #########################
        if optimize_hype:
            #print("Optimizing Hyperparameters..")
            #start = time()
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}             
            tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            
            n_parameter = tuning.best_params_["n_estimators"]
            lr_parameter = tuning.best_params_["learning_rate"] 
            md_parameter = tuning.best_params_["max_depth"] 
        else:
            n_parameter = 9000
            lr_parameter = 0.001
            md_parameter = 3

        #########################
        ## 4. Training on kfold
        #########################
        tree = xgboost.XGBRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter,
            importance_type = 'total_gain')

        # Train the model
        model = tree.fit(X_train, y_train)
        y_predict = tree.predict(X_test)

        #########################
        ## 5. Results
        #########################
        # Append results to df_results
        df_test['y_predict_trees'] = y_predict
        df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
        df_test['error_abs_trees'] = abs(df_test['error_trees'])

        # Metric of model
        r2_model = tree.score(X_train,y_train)
        # Metrics of prediction
        r2_pred = tree.score(X_test,y_test)
        mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
        rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))


        #########################
        ## 8. Append 
        #########################
        # Append stats in output df
        val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
        out.append(val_list)

    # -------------- End k fold CROSSVALIDATION -----------------------#

    # report length od training and test dataset
    print('------------------------')
    print('{}. Results'.format(test_perc))
    print('{} percent of the data used for training'.format(test_perc*10))
    print('Lentgh of training sample is: ',len(df_train))
    print('Lentgh of test sample is: ',len(df_test))

    #########################
    ## 9. Mean over k folds 
    #########################    
    # Get Mean of stats 
    col = ['n_parameter','lr_parameter','md_parameter','r2_model','r2_pred','mae_pred','rmse_pred']
    #col.extend(X_train.columns.values.tolist())
    # collect in one df
    df_cv = pd.DataFrame(data = out, columns = col)
    # calc average 
    df_out = pd.DataFrame(df_cv.mean()).transpose()

    print('- - -')
    print('Average parameters found: learning rate: {}, max_depth {}, n_estimators: {}'
                                                    .format((round(df_cv.iloc[0][1],3)),
                                                    (round(df_out.iloc[0][2],3)),
                                                    (round(df_out.iloc[0][0],3))))
    print('- - -')
    print('R2 Model: {} '.format(round(df_cv.iloc[0][3],3)))
    print('R2 Prediction: {} '.format(round(df_cv.iloc[0][4],3)))
    print('MAE Prediction: {} '.format(round(df_cv.iloc[0][5],3)))
    print('RMSE Prediction: {} '.format(round(df_cv.iloc[0][6],3)))
    print('------------------------')

    return (df_out, ale_out) if calc_ale else (df_out)
    

def xgb_nat_imp(df,
                target="mean",
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates ALE Shap values on whole dataset!

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)
    
    
    ##################################################
    ## 10 Take best Hyperparameters from kfolds and assign data
    ##################################################
    # assign features to X
    X_train_ws = df[[col for col in df.columns if "feature_" in col]]
    X_train_ws["feature_noise"] = np.random.normal(size=len(df))
    
    # assign target to y
    y_train_ws = df[target]

    # assign mean values from folds 
    # Origin
    #md_ave = int(round(2.4,0))
    #n_ave = int(round(8800.0,0))
    #lr_ave = 0.001
    # Destination
    md_ave = int(round(3.2,0))
    n_ave = int(round(6800.0,0))
    lr_ave = 0.001

    ##################################################
    ## 11. Training whole sample
    ##################################################
    print('------------------------')
    print('Final training of model with hyperparameter from CV')
    tree = xgboost.XGBRegressor(
        max_depth=md_ave, 
        n_estimators=n_ave, 
        learning_rate=lr_ave,
        importance_type = 'total_gain')

    # Train the model on ws
    model = tree.fit(X_train_ws, y_train_ws)

    # Metric of model on ws
    r2_model = tree.score(X_train_ws,y_train_ws)
    
    #########################
    ## 9. Stats on whole sample
    #########################
    # Calculate feature importances
    feature_importance = model.feature_importances_
    
    print('--------------------')
    print('Metrics of model on ws')
    print('R2 Model: {}'.format(round(r2_model,3)))

   # print importance sorted 
    #feat_imp = df_out[X_train_raw.columns]
    feat_imp = feature_importance
    sorted_idx = np.argsort(feat_imp)[::-1]
    for feature, p in zip(X_train_ws.columns[sorted_idx], sorted(feat_imp,reverse=True)):
        print("{}: {}".format(
            feature,
            round(p, 3)
            ))


def feature_selection(df,
                target="mean",
                optimize_hype = False,
                cutoff=10):
    """
    Does Feature Selction based on Shap and Random Probing

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
        df_ale_mean: pd.DataFrame; contains mean ALE plot data (feat_value & effect_value) 
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)

    
    #########################
    ## 2. Get train & test data
    #########################
    # assign features to X
    X_train = df[[col for col in df.columns if "feature_" in col]]
    X_train["feature_noise"] = np.random.normal(size=len(df))
    
    # assign target to y
    y_train = df[target]
 
    
    #########################
    ## 3. Tune Hyperparameters
    #########################
    if optimize_hype:
        print("Optimizing Hyperparameters..")
        #start = time()
        LR = {"learning_rate": [0.001], 
                "n_estimators": [1000, 3000, 5000, 7000, 10000],
                "max_depth":[1, 2, 3, 5,7,10]}             
        tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
        tuning.fit(X_train, y_train)
        #end = time()
        print("Best Parameters found: ", tuning.best_params_)
        #print("After {} s".format(end - start))

        n_parameter = tuning.best_params_["n_estimators"]
        lr_parameter = tuning.best_params_["learning_rate"] 
        md_parameter = tuning.best_params_["max_depth"] 
    else:
        n_parameter = 100
        lr_parameter = 0.05
        md_parameter = 2

    #########################
    ## 4. Training on kfold
    #########################
    tree = xgboost.XGBRegressor(
        max_depth=md_parameter, 
        n_estimators=n_parameter, 
        learning_rate=lr_parameter,
        importance_type = 'total_gain')

    # Train the model
    model = tree.fit(X_train, y_train)
    #y_predict = tree.predict(X_test)

    #########################
    ## 5. Results
    #########################
    # Metric of model
    r2_model = tree.score(X_train,y_train)

    #Plot metrics each round!
    print('R2 Model: {}'.format(r2_model))
    print('-----------')

    #########################
    ## Feature Importance 
    #########################

    print('Feature Ranking based on SHAP:')
    # Calculate Fetaure Importance based on SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    vals = np.abs(shap_values.values).mean(0)
    feature_names = X_train.columns

    fi_shap = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    fi_shap.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    # get relative vals in %
    fi_shap['rel_vals'] = fi_shap.feature_importance_vals/sum(fi_shap.feature_importance_vals)
    
    for feat, imp in zip(fi_shap.col_name, fi_shap.rel_vals):
       print("{}: {}".format(
            feat,
            round(imp, 3)
            )) 
    print('-----------')
    # take only features wich have a higher impact than the noise feature
    noise_value = fi_shap.loc[fi_shap.col_name =='feature_noise'].rel_vals.values[0]
    feat_excluded = fi_shap.loc[fi_shap.rel_vals <= (noise_value+0.005)].col_name.values
    # remove noise feature from this list
    feat_excluded = np.delete(feat_excluded, np.where(feat_excluded=='feature_noise'))
    print('Based on this, we exclude {} of {} features in the analysis:'.format(len(feat_excluded),len(X_train.columns)-1))
    print(feat_excluded)

    return feat_excluded