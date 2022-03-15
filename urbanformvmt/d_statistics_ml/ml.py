import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from time import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import plot_partial_dependence
from sklearn import metrics
from e_utils.utils import make_pizza
from PyALE import ale
import xgboost
import shap
import pickle

import warnings
warnings.filterwarnings('ignore')
 


def boosted_trees(df,
                target="mean",
                split_angle = None, 
                normalize=False,
                optimize_hype = False,
                print_importance=False,
                print_partial_dependence=False,
                print_split=False,
                cutoff=10):
    """
    Conducts analysis of feature importance using Boosted Gradient Trees. All
    columns of DataFrame df with name containing 'feature_' are interpreted as
    feature. Chooses target variable using the 'target' input (which should
    label a column in df)
    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        normalizes: bool; normalizes feature importance
    Out:
        plots the resulting importance
    """
    #########################
    # 1. Preprocessing
    #########################
    # drop Nans
    df = df.dropna()
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    df = df.reset_index(drop=True)
    # normalize to 1x1 problem
    """
    if normalize:
        df.tripdistancemeters = df.tripdistancemeters/max(df.tripdistancemeters) 
        df.feature_distance_cbd = df.feature_distance_cbd/max(df.feature_distance_cbd) 
        df.feature_distance_local_cbd = df.feature_distance_local_cbd/max(df.feature_distance_local_cbd)
        df.feature_pop_density = df.feature_pop_density/max(df.feature_pop_density)
        df.feature_transit_density = df.feature_transit_density/max(df.feature_transit_density)
        df.feature_social_status_index = df.feature_social_status_index/max(df.feature_social_status_index)
    """
    # train, test split
    if split_angle is not None:
        df_pizza = make_pizza(df,do_plot=print_split,share = 0.2,angle=0)
        df_train = df_pizza[df_pizza.pizza.str.match('train')]
        df_train = df_train.drop(columns='pizza')
        df_test = df_pizza[df_pizza.pizza.str.match('test')]
        df_test = df_test.drop(columns='pizza')
    else:
        df_train = df.sample(frac=0.8,random_state=0)
        df_test = df.drop(df_train.index) 

    # assign features to X
    X_train = df_train[[col for col in df_train.columns if "feature_" in col]]
    X_train["feature_noise"] = np.random.normal(size=len(df_train))
    X_test = df_test[[col for col in df_test.columns if "feature_" in col]]
    X_test["feature_noise"] = np.random.normal(size=len(df_test))
    # Categorical variables
    #if 'feature_ua' in X_train.columns:
    #    # Apply one hot encoding for land use variables
    #    X_train = pd.get_dummies(X_train, columns=["feature_ua"], prefix=["feature_ua_"])
    #    X_test = pd.get_dummies(X_test, columns=["feature_ua"], prefix=["feature_ua_"])
    
    # assign target to y
    y_train = df_train[target]
    y_test = df_test[target]

    # create results df
    df_test = df_test[['tripdistancemeters','hex_id','geometry']]
    df_test.rename(columns={'tripdistancemeters':'y_test'},inplace=True) 

    # checky-check
    print('X_train: {}'.format(X_train.shape))
    print('y_train: {}'.format(y_train.shape))
    print('X_test: {}'.format(X_test.shape))
    print('y_test: {}'.format(y_test.shape))
    print('Train-Split: {} %'.format(round((len(df_train)/len(df)), 2)))

    #########################
    # 2. Hyperpara Optimization
    #########################
    
    if optimize_hype:
        print("Optimizing Hyperparameters..")
        start = time()
        LR = {"learning_rate": [0.001], 
                #"n_estimators": [10, 50, 100, 150, 500, 1000, 15000],
                "n_estimators": [1000, 3000, 5000, 7000, 10000],
                "max_depth":[1, 2, 3, 5,7,10]}
        tuning = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=LR, scoring="r2")
        tuning.fit(X_train, y_train)
        end = time()
        print("Best Parameters found: ", tuning.best_params_)
        print("After {} s".format(end - start))

        n_parameter = tuning.best_params_["n_estimators"]
        lr_parameter = tuning.best_params_["learning_rate"] 
        md_parameter = tuning.best_params_["max_depth"] 
    else:
        n_parameter = 15000
        lr_parameter = 0.001
        md_parameter = 2


    #########################
    # 3. Model Training
    #########################
    tree = GradientBoostingRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter
            )
    
    print('--------------------')
    print("Fitting Boosted Trees")

    # Train the model
    model = tree.fit(X_train, y_train)
    y_predict = tree.predict(X_test)

    # Append results to df_results
    df_test['y_predict_trees'] = y_predict
    df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
    df_test['error_abs_trees'] = abs(df_test['error_trees'])

    # Metric of model
    r2_model = tree.score(X_train,y_train)
    print('--------------------')
    print('Metrics of Model')
    print('--------------------')
    print("R2: ", r2_model)
    
    ## Prediction Metrics
    r2_pred = tree.score(X_test,y_test)
    mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
    rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))

    print('--------------------')
    print('Metrics of prediction')
    print('R2: {}'.format(r2_pred))
    print('MAE: {} m'.format(mae_pred))
    print('RMSE: {} m'.format(rmse_pred))

    # calculation of feature importance & standard deviations
    feature_importance = model.feature_importances_
    stds = np.std([tree[0].feature_importances_ for tree in model.estimators_], axis=0)

    # Nomarlisation
    if normalize:
        feature_importance = 100. * (feature_importance / feature_importance.max())

    # assign to df_out (before values get sorted)
    val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
    val_list.extend(feature_importance.tolist())
    val_list.extend(stds.tolist())
    col = ['n_parameter','lr_parameter','md_parameter','r2_model','r2_pred','mae_pred','rmse_pred']
    col.extend(X_train.columns.values.tolist())
    col.extend(['stds_'+ col for col in X_train.columns])
    df_out = pd.DataFrame([val_list],columns=tuple(col))

    # Feature importance
    print('--------------------')
    print('Feature importance (sorted in %): ')
    print('--------------------')
    sorted_idx = np.argsort(feature_importance)[::-1]

    for feature, p in zip(X_train.columns[sorted_idx], sorted(feature_importance,reverse=True)):
        print("{}: {}".format(
            feature, 
            round(p, 4)
            ))

    #########################
    # 4. Create Plots
    #########################
    # plot partial dependence
    if print_partial_dependence:
        for i in range(len(X_train.columns)):
            plot_partial_dependence(model, X_train,[i])
    
    # plopt bars of importance
    if print_importance:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        stds = np.std([tree[0].feature_importances_ for tree in model.estimators_], axis=0)
        importances = pd.Series(feature_importance, index=X_train.columns)
        importances.plot.bar(yerr=stds, ax=ax)

        """
        ax.barh(pos, feature_importance[sorted_idx], align="center")        
        ax.set_yticks(range(len(pos)))
        ax.set_yticklabels(X_train.columns[sorted_idx])
        ax.set_xlabel("Feature importance")
        """
        
        plt.show()

    return df_out
    
    
def get_pearson(df, target="mean", ignore=[], cutoff=10):
    """
    Analyses linear correlation between all columns containing keyword "feature_"
    of df and target column. Sorts and prints the result
    In:
        df: pd.DataFrame
        target: str; column of df with target variable
        ignore: list of str; names of columns for which we are not interested in linear correlations
    """
    print('len df:',len(df))
    df = df.dropna()
    print('len df:',len(df))
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    print('len df:',len(df))

    X = df[[col for col in df.columns if "feature_" in col]]
    X = X[[col for col in X.columns if col not in ignore]]
    X["feature_noise"] = np.random.normal(size=len(X))
    
    y = df[target]
    
    feature_names = []
    r_vals = []
    p_vals = []
    
    # get correlations
    start = time()
    for feature in X:
        print("Analysing feature {}...".format(feature))
        predictor = np.array(X[feature].tolist())
        print('predictor:',len(predictor))
        print('-----')
        print('target:',len(y))
        r, p = pearsonr(predictor, y)
        
        feature_names.append(feature)
        r_vals.append(r)
        p_vals.append(p)
        
    print("Determined all correlations in {} s \n".format(time() - start))
    # sort
    zipped = sorted(zip(r_vals, feature_names, p_vals))
    feature_names = [name for _, name, _ in zipped]
    p_vals = [p for _, _, p in zipped]
    r_vals = [r for r, _, _ in zipped]

    # added to df_out (to results values in df)
    df_out = pd.DataFrame(columns = feature_names, data = [r_vals])
    df1 = pd.DataFrame(columns = feature_names, data = [p_vals])
    df_out = df_out.append(df1)
    df_out['Stat_val'] = ['pears_r_val','pears_p_val']    
        
    print("Analysis of Linear Correlation: \n")
    for feature, r, p in zip(feature_names, r_vals, p_vals):
        print("Predictor {}: R: {}, p: {}".format(
             feature, 
             round(r,2), 
             round(p, 2)
             ))
    return df_out
        
        
def inter_feature(df, cutoff=10, do_plot=True):
    """
    determines pearson correlations between all columns labelled as "feature_<feature_name>"
    unless the feature is in ignore. Only considers lines with number of trips > cutoff
    In:
        df: pd.DataFrame; df of interest
        cutoff: int; minimum number of trips from area to be considered 
    Out:
        np.array: matrix of inter-feature-correlations
    """
    df = df.dropna()
    
    # shave of statistically insignificant rows 
    df = df.loc[df["points_in_hex"] > cutoff]
    # shave of columns that are not features (non-numerical columns are ignored by pandas automatically)
    df = df[[col for col in df.columns if "feature_" in col]]
    # add noise column for gauge
    df["feature_noise"] = np.random.normal(size=len(df))
    
    # get pearson correlations
    corr = df.corr()
    
    if do_plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
                    cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True),
                    square=True, ax=ax)
        plt.show()
    
    return corr



def sweep_feature_importance(df, feature, steps=10, do_plot=False, target="tripdistancemeters", cutoff=10):
    """
    sweeps a selected feature in df and determines the respective feature 
    importance in each sweep step. Method uses gradient boosted trees. Also re-
    turns for all sweep step the relative number of examples per step
    In:
        df: pd.DataFrame; Contains all required data
        feauture: str; feature to be sweeped
        steps: int; number of steps
        do_plot: bool; plots results if True
        target: str; target the method is attempting to predict
        cutoff: int; minimum number of trips per considered instance
    Out:
        density: list of float; share of total number of examples in that sweep step
        pearsonr_vals: list of float; feature importance at all sweep steps determined by pearsonr
        pearsonr_vals: list of float; p-values of that analysis
        tree_r: list of float; importance as determined by gradient boost method
    """
    df = df.dropna()
    df = df.loc[df["points_in_hex"] > cutoff]
    
    if not "feature_" in feature:
        feature = "feature_" + feature
    
    # determine sweep steps
    col = df[feature]
    vals = np.linspace(col.min(), col.max(), steps+1)
    
    # instantiate quantities of interest
    density = []
    pearsonr_vals = []
    pearsonp_vals = []
    tree_r = []
    num_examples = len(df)
    
    for i in range(steps):
        lower, upper = vals[i], vals[i+1]
        
        cond = df[feature] > lower
        cond *= df[feature] < upper
        
        curr_df = df.loc[cond]
        
        X = curr_df[[col for col in curr_df.columns if "feature_" in col]]
        y = curr_df[target]
        
        density.append(len(X) / num_examples)
        
        # pearsonr analysis
        predictor = np.array(X[feature].tolist())
        r, p = pearsonr(predictor, y)
        pearsonr_vals.append(r)
        pearsonp_vals.append(p)
        
        # boosted trees analysis
        tree = GradientBoostingRegressor(
            max_depth=2, 
            n_estimators=250, 
            learning_rate=0.05
            )
        model = tree.fit(X, y)
        
        # get resulting feature importance
        feature_importances = model.feature_importances_
        idx = X.columns.get_loc(feature)
        tree_r.append(feature_importances[idx])
        
    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        x = [np.mean([vals[i], vals[i+1]]) for i in range(steps)]
        ax.plot(x, density, label="density")
        ax.plot(x, pearsonr_vals, label="pearson r")
        ax.plot(x, pearsonp_vals, label="pearson p")
        ax.plot(x, tree_r, label="tree r")
        
        ax.grid(True, linestyle="dashed")
        ax.legend()
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Importance of {}".format(feature))
        ax.set_xlabel(feature.replace("feature_", ""))
        plt.show()
        
    return density, pearsonr_vals, pearsonp_vals, tree_r


def boosted_trees_spatial_cv(df,
                target="mean",
                kfold = 5,
                optimize_hype = False,
                calc_ale = False,
                print_split=False,
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates average model and prediction errors as well as average feature importance over
    all kfolds.

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
            start = time()
            #LR = {"learning_rate": [0.1,0.01,0.001], 
            #        "n_estimators": [150, 300, 1000, 15000],
            #        "max_depth":[2,3,5]}
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}            
            tuning = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            end = time()
            print("Best Parameters found: ", tuning.best_params_)
            print("After {} s".format(end - start))

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
        tree = GradientBoostingRegressor(
            max_depth=md_parameter, 
            n_estimators=n_parameter, 
            learning_rate=lr_parameter)

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
        ## 6. Feature Importance
        #########################
        # Calculate feature importances
        feature_importance = model.feature_importances_
        stds = np.std([tree[0].feature_importances_ for tree in model.estimators_], axis=0)

        #########################
        ## 7. Partial Dependence
        #########################
        if calc_ale:
            if 'feature_social_status_index' in X_train.columns:
                X_train['feature_social_status_index'] = X_train['feature_social_status_index'].astype(int)
                X_train['feature_social_dynamic_index'] = X_train['feature_social_dynamic_index'].astype(int)
            #for each feature calculate ALE
            for i,col in enumerate(X_train):
                ale_eff = ale(X=X_train, model = model, feature = [X_train.columns[i]], grid_size=50, include_CI=False, plot=False).reset_index()
                ale_eff = ale_eff.rename(columns={'eff':col+'_eff','size':col+'_size'})
                ale_out = pd.concat([ale_out,ale_eff],axis=1)
            
        #########################
        ## 8. Append 
        #########################
        # Append stats in output df
        val_list = [n_parameter,lr_parameter,md_parameter,r2_model,r2_pred,mae_pred,rmse_pred]
        val_list.extend(feature_importance.tolist())
        val_list.extend(stds.tolist())
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
    col.extend(['stds_'+ col for col in X_train.columns])
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
    


def baseline_spatial_cv(df,
                target="mean",
                kfold = 5,
                print_split=False,
                cutoff=10):
    """
    Does spatial cross validation for any given number of kfolds.
    Calculates average model and prediction errors as well as average feature importance over
    all kfolds.

    In:
        df: pd.DataFrame; dataframe with feautures and target column
        target: str; chosen target variable
        kfold: num; number of folds or pizza slices
        optimize_hype: bool; if True, hypeparameter optimisation for each kfold will be conducted
        print_split: bool; if True, kfold splits will be printed
    Out:
        df_out: pd.DataFrame; contains average model, prediction errors and feature importances 
    """
    print('--------------------')
    print('Calculating baseline Model')
    print('--------------------')
    
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

        #########################
        ## 4. Training on kfold
        #########################
        y_train_median = np.median(y_train)
        y_predict=np.empty(len(y_test)) 
        y_predict.fill(y_train_median) 

        #########################
        ## 5. Results
        #########################
        # Append results to df_results
        df_test['y_predict_trees'] = y_predict
        df_test['error_trees'] = df_test ['y_predict_trees'] - df_test ['y_test']
        df_test['error_abs_trees'] = abs(df_test['error_trees'])

        # Metric of model
        #r2_model = tree.score(X_train,y_train)
        # Metrics of prediction
        #r2_pred = tree.score(X_test,y_test)
        mae_pred = metrics.mean_absolute_error(df_test['y_test'],df_test['y_predict_trees'])
        rmse_pred = np.sqrt(metrics.mean_squared_error(df_test['y_test'],df_test['y_predict_trees']))
        y_mean = np.mean(df[target])

        #########################
        ## 8. Append 
        #########################
        # Append in output df
        val_list = [y_mean,mae_pred,rmse_pred]
        out.append(val_list)
        
    
    # Write in to df
    col = ['y_mean','mae_pred','rmse_pred']
    df_cv = pd.DataFrame(out,columns=tuple(col))
    
    # Get Mean
    df_out = pd.DataFrame(df_cv.mean()).transpose()
    # Print Metrics
    print('--------------------')
    print('Metrics of prediction after cv')
    print('Mean: {}'.format(round(df_out.y_mean,3)))
    print('MAE Pred.: {} m'.format(round(df_out.mae_pred,3)))
    print('RMSE Pred.: {} m'.format(round(df_out.rmse_pred,3)))

    return df_out


def xgb_spatial_cv(df,
                target="mean",
                kfold = 5,
                optimize_hype = False,
                calc_ale = False,
                print_split=False,
                cutoff=10):
    """
    Does spatial cross validation with hyperparameter tuning for any given number of kfolds.
    Calculates average model and prediction errors as well as average feature importance over
    all kfolds.

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
    if 'feature_social_status_index' in df.columns:
        df['feature_social_status_index'] = df['feature_social_status_index'].astype(int)
        df['feature_social_dynamic_index'] = df['feature_social_dynamic_index'].astype(int)

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
            start = time()
            LR = {"learning_rate": [0.001], 
                    "n_estimators": [1000, 3000, 5000, 7000, 10000],
                    "max_depth":[1, 2, 3, 5,7,10]}             
            tuning = GridSearchCV(estimator=xgboost.XGBRegressor(), param_grid=LR, scoring="r2")
            tuning.fit(X_train, y_train)
            end = time()
            print("Best Parameters found: ", tuning.best_params_)
            print("After {} s".format(end - start))

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

    
