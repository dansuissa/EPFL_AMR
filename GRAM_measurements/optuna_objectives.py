"""
Objective functions for Optuna optimization.

This module contains the classs containing objective functions for optimizing the hyperparameters
of the XGBoost and random forest classifiers for Gram type, shape type and bacteria type classification.

Functions:
    objective_xgb_gram(self, trial)
    objective_xgb_shape(self, trial)
    objective_xgb_bacteria(self, trial)
    objective_rf_gram(self, trial)
    objective_rf_shape(self, trial)
    objective_rf_bacteria(self, trial)
    get_median_best_params(self, study) #optional to use if we want to choose the median best parameters from the search
"""

"""
   quick explenation of how the library optuna works :

   Optuna is an automatic hyperparameter optimization software framework designed for machine learning.
   It allows users to find the best hyperparameters for their models by defining an objective function
   that should be minimized or maximized. Optuna is used because it provides a simple yet powerful interface
   for hyperparameter tuning, supports a wide range of optimization algorithms, and is highly efficient in
   finding optimal hyperparameters compared to grid search or random search.

   How Optuna works:
   1. **Define the Objective Function**: The objective function is the function that you want to optimize.
      In this case, it is the function that trains and evaluates the XGBoost classifier.
   2. **Suggest Hyperparameters**: Within the objective function, `trial.suggest_*` methods are used to
      suggest hyperparameters. These suggestions are based on prior trials and the optimization algorithm
      (e.g., Tree-structured Parzen Estimator).
   3. **Evaluate the Objective Function**: The suggested hyperparameters are used to train the model, and
      the performance of the model is evaluated on a validation set.
   4. **Optimize**: Optuna optimizes the hyperparameters by minimizing or maximizing the return value of
      the objective function over multiple trials. It uses sophisticated algorithms to explore the
      hyperparameter space efficiently.

   Args:
       trial (optuna.trial.Trial): Optuna trial object for suggesting hyperparameters.
       X_train (ndarray): Training features.
       y_train (ndarray): Training labels.
       X_test (ndarray): Testing features.
       y_test (ndarray): Testing labels.

   Returns:
       float: Accuracy of the model on the test set.
   """
import optuna
from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
#objective function for optuna optimization for both gram and bacteria classification
class ObjectiveXGBModel:
    def __init__(self, X_train, X_valid, X_test, y_train, y_valid, y_test,
                 X_train_bacteria, X_valid_bacteria, X_test_bacteria,
                 y_train_bacteria, y_valid_bacteria, y_test_bacteria,
                 X_train_shape, X_valid_shape, X_test_shape, y_train_shape, y_valid_shape, y_test_shape):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.X_train_bacteria = X_train_bacteria
        self.X_valid_bacteria = X_valid_bacteria
        self.X_test_bacteria = X_test_bacteria
        self.y_train_bacteria = y_train_bacteria
        self.y_valid_bacteria = y_valid_bacteria
        self.y_test_bacteria = y_test_bacteria
        self.X_train_shape = X_train_shape
        self.X_valid_shape = X_valid_shape
        self.X_test_shape = X_test_shape
        self.y_train_shape = y_train_shape
        self.y_valid_shape = y_valid_shape
        self.y_test_shape = y_test_shape

    def objective_xgb_gram(self, trial):
        param = {
            #nb of boosting round (trees) in the forest
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),

            #max depth of each tree
            'max_depth': trial.suggest_int('max_depth', 3, 5),

            #step size shrinkage used to prevent overfiting: controls the pace at which the algorithm learns or update the values of a param estimate
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),

            #subsample ratio of the training intances
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),

            #ratio of features (columns) to be randomly sampled for each tree
            #this helps in reducing overfitting by introducing randomness in the feature selection
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),

            #minimum loss reduction required to make a further partition on a leaf node of the tree
            'gamma': trial.suggest_float('gamma', 0.1, 10.0, log=True),

            #L1 regularization on weights
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),

            #L2
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),

            #min sum of instance weights (hessian) needed in a child
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        }

        #initiates the xgb classifier with method trial.suggest that sugests parameters
        xgb_gram = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
        #split the data into 5 folds while maintaining the class distribution
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #evaluates the model using the specified cross-validation strategy and returns the accuracy scores for each fold
        scores = cross_val_score(xgb_gram, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
        #lastly optuna records the mean performance score across all folds and uses it to update its understanding of the search space to suggest better hyperparameters in the next trial
        return accuracy
    """
        xgb_gram = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
        xgb_gram.fit(self.X_train, self.y_train, eval_set=[(self.X_valid, self.y_valid)], early_stopping_rounds=10,
                     verbose=False)
        accuracy = xgb_gram.score(self.X_valid, self.y_valid)
        return accuracy
        """

    def objective_xgb_bacteria(self, trial):
        param = {
            #'objective': 'multi:softmax',
            'objective': 'multi:softmax',
            'num_class': len(np.unique(self.y_train_bacteria)),
            'n_estimators': trial.suggest_int('n_estimators', 165, 171),
            'max_depth': trial.suggest_int('max_depth', 5, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.73, 0.74),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.790, 0.795), #old 0.5,0.8
            'gamma': trial.suggest_float('gamma', 0.025, 0.029, log=True), #old 0.1,10
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.02, log=True), #old 0.1,10
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.002, log=True),#old 0.1,10
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),#old 1,6
            'max_delta_step': trial.suggest_float('max_delta_step', 5.5, 6.1),  #meilleur c comme ca
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        }

        xgb_bacteria = XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(xgb_bacteria, self.X_train_bacteria, self.y_train_bacteria, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
        return accuracy







    """
        xgb_bacteria = XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
        xgb_bacteria.fit(self.X_train_bacteria, self.y_train_bacteria,
                         eval_set=[(self.X_valid_bacteria, self.y_valid_bacteria)], verbose=False)
        accuracy = xgb_bacteria.score(self.X_valid_bacteria, self.y_valid_bacteria)
        return accuracy
         """

    def objective_xgb_shape(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'gamma': trial.suggest_float('gamma', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        }
        xgb_shape = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
        xgb_shape.fit(self.X_train_shape, self.y_train_shape, eval_set=[(self.X_valid_shape, self.y_valid_shape)],
                      early_stopping_rounds=10, verbose=False)
        accuracy = xgb_shape.score(self.X_valid_shape, self.y_valid_shape)
        return accuracy

    """
        xgb_shape = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(xgb_shape, self.X_train_shape, self.y_train_shape, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
        return accuracy
        """

    def get_median_best_params(self, study):
        trials_df = study.trials_dataframe()
        median_params = {}
        for param in trials_df.columns:
            if param.startswith('params_'):
                param_name = param.split('params_')[1]
                median_params[param_name] = np.median(trials_df[param])
                if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                    median_params[param_name] = int(median_params[param_name])
        return median_params

class ObjectiveRFModel:
    def __init__(self, X_train, X_test, y_train, y_test, X_train_bacteria, X_test_bacteria, y_train_bacteria, y_test_bacteria, X_train_shape, X_test_shape, y_train_shape, y_test_shape):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_bacteria = X_train_bacteria
        self.X_test_bacteria = X_test_bacteria
        self.y_train_bacteria = y_train_bacteria
        self.y_test_bacteria = y_test_bacteria
        self.X_train_shape = X_train_shape
        self.X_test_shape = X_test_shape
        self.y_train_shape = y_train_shape
        self.y_test_shape = y_test_shape

    def objective_rf_gram(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        rf_gram = RandomForestClassifier(**param, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(rf_gram, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
        print(f'Cross-validation accuracy for trial {trial.number}: {accuracy:.4f}')
        return accuracy

    def objective_rf_shape(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        rf_shape = RandomForestClassifier(**param, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(rf_shape, self.X_train_shape, self.y_train_shape, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
        print(f'Cross-validation accuracy for trial {trial.number}: {accuracy:.4f}')
        return accuracy

    def objective_rf_bacteria(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        rf_bacteria = RandomForestClassifier(**param, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(rf_bacteria, self.X_train_bacteria, self.y_train_bacteria, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
        print(f'Cross-validation accuracy for trial {trial.number}: {accuracy:.4f}')
        return accuracy
    def get_median_best_params(self, study):
        trials_df = study.trials_dataframe()
        median_params = {}
        int_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']
        for param in study.best_trial.params.keys():
            param_name = f'params_{param}'  #adjust parameter name to match the dataframe columns
            if param_name not in trials_df.columns:
                raise KeyError(f"Parameter {param_name} not found in trials dataframe columns")

            if param == 'max_features':
                #handle categorical parameter separately
                mode_value = trials_df[param_name].mode()[0]  #get the mode (most frequent value)
                median_params[param] = mode_value
            else:
                #convert to numeric and drop non-numeric values
                param_values = pd.to_numeric(trials_df[param_name], errors='coerce').dropna()
                median_value = np.median(param_values)

                if param in int_params:
                    median_params[param] = int(median_value)
                else:
                    median_params[param] = median_value
        return median_params

