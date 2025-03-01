#Standard Libraries
import os
import time
import numbers
import random
import copy
import json
import csv
import argparse
import pickle as pkl
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Data and Model handling libraries
import numpy as np
import pandas as pd
import joblib
import wandb

#Sckit-learn libraries
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score, roc_auc_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, Birch
from sklearn.kernel_approximation import Nystroem
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.experimental import enable_halving_search_cv  # noqa

#Additional ML Libraries
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb

#Bayesian Optimization
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Categorical, Integer

# PyOD (Python Outlier Detection) Models
from pyod.models.ecod import ECOD
from pyod.models.xgbod import XGBOD
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.vae import VAE
from pyod.models.auto_encoder import AutoEncoder as AE_flow
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.models.hbos import HBOS  # Histogram Based Outlier Score
from pyod.models.anogan import AnoGAN
from pyod.models.cblof import CBLOF
from pyod.models.kde import KDE

# DeepOD (Deep Outlier Detection) Models
from deepod.models.tabular import DevNet, DeepSAD, SLAD, RCA, ICL, DeepSVDD, PReNet

# Custom Modules
from PCAModel import Pca
from GetMetrics import precision_recall_f1_pertype, ap_at_k, ar_at_k, get_ap_at_perType


# FILE_PATH = 'PATH_TO_Data_Folder'
# Custom action to parse a string into a dictionary, with automatic type conversion for numerical values and underscore for lists.
class DictParseAction(argparse.Action):
    """Custom action to parse a string into a dictionary, with automatic type conversion for numerical values and underscore for lists."""

    def __call__(self, parser, namespace, values, option_string=None):
        args_dict = {}
        if values:
            for arg in values.split(","):
                key, value = arg.split(":")
                key = key.strip()
                value = self.convert_value(value.strip())
                args_dict[key] = value
        setattr(namespace, self.dest, args_dict)

    def convert_value(self, value):
        """Converts string values to int, float, or list if possible, otherwise returns the original string."""
        if "_" in value:  # Handle list values
            return [self.convert_single_value(v.strip()) for v in value.split("_")]
        else:
            return self.convert_single_value(value)

    def convert_single_value(self, value):
        """Converts a single string value to int or float if possible, otherwise returns the original string."""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                if value == "True":
                    return True
                elif value == "False":
                    return False
                return value  # Return as string if it's not a number


class PyODModel:
    def __init__(self, model, **params):
        self.threshold = None
        if model == "ECOD":
            self.model = ECOD(**params)
        elif model == "XGBOD":
            estimator_list = []
            if "estimator" in params:
                if params["estimator"] == "Mahalanobis":
                    estimator_list.append(
                        Pca(method="mahalanobis", n_selected_components=-1)
                    )
                elif params["estimator"] == "AE_flow":
                    estimator_list.append(AE_flow(**params))
            else:
                estimator_list = [
                    Pca(method="mahalanobis", n_selected_components=-1),
                    AE_flow(**params),
                ]
            print("estimator_list :", estimator_list)
            self.model = XGBOD(
                estimator_list=estimator_list, verbose=0, random_state=42, **params
            )
        elif model == "OCSVM":
            self.model = OCSVM(**params)
        elif model == "IForest":
            self.model = IForest(**params)
        elif model == "VAE":
            if ("encoder_neurons" in params) and isinstance(
                params["encoder_neurons"], int
            ):
                params["encoder_neurons"] = [params["encoder_neurons"]]
            if ("decoder_neurons" in params) and isinstance(
                params["decoder_neurons"], int
            ):
                params["decoder_neurons"] = [params["decoder_neurons"]]
            self.model = VAE(**params)
        elif model == "AE":
            if ("hidden_neurons" in params) and isinstance(
                params["hidden_neurons"], int
            ):
                params["hidden_neurons"] = [params["hidden_neurons"]]
            self.model = AutoEncoder(**params)
        elif model == "HBOS":
            self.model = HBOS(**params)
        elif model == "dif":
            self.model = DIF(**params)
        elif model == "AnoGAN":
            self.model = AnoGAN(**params)
        elif model == "CBLOF":
            if "clustering_estimator" in params:
                if params["clustering_estimator"] == "DBSCAN":
                    params["clustering_estimator"] = DBSCAN()
                elif params["clustering_estimator"] == "Birch":
                    params["clustering_estimator"] = Birch()
            self.model = CBLOF(**params)
        elif model == "KDE":
            self.model = KDE(**params)
        elif model == "DevNet":
            self.model = DevNet(**params)
        elif model == "DeepSAD":
            self.model = DeepSAD(**params)
        elif model == "DeepSVDD":
            self.model = DeepSVDD(**params)
        elif model == "AnoGAN":
            self.model = AnoGAN(**params)
        elif model == "DIF":
            self.model = DIF(**params)
        elif model == "SLAD":
            self.model = SLAD(**params)
        elif model == "RCA":
            self.model = RCA(**params)
        elif model == "ICL":
            # For ICL hidden_dims should be a list so if it is int convert it to list
            if isinstance(params["hidden_dims"], int):
                params["hidden_dims"] = [params["hidden_dims"]]
            self.model = ICL(**params)
        elif model == "PReNet":
            self.model = PReNet(**params)

    def _predict_window_scorer(self, X):
        window_scores = self.model.decision_function(X)
        return window_scores

    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self._predict_window_scorer(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        scores = self._predict_window_scorer(X)
        preds = (scores >= self.threshold).astype(int)
        return preds
    
    def get_params(self, **params):
        return self.model.get_params(**params)

    def set_params(self, **params):
        self.model.set_params(**params)
        
    def finetune(self, X, y, X_val=None, y_val=None, **params):
        # if there are anomaly points in the given data
        if sum(y) > 0:
            scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
            self.model.set_params(warm_start=True)
            self.model.n_estimators += 100
            self.model.fit(X, y)
        return self
    
class XGBModel:
    def __init__(self, best_thr=False, **params):
        self.best_thr = best_thr
        print("best_thr", best_thr, type(best_thr))
        self.threshold = None
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.best_thr:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first.")
            scores = self.decision_function(X)
            preds = (scores >= self.threshold).astype(int)
            return preds
        else:
            return self.model.predict(X)

    def decision_function(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, **params):
        return self.model.get_params(**params)

    def set_params(self, **params):
        self.model.set_params(**params)

    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores

    def finetune(self, X, y, X_val=None, y_val=None, **params):
        scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
        classifier = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
        )
        classifier.fit(
            X_train_bank, y_train_bank_mask, xgb_model=params["general_model"]
        )
        return classifier


class RFModel:
    def __init__(self, best_thr=True, **params):
        self.best_thr = best_thr
        print("best_thr", best_thr, type(best_thr))
        self.threshold = None
        self.model = RandomForestClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.best_thr:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first.")
            scores = self.decision_function(X)
            preds = (scores >= self.threshold).astype(int)
            return preds
        else:
            return self.model.predict(X)

    def decision_function(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, **params):
        return self.model.get_params(**params)

    def set_params(self, **params):
        self.model.set_params(**params)

    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores

    def finetune(self, X, y, X_val=None, y_val=None, **params):
        # if there are anomaly points in the given data
        if sum(y) > 0:
            scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
            self.model.set_params(warm_start=True)
            self.model.n_estimators += 100
            self.model.fit(X, y)

        return self


class RFMahal:
    def __init__(
        self,
        MahalPath="PATH_To_Mahalanobis_model",
        best_thr=True,
        **params,
    ):
        self.model = RandomForestClassifier(**params)
        with open(MahalPath, "rb") as f:
            self.Mahal = pkl.load(f)
        print("Loaded Mahalanobis model", MahalPath)
        self.threshold = None
        self.best_thr = best_thr

    def fit(self, X, y):
        X_latent = self.Mahal.transform_windows(X)
        self.model.fit(X_latent, y)

    def predict(self, X):
        if self.best_thr:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first.")
            scores = self.decision_function(X)
            preds = (scores >= self.threshold).astype(int)
            return preds
        else:
            return self.model.predict(X)

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        X_latent = self.Mahal.transform_windows(X)
        return self.model.predict_proba(X_latent)

    def get_params(self, **params):
        return self.model.get_params(**params)

    def set_params(self, **params):
        self.model.set_params(**params)

    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores

    def finetune(self, X, y, X_val=None, y_val=None, **params):
        scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced",
        )
        classifier.fit(X_train_bank, y_train_bank_mask)
        return classifier


class RFODMahal:
    def __init__(
        self,
        MahalPath="Path_to_Mahalanobis_Model",
        best_thr=True,
        **params,
    ):
        self.best_thr = best_thr
        print("best_thr", best_thr, type(best_thr))
        self.threshold = None
        with open(MahalPath, "rb") as f:
            self.Mahal = pkl.load(f)
        self.model = RandomForestClassifier(**params)

    def fit(self, X, y):
        Mahal_pred = self.Mahal.decision_function(X)
        X_new = np.concatenate((X, Mahal_pred[:, None]), axis=1)
        print("X_new shape:", X_new.shape)
        self.model.fit(X_new, y)

    def predict(self, X):
        if self.best_thr:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first.")
            scores = self.decision_function(X)
            preds = (scores >= self.threshold).astype(int)
            return preds
        else:
            return self.model.predict(X)

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        Mahal_pred = self.Mahal.decision_function(X)
        X_new = np.concatenate((X, Mahal_pred[:, None]), axis=1)
        return self.model.predict_proba(X_new)

    def get_params(self, **params):
        return self.model.get_params(**params)

    def set_params(self, **params):
        self.model.set_params(**params)

    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores

    def finetune(self, X, y, X_val=None, y_val=None, **params):
        scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
        classifier = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
        )
        classifier.fit(X, y, xgb_model=params["general_model"])
        return classifier


class XGBODMahal:
    def __init__(
        self,
        MahalPath="Path_to_Mahalanobis_Model",
        best_thr=True,
        **params,
    ):
        self.best_thr = best_thr
        print("best_thr", best_thr, type(best_thr))
        self.threshold = None
        with open(MahalPath, "rb") as f:
            self.Mahal = pkl.load(f)
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        Mahal_pred = self.Mahal.decision_function(X)
        X_new = np.concatenate((X, Mahal_pred[:, None]), axis=1)
        print("X_new shape:", X_new.shape)
        self.model.fit(X_new, y)

    def predict(self, X):
        if self.best_thr:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first.")
            scores = self.decision_function(X)
            preds = (scores >= self.threshold).astype(int)
            return preds
        else:
            return self.model.predict(X)

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        Mahal_pred = self.Mahal.decision_function(X)
        X_new = np.concatenate((X, Mahal_pred[:, None]), axis=1)
        return self.model.predict_proba(X_new)

    def get_params(self, **params):
        return self.model.get_params(**params)

    def set_params(self, **params):
        self.model.set_params(**params)

    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores

    def finetune(self, X, y, X_val=None, y_val=None, **params):
        scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
        classifier = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
        )
        classifier.fit(X, y, xgb_model=params["general_model"])
        return classifier


class SkModelWrapper:
    def __init__(self, model, **params):
        self.model = model
        self.threshold = None

    def decision_function(self, X):
        if hasattr(self.model, "decision_function"):
            window_scores = self.model.decision_function(X)
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.shape[1] == 2:
                window_scores = probs[:, 1]
            else:
                window_scores = np.zeros(X.shape[0])
        else:
            raise AttributeError(
                "Model does not have decision_function or predict_proba method."
            )
        return window_scores

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not have predict_proba method.")

    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        scores = self.decision_function(X)
        preds = (scores >= self.threshold).astype(int)
        # preds = self.model.predict(X)
        return preds

class IForestModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params  # Store initial parameters
        self.model = IForest(**params)
        self.threshold = None  # Initialize threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        """Fit the IForest model to the training data."""
        self.model.fit(X, y)
        return self  # Return self to comply with sklearn fit requirements
    
    def predict(self, X):
        """Predict if each sample in X is an outlier or not based on the threshold."""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call `set_threshold` first.")
        scores = self.model.decision_function(X)
        preds = (scores >= self.threshold).astype(int)
        return preds
    
    def decision_function(self, X):
        """Return the outlier score for each sample in X."""
        return self.model.decision_function(X)
    
    def set_threshold(self, X_val, y_val):
        """Calculate and set the threshold based on validation data."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1_scores = 2 * (precision * recall) / (precision + recall)
        F1_scores = np.nan_to_num(F1_scores)
        best_threshold = thresholds[np.argmax(F1_scores)]
        self.threshold = best_threshold  # Set the threshold
        return self.threshold
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        self.params.update(params)
        self.model.set_params(**params)
        return self  # Return self to support chaining

    def score(self, X, y):
        """Calculate a custom score (if required for BayesSearchCV compatibility)."""
        preds = self.predict(X)
        return np.mean(preds == y)  # Example score: accuracy

def subsampleNormalData(X_train, y_train, subsample_ratio=1.0):
    # Assuming you have X_train and y_train as numpy arrays
    # Filter indices where y_train is equal to 0 (normal class)
    normal_indices = np.where(y_train == 0)[0]
    anomaly_indices = np.where(y_train > 0)[0]

    # Calculate the number of normal samples to subsample
    num_normal_samples_to_subsample = int(len(normal_indices) * subsample_ratio)

    # Randomly select indices to subsample from the normal class
    subsampled_normal_indices = np.random.choice(
        normal_indices, num_normal_samples_to_subsample, replace=False
    )

    # Combine the subsampled normal indices with the anomaly indices
    selected_indices = np.concatenate((subsampled_normal_indices, anomaly_indices))
    selected_indices = np.sort(selected_indices)

    # Use the selected indices to get the subsampled data
    X_subsampled = X_train[selected_indices]
    y_subsampled = y_train[selected_indices]
    return X_subsampled, y_subsampled


def hyperparameterTuning(
    model, X_train, y_train, X_val, y_val, hypertuning_subsample=1.0, seed=None, minutes_hyperparameter_search=120
):
    X_train_small, y_train_small = X_train, y_train
    # Subsample the training data for hyperparameter tuning
    if hypertuning_subsample < 1.0:
        X_train_small, y_train_small = subsampleNormalData(
            X_train, y_train, hypertuning_subsample
        )
    # Define the parameter search space
    if model == "XGB":
        classifier = XGBModel(objective="binary:logistic", random_state=seed).model
        param_space = {
            "n_estimators": Integer(10, 500),  # Number of trees
            "max_depth": Integer(2, 15),  # Maximum tree depth
            "max_features": Categorical(
                ["auto", "sqrt", "log2", None]
            ),  # Number of features to consider at each split
            "subsample": Real(0.8, 1.0),  # Subsample ratio
            "learning_rate": (0.01, 0.5, "log-uniform"),
            "colsample_bytree": Real(0.4, 1.0),
        }
    elif model == "RF":
        classifier = RandomForestClassifier(random_state=seed)
        param_space = {
            "n_estimators": (10, 500),  # Number of trees in the forest
            "max_depth": (1, 32),  # Maximum depth of each tree
            "min_samples_split": (2, 10),  # Minimum samples required to split a node
            "min_samples_leaf": (1, 10),  # Minimum samples required at each leaf node
        }
    elif "_" in model:
        modelsplit = model.split("_")
        if modelsplit[0].lower() == "pyod":
            if modelsplit[1] == "HBOS":
                classifier = PyODModel(modelsplit[1])
                param_space = {
                    "n_bins" : ["auto"], # The number of bins. "auto" uses the birge-rozenblac method for automatic selection of the optimal number of bins for each feature.
                    "alpha ": (0, 1),  # The regularizer for preventing overflow.
                    "tol ": (0, 1),  # The parameter to decide the flexibility while dealing the samples falling outside the bins.
                }
            elif modelsplit[1] == "IForest":
                classifier = IForestModel(random_state=seed)
                # print("classifier looks like: ", classifier.model, modelsplit[1])
                param_space = {
                    "n_estimators": (50, 200),  # The number of base estimators in the ensemble.
                    "max_samples": (128, 512),  # The number of samples to draw from X to train each base estimator
                    "max_features": (.5, 1),  # The number of features to draw from X to train each base estimator.
                    "bootstrap": [True,False],  # If True, individual trees are fit on random subsets of the training data sampled with replacement. If False, sampling without replacement is performed.
                    "behaviour": ["New"],  #  Passing behaviour='new' makes the decision_function change to match other anomaly detection algorithm API which will be the default behaviour in the future.
                }
            elif modelsplit[1] == "CBLOF":
                classifier = PyODModel(modelsplit[1])
                param_space = {
                    "n_clusters": (8, 128),  # The number of clusters to form as well as the number of centroids to generate.
                    "alpha": (.5, 1),  # Coefficient for deciding small and large clusters. The ratio of the number of samples in large clusters to the number of samples in small clusters.
                    "beta": (1, 20),  # Coefficient for deciding small and large clusters. For a list sorted clusters by size |C1|, |C2|, ..., |Cn|, beta = |Ck|/|Ck-1|
                    "use_weights": [True,False],  #  If set to True, the size of clusters are used as weights in outlier score calculation
                }
    # Combine your training and validation sets
    X_combined = np.vstack((X_train_small, X_val))
    y_combined = np.concatenate((y_train_small, y_val))

    # Create the split index (train=-1, validation=0)
    split_index = [-1] * len(X_train_small) + [0] * len(
        X_val
    )  # -1 for training, 0 for validation
    pds = PredefinedSplit(test_fold=split_index)
    f1 = make_scorer(f1_score, average="binary")
    # Perform Bayesian Hyperparameter Optimization
    bayes_search = BayesSearchCV(
        classifier,
        param_space,
        n_iter=40,  # Number of search iterations
        scoring="roc_auc",
        # scoring=f1,
        cv=pds,  # Number of cross-validation folds
        n_jobs=20,
        random_state=seed,
        verbose=10,
    )

    start_time = time.time()
    time_budget = 60*minutes_hyperparameter_search #convert minute to seconds
    print("time budget for hyperparameter tuning: ", time_budget)
    def stop_after_time_budget(result):
        if time.time() - start_time > time_budget:
            print(f"Time budget exceeded: {(time.time() - start_time)/60:.2f} minutes.")
            return True
        return False

    # bayes_search.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
    #bayes_search.fit(X_combined, y_combined,callback=[stop_after_time_budget])
    bayes_search.fit(X_combined, y_combined)
    # Get the best hyperparameters
    best_params = bayes_search.best_params_

    if model == "XGB":
        # Train the final model with the best hyperparameters on the combined train and validation data
        final_model = XGBModel(random_state=seed, **best_params)
    elif model == "RF":
        final_model = SkModelWrapper(
            RandomForestClassifier(random_state=seed, **best_params)
        )
    elif "_" in model:
        modelsplit = model.split("_")
        if modelsplit[0].lower() == "pyod":
            final_model = IForestModel(random_state=seed, **best_params)
    return final_model, best_params


def getModel(
    model="RF",
    model_path=None,
    ClassWeights=None,
    NumTrees=100,
    max_depth=None,
    seed=-1,
    scale_pos_weight=1,
    max_features="sqrt",
    n_features=None,
    modelargs={},
):
    best_model = None
    if model_path is not None:
        if model_path.split(".")[-1] == "pkl":
            with open(model_path, "rb") as f:
                best_model = pkl.load(f)
        else:
            best_model = joblib.load(model_path)
    else:
        if model == "RF":
            # best_model = RandomForestClassifier(**grid_search.best_params_)
            if ClassWeights is None:
                if seed != -1:
                    best_model = SkModelWrapper(
                        RandomForestClassifier(
                            random_state=random.seed(seed),
                            n_jobs=-1,
                            n_estimators=NumTrees,
                            max_depth=max_depth,
                            max_features=max_features,
                        )
                    )
                else:
                    best_model = SkModelWrapper(
                        RandomForestClassifier(
                            n_jobs=-1,
                            n_estimators=NumTrees,
                            max_depth=max_depth,
                            max_features=max_features,
                        )
                    )
            elif ClassWeights == "balanced" or ClassWeights == "balanced_subsample":
                if seed != -1:
                    best_model = SkModelWrapper(
                        RandomForestClassifier(
                            class_weight=ClassWeights,
                            random_state=random.seed(seed),
                            n_jobs=-1,
                            n_estimators=NumTrees,
                            max_depth=max_depth,
                            max_features=max_features,
                        )
                    )
                else:
                    best_model = SkModelWrapper(
                        RandomForestClassifier(
                            class_weight=ClassWeights,
                            n_jobs=-1,
                            n_estimators=NumTrees,
                            max_depth=max_depth,
                            max_features=max_features,
                        )
                    )
            else:
                if seed != -1:
                    best_model = SkModelWrapper(
                        BalancedRandomForestClassifier(
                            random_state=random.seed(seed), max_features=max_features
                        )
                    )
                else:
                    best_model = SkModelWrapper(
                        BalancedRandomForestClassifier(max_features=max_features)
                    )
        elif model == "LR":
            # best_model = LogisticRegression(**grid_search.best_params_)
            best_model = SkModelWrapper(LogisticRegression(**modelargs))
            # best_model = LogisticRegression()
        elif model == "KNN":
            best_model = SkModelWrapper(KNeighborsClassifier(**modelargs))
        elif model == "MLP":
            best_model = SkModelWrapper(MLPClassifier(**modelargs))
        elif model == "NB":
            best_model = SkModelWrapper(GaussianNB(**modelargs))
        elif model == "XGB":
            seed = None if seed == -1 else seed
            if ClassWeights is not None:
                best_model = XGBModel(
                    seed=seed,
                    n_jobs=-1,
                    n_estimators=NumTrees,
                    scale_pos_weight=scale_pos_weight,
                    **modelargs,
                )
            else:
                best_model = XGBModel(
                    seed=seed, n_jobs=-1, n_estimators=NumTrees, **modelargs
                )
        elif model == "SVM":
            best_model = SVC()
        elif model == "SVMapprox":
            best_model = Pipeline(
                [
                    (
                        "feature_map",
                        Nystroem(gamma=1 / n_features, n_components=300, n_jobs=-1),
                    ),  # Kernel approximation
                    (
                        "sgd",
                        SGDClassifier(
                            loss="hinge",
                            max_iter=1000,
                            tol=1e-3,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),  # Linear SVM approximation
                ]
            )
        elif model == "SVMbagging":
            n_estimators = 5
            best_model = BaggingClassifier(
                base_estimator=SVC(),
                max_samples=1.0 / n_estimators,
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=5,
                bootstrap=False,
            )
        elif model == "PCA":
            best_model = Pca(
                method="mahalanobis", n_selected_components=-1, **modelargs
            )
        elif model == "PCA_reconstruction":
            print(modelargs)
            best_model = Pca(method="reconstruction", **modelargs)
        elif "_" in model:
            modelsplit = model.split("_")
            if modelsplit[0].lower() == "pyod":
                if modelsplit[1] == "VAE":
                    best_model = PyODModel(
                        model=modelsplit[1],
                        encoder_neurons=[64, 32],
                        decoder_neurons=[32, 64],
                        epochs=10,
                        batch_size=64,
                        contamination=1e-5,
                        verbose=1,
                    )
                elif modelsplit[1] == "AE":
                    best_model = PyODModel(
                        model=modelsplit[1],
                        hidden_neurons=[64],
                        epochs=10,
                        batch_size=64,
                        contamination=1e-5,
                    )
                elif modelsplit[1] == "IForest":
                    best_model = IForestModel(random_state=seed, **modelargs)
                else:
                    best_model = PyODModel(model=modelsplit[1], **modelargs)
        elif model == "RFMahal":
            best_model = RFMahal(
                random_state=random.seed(seed),
                n_jobs=-1,
                n_estimators=NumTrees,
                max_depth=max_depth,
                max_features=max_features,
                **modelargs,
            )
        elif model == "XGBODMahal":
            best_model = XGBODMahal(
                seed=seed, n_jobs=-1, n_estimators=NumTrees, **modelargs
            )
        elif model == "RFODMahal":
            best_model = RFODMahal(
                random_state=random.seed(seed),
                n_jobs=-1,
                n_estimators=NumTrees,
                max_depth=max_depth,
                max_features=max_features,
                **modelargs,
            )
    return best_model


def get_results(
    X,
    y,
    y_mask,
    modelstr,
    best_model,
    Type="Test",
    save_path=f"{FILE_PATH}/Predictions",
):
    # if base_model is PCA or PyOD, set the threshold
    if (
        (modelstr == "PCA")
        or (modelstr == "PCA_reconstruction")
        or (modelstr.split("_")[0].lower() == "pyod")
        or (modelstr == "LR")
        or (modelstr == "KNN")
        or (modelstr == "MLP")
        or (modelstr == "NB")
        or (modelstr == "SVM")
        or (modelstr == "SVMapprox")
        or (isinstance(best_model, SkModelWrapper))
        or (isinstance(best_model, RFModel))
        or (modelstr == "XGB" and best_model.best_thr)
        or (modelstr == "RFMahal" and best_model.best_thr)
        or (modelstr == "XGBODMahal" and best_model.best_thr)
        or (modelstr == "RFODMahal" and best_model.best_thr)
    ):
        print("Setting threshold")
        scores = best_model.set_threshold(X, y_mask)
        print("Best Threshold:", best_model.threshold)
    else:
        scores = best_model.decision_function(X)
    # Evaluate the model on the validation set
    y_predict = best_model.predict(X)
    results_dict = classification_report(y_mask, y_predict, output_dict=True)
    df = pd.DataFrame(results_dict).T
    f_score_dict, precision, recall_dict = precision_recall_f1_pertype(y, y_predict)
    # if no anomaly in y_mask set auc, and apk, ark to -1
    if sum(y_mask) == 0:
        scores_auc = -1
        apk = -1
        ark = -1
        F1k = -1
        apks = [-1, -1, -1, -1]
        arks = [-1, -1, -1, -1]
        F1ks = [-1, -1, -1, -1]
        precisions = [-1] * 200
        recalls = [-1] * 200
    else:
        # get auc score
        scores_auc = roc_auc_score(y_mask, scores)
        apk, precisions = ap_at_k(scores, y_mask, 200)
        ark, recalls = ar_at_k(scores, y_mask, 200)
        apks, arks = get_ap_at_perType(scores, y, 200)
        if apk == 0 and ark == 0:
            F1k = 0.0 
        else:
            F1k = 2 * (apk * ark) / (apk + ark)
    # save the results
    np.save(os.path.join(save_path, f"{Type}_scores.npy"), scores)
    np.save(os.path.join(save_path, f"{Type}_y_predict.npy"), y_predict)
    np.save(os.path.join(save_path, f"{Type}_precisions.npy"), precisions)
    np.save(os.path.join(save_path, f"{Type}_recalls.npy"), recalls)
    wandb.log({f'{Type}_f1': f_score_dict['mixed'], f'{Type}_precision': precision, f'{Type}_recall': recall_dict['mixed'], f'{Type}_auc': scores_auc})
    wandb.log({f'{Type}_f1_T{int(k)}': val for k, val in f_score_dict.items() if isinstance(k, numbers.Number)})
    wandb.log({f'{Type}_recall_T{int(k)}': val for k, val in recall_dict.items() if isinstance(k, numbers.Number)})
	
    return (
        results_dict,
        df,
        f_score_dict,
        precision,
        recall_dict,
        scores_auc,
        apk,
        ark,
        F1k,
        apks,
        arks,
    )


def get_result_as_csv(
    y_mask, y_bank, results_dict, f_score_dict, precision, recall_dict
):
    result_row = [-1] * 20
    result_row[3] = len(y_mask)
    result_row[7] = sum(y_bank == 1)
    result_row[11] = sum(y_bank == 2)
    result_row[15] = sum(y_bank == 3)
    result_row[19] = sum(y_bank == 4)
    if sum(y_mask) == 0:
        return result_row
    else:
        if "1" in results_dict:
            result_row[:4] = [
                results_dict["1"]["f1-score"],
                results_dict["1"]["precision"],
                results_dict["1"]["recall"],
                len(y_mask),
            ]
        if 1 in f_score_dict:
            result_row[4:8] = [
                f_score_dict[1],
                precision,
                recall_dict[1],
                sum(y_bank == 1),
            ]
        if 2 in f_score_dict:
            result_row[8:12] = [
                f_score_dict[2],
                precision,
                recall_dict[2],
                sum(y_bank == 2),
            ]
        if 3 in f_score_dict:
            result_row[12:16] = [
                f_score_dict[3],
                precision,
                recall_dict[3],
                sum(y_bank == 3),
            ]
        if 4 in f_score_dict:
            result_row[16:20] = [
                f_score_dict[4],
                precision,
                recall_dict[4],
                sum(y_bank == 4),
            ]
        return result_row


# Set up the argparse argument parser
parser = argparse.ArgumentParser(description="Models for anomaly detection")
parser.add_argument(
    "--datapath",
    default="PATH_TO_DATA",
    help="Path to the folder that contains data folders",
)
parser.add_argument(
    "--datafolder",
    default="",
    help="Name of the data folder",
)
parser.add_argument(
    "--subsample",
    type=float,
    default=1.0,
    help="subsample proportion for hyperparameter tuning",
)
parser.add_argument("--model", type=str, default="RF", help="which model to use")
parser.add_argument(
    "--ClassWeights", type=str, default=None, help="ClassWeights for RF"
)
parser.add_argument("--Load", type=str, default=None, help="Model checkpoint to load")
parser.add_argument("--seed", type=int, default=123, help="random seed")
parser.add_argument(
    "--NumTrAno", type=int, default=600, help="Number of anomaly data in training set"
)
parser.add_argument(
    "--MultClass",
    type=str,
    default="False",
    help="Use multiclass classification using types as labels",
)
parser.add_argument(
    "--NumTrees",
    type=int,
    default=100,
    help="Number of ensemble trees for xgboost and random forest",
)
parser.add_argument("--max_depth", type=int, default=None, help="max depth in RF")
parser.add_argument(
    "--NormSub", type=float, default=1.0, help="subsample proportion for normal data"
)
parser.add_argument(
    "--PerClassTrain", type=str, default="False", help="Train each class separately"
)
parser.add_argument(
    "--HyperTune", type=str, default="False", help="Hyperparameter tuning"
)
parser.add_argument(
    "--hypertuning_subsample",
    type=float,
    default=1.0,
    help="subsample proportion for hyperparameter tuning",
)
parser.add_argument(
    "--prefix", type=str, default="Rank", help="prefix for the model name"
)
parser.add_argument(
    "--IncludedFeatures", type=str, default="All", help="Features to be included"
)
parser.add_argument(
    "--RemoveFeatures", type=str, default="", help="Features to be removed"
)
parser.add_argument(
    "--OutFile", type=str, default="outputs_NewData2.txt", help="Output file name"
)
parser.add_argument(
    "--TrainPerBank", type=str, default="False", help="Train each bank separately"
)
parser.add_argument(
    "--GeneralModel", type=str, default="False", help="Use General model for all banks"
)
parser.add_argument(
    "--DropNaNs", type=str, default="False", help="Drop rows with NaNs from the data"
)
parser.add_argument(
    "--FineTune",
    type=str,
    default="False",
    help="Fine tune the general model for each bank",
)
parser.add_argument(
    "--modelargs",
    action=DictParseAction,
    default={},
    help='Model-specific arguments (e.g., "max_depth:5,n_estimators:100,encoder_neurons:32_64")',
)
parser.add_argument("--minutes_hyperparameter_search", type=int, default=120, help="number of minutes allowed for hyperparameter search")


# Parse the command line arguments
args = parser.parse_args()

project_name = f"{args.model}_{args.NumTrAno}_Sweep"
wandb.init(project=project_name)

print(args)

data_path = os.path.join(args.datapath, args.datafolder)
Folder_prefix = str(args.NumTrAno)
# Load the data
X_train = np.load(os.path.join(data_path, "train.npz"))["sender_2"]
y_train = np.load(os.path.join(data_path, "y_train.npz"))["sender_2"]

if args.NormSub < 1.0:
    X_train, y_train = subsampleNormalData(X_train, y_train, args.NormSub)

X_val = np.load(os.path.join(data_path, "val.npz"))["sender_2"]
y_val = np.load(os.path.join(data_path, "y_val.npz"))["sender_2"]

X_test = np.load(os.path.join(data_path, "test.npz"))["sender_2"]
y_test = np.load(os.path.join(data_path, "y_test.npz"))["sender_2"]

with open(os.path.join(data_path, "features_info.json"), "r") as f:
    data_info = json.load(f)

feature_names = data_info["output_feature_names"]

with open(os.path.join(data_path, "bank_to_id.json"), "r") as f:
    bank_to_id = json.load(f)
# reverse bank_to_id to get id_to_bank
id_to_bank = {v: k for k, v in bank_to_id.items()}

if args.IncludedFeatures != "All":
    IncludedFeatures = args.IncludedFeatures.split(",")
    IncludedFeat_indxs = [feature_names.index(feat) for feat in IncludedFeatures]
    X_train = X_train[:, IncludedFeat_indxs]
    X_val = X_val[:, IncludedFeat_indxs]
    X_test = X_test[:, IncludedFeat_indxs]
if args.RemoveFeatures != "":
    RemoveFeatures = args.RemoveFeatures.split(",")
    RemoveFeat_indxs = [feature_names.index(feat) for feat in RemoveFeatures]
    X_train = np.delete(X_train, RemoveFeat_indxs, axis=1)
    X_val = np.delete(X_val, RemoveFeat_indxs, axis=1)
    X_test = np.delete(X_test, RemoveFeat_indxs, axis=1)
if args.MultClass == "False":
    y_train_mask = y_train > 0
    y_train_mask = y_train_mask.astype(np.int8)

    y_val_mask = y_val > 0
    y_val_mask = y_val_mask.astype(np.int8)

    y_test_mask = y_test > 0
    y_test_mask = y_test_mask.astype(np.int8)
else:
    y_train_mask = y_train
    y_val_mask = y_val
    y_test_mask = y_test

if args.DropNaNs == "True":
    # drop rows with NaNs
    train_nan_rows = np.where(np.isnan(X_train).any(axis=1))[0]
    X_train = np.delete(X_train, train_nan_rows, axis=0)
    y_train_mask = np.delete(y_train_mask, train_nan_rows, axis=0)
    y_train = np.delete(y_train, train_nan_rows, axis=0)

    val_nan_rows = np.where(np.isnan(X_val).any(axis=1))[0]
    X_val = np.delete(X_val, val_nan_rows, axis=0)
    y_val_mask = np.delete(y_val_mask, val_nan_rows, axis=0)
    y_val = np.delete(y_val, val_nan_rows, axis=0)

    test_nan_rows = np.where(np.isnan(X_test).any(axis=1))[0]
    X_test = np.delete(X_test, test_nan_rows, axis=0)
    y_test_mask = np.delete(y_test_mask, test_nan_rows, axis=0)
    y_test = np.delete(y_test, test_nan_rows, axis=0)

if args.TrainPerBank == "True":
    senderID_indx = feature_names.index("SenderID")
    BanksIDs_are_Scaled = False
    if (X_train[:, senderID_indx] < 0).any():
        # if there are negative senderIDs, means that they are normalized, we have to inverse transform to get the values
        with open(os.path.join(data_path, "scaler.pkl"), "rb") as f:
            scaler = pkl.load(f)
        SenderID_mean, SenderID_std = (
            scaler.mean_[senderID_indx],
            scaler.scale_[senderID_indx],
        )
        BankID_scaled = X_train[:, senderID_indx] * SenderID_std + SenderID_mean
        print(f"BankID_scaled {BankID_scaled}")
        BankID_scaled = np.round(BankID_scaled)
        bank_to_bankIDscaled = dict(zip(X_train[:, senderID_indx], BankID_scaled))
        BanksIDs_are_Scaled = True
    Banks = np.unique(X_train[:, senderID_indx])
    out_name = f"Results/PerBank/Outputs_{args.prefix}{args.model}_PerBank.txt"
    csv_out = f"Results/PerBank/Outputs_{args.prefix}{args.model}_PerBank.csv"
    if args.GeneralModel == "True":
        out_name = (
            f"Results/PerBank/Outputs_{args.prefix}{args.model}_PerBank_General.txt"
        )
        csv_out = (
            f"Results/PerBank/Outputs_{args.prefix}{args.model}_PerBank_General.csv"
        )
    elif args.FineTune == "True":
        out_name = (
            f"Results/PerBank/Outputs_{args.prefix}{args.model}_PerBank_FineTune.txt"
        )
        csv_out = (
            f"Results/PerBank/Outputs_{args.prefix}{args.model}_PerBank_FineTune.csv"
        )
    with open(out_name, "a+") as f:
        f.write(
            f"{args.model}, f1_scoring, data_path={data_path}, ClassWeights={args.ClassWeights}, train_shape={X_train.shape}, n_features={X_train.shape[1]}, Load={args.Load}, seed={args.seed}, Multi={args.MultClass}, NumTrees={args.NumTrees}, maxDepth={args.max_depth}, NormSub={args.NormSub}, hypertuning_subsample={args.hypertuning_subsample} \n"
        )
    results_csv = open(csv_out, "w+")
    results_writer = csv.writer(results_csv)
    results_writer.writerow(
        [
            "SenderID",
            "BIC",
            "Train_f1",
            "Train_precision",
            "Train_recall",
            "NumTrain",
            "T1_f1score_train",
            "T1_precision_train",
            "T1_recall_train",
            "NumT1_train",
            "T2_f1score_train",
            "T2_precision_train",
            "T2_recall_train",
            "NumT2_train",
            "T3_f1score_train",
            "T3_precision_train",
            "T3_recall_train",
            "NumT3_train",
            "Val_f1",
            "Val_precision",
            "Val_recall",
            "NumVal",
            "T1_f1score_val",
            "T1_precision_val",
            "T1_recall_val",
            "NumT1_val",
            "T2_f1score_val",
            "T2_precision_val",
            "T2_recall_val",
            "NumT2_val",
            "T3_f1score_val",
            "T3_precision_val",
            "T3_recall_val",
            "NumT3_val",
            "Test_f1",
            "Test_precision",
            "Test_recall",
            "NumTest",
            "T1_f1score_test",
            "T1_precision_test",
            "T1_recall_test",
            "NumT1_test",
            "T2_f1score_test",
            "T2_precision_test",
            "T2_recall_test",
            "NumT2_test",
            "T3_f1score_test",
            "T3_precision_test",
            "T3_recall_test",
            "NumT3_test",
        ]
    )
    results_csv.flush()
    # Create and train a Random Forest classifier for each class
    classifiers = {}
    GenModel = None
    if (args.GeneralModel == "True") or (
        (args.FineTune == "True") and (args.model != "PCA")
    ):
        if args.Load is None:
            scale_pos_weight = np.sum(y_train_mask == 0) / np.sum(y_train_mask == 1)
            GenModel = getModel(
                model=args.model,
                model_path=args.Load,
                ClassWeights=args.ClassWeights,
                NumTrees=args.NumTrees,
                max_depth=args.max_depth,
                seed=args.seed,
                scale_pos_weight=scale_pos_weight,
                max_features=None,
                n_features=X_train.shape[1],
                modelargs=args.modelargs,
            )
            GenModel.fit(X_train, y_train_mask)
        else:
            with open(args.Load, "rb") as f:
                GenModel = pkl.load(f)
                # if GenModel is XGBClassifier wrap it in XGBModel
                if isinstance(GenModel, xgb.XGBClassifier):
                    xgbmodel = XGBModel()
                    xgbmodel.model = GenModel
                    GenModel = xgbmodel
                # if GenModel is RandomForestClassifier wrap it in RFModel
                elif isinstance(GenModel, RandomForestClassifier):
                    rfmodel = RFModel()
                    rfmodel.model = GenModel
                    GenModel = rfmodel
                elif isinstance(GenModel, SkModelWrapper) and isinstance(
                    GenModel.model, RandomForestClassifier
                ):
                    rfmodel = RFModel()
                    rfmodel.model = GenModel.model
                    GenModel = rfmodel
    for bank in Banks:
        bankID = bank
        if BanksIDs_are_Scaled:
            bankID = bank_to_bankIDscaled[bank]
        print(f"-----------SenderID {bankID} Bank {id_to_bank[bankID]} -------------")
        X_train_bank = X_train[(X_train[:, senderID_indx] == bank), :]
        y_train_bank_mask = y_train_mask[(X_train[:, senderID_indx] == bank)]

        X_val_bank = X_val[(X_val[:, senderID_indx] == bank), :]
        y_val_bank_mask = y_val_mask[X_val[:, senderID_indx] == bank]
        y_val_bank = y_val[X_val[:, senderID_indx] == bank]

        # if (sum(y_train_bank_mask) > 0) or (args.GeneralModel == "True"): # if there is at least one anomaly in the training set of the bank or we are using a general model
        if (args.GeneralModel == "False") and (args.FineTune == "False"):
            scale_pos_weight = np.sum(y_train_bank_mask == 0) / np.sum(
                y_train_bank_mask == 1
            )
            clf = getModel(
                model=args.model,
                model_path=args.Load,
                ClassWeights=args.ClassWeights,
                NumTrees=args.NumTrees,
                max_depth=args.max_depth,
                seed=args.seed,
                scale_pos_weight=scale_pos_weight,
                max_features=None,
                n_features=X_train.shape[1],
                modelargs=args.modelargs,
            )
            # Create binary labels where the current class is 1 and others are 0
            clf.fit(X_train_bank, y_train_bank_mask)
            if (
                (args.model == "PCA") or (args.model.split("_")[0].lower() == "pyod")
            ) and (args.Load is None):
                clf.set_threshold(X_val_bank, y_val_bank_mask)
            classifiers[bankID] = clf
        else:
            clf = copy.deepcopy(GenModel)
            if args.FineTune == "True":
                print(f"finetuning {args.model} model for bank {bankID}")
                if (
                    (args.model == "XGB")
                    or (args.model == "RF")
                    or (args.model == "RFMahal")
                    or (args.model == "XGBODMahal")
                    or (args.model == "RFODMahal")
                ):
                    clf = clf.finetune(
                        X_train_bank, y_train_bank_mask, general_model=GenModel.model
                    )
                elif args.model == "PCA":
                    weights = np.ones_like(X_train)
                    # weights[(X_train[:, senderID_indx] == bank), :] = X_train.shape[0] / X_train_bank.shape[0]
                    weights[(X_train[:, senderID_indx] == bank), :] = 10
                    clf = Pca(method="mahalanobis", n_selected_components=-1)
                    clf.finetune(
                        X_train,
                        y_train_mask,
                        X_val=X_val_bank,
                        y_val=y_val_bank_mask,
                        weights=weights,
                    )
            classifiers[bankID] = clf
        # Evaluate and write results for each class
        result_row = []
        result_row += [bankID, id_to_bank[bankID]]
        y_train_bank = y_train[X_train[:, senderID_indx] == bank]
        prefix = "PerBank"
        if args.FineTune == "True":
            prefix = "FineTune"
        elif args.GeneralModel == "True":
            prefix = "General"
        pred_path = f"Predictions_PerBank/{prefix}_{args.prefix}_{args.model}_{Folder_prefix}_model_{args.MultClass}_NT{args.NumTrees}_MD{args.max_depth}_{args.NormSub}_{args.ClassWeights}_{args.HyperTune}/Bank_{id_to_bank[bankID]}/"
        os.makedirs(pred_path, exist_ok=True)
        # Evaluate on train set
        (
            train_results_dict,
            df_train,
            f_score_dict_train,
            precision_train,
            recall_dict_train,
            auc_train,
            apk_train,
            ark_train,
            F1k_train,
            apks_train,
            arks_train,
        ) = get_results(
            X_train_bank,
            y_train_bank,
            y_train_bank_mask,
            args.model,
            clf,
            Type="Train",
            save_path=pred_path,
        )
        print(f"Train Results: {df_train}")
        print(
            f"f1_scores: {f_score_dict_train}\n precision: {precision_train}\n recall_dict: {recall_dict_train}\n auc: {auc_train}\n"
        )
        print(
            f"apk: {apk_train}, ark: {ark_train}, F1k: {F1k_train}, apks: {apks_train}, arks: {arks_train}\n"
        )
        result_row += get_result_as_csv(
            y_train_bank_mask,
            y_train_bank,
            train_results_dict,
            f_score_dict_train,
            precision_train,
            recall_dict_train,
        )

        # Evaluate on validation set
        (
            Val_results_dict,
            df_val,
            f_score_dict_val,
            precision_val,
            recall_dict_val,
            auc_val,
            apk_val,
            ark_val,
            F1k_val,
            apks_val,
            arks_val,
        ) = get_results(
            X_val_bank,
            y_val_bank,
            y_val_bank_mask,
            args.model,
            clf,
            Type="Val",
            save_path=pred_path,
        )
        print(f"Val Results: {df_val}")
        print(
            f"f1_scores: {f_score_dict_val}\n precision: {precision_val}\n recall_dict: {recall_dict_val}\n auc: {auc_val}\n"
        )
        print(
            f"apk: {apk_val}, ark: {ark_val}, F1k: {F1k_val}, apks: {apks_val}, arks: {arks_val}\n"
        )
        result_row += get_result_as_csv(
            y_val_bank_mask,
            y_val_bank,
            Val_results_dict,
            f_score_dict_val,
            precision_val,
            recall_dict_val,
        )

        # Evaluate on test set
        X_test_bank = X_test[(X_test[:, senderID_indx] == bank), :]
        y_test_bank_mask = y_test_mask[X_test[:, senderID_indx] == bank]
        y_test_bank = y_test[X_test[:, senderID_indx] == bank]
        (
            test_results_dict,
            df_test,
            f_score_dict_test,
            precision_test,
            recall_dict_test,
            auc_test,
            apk_test,
            ark_test,
            F1k_test,
            apks_test,
            arks_test,
        ) = get_results(
            X_test_bank,
            y_test_bank,
            y_test_bank_mask,
            args.model,
            clf,
            Type="Test",
            save_path=pred_path,
        )
        print(f"Test Results: {df_test}")
        # recall_dict: {'mixed': , 1: , 2: , 3: , 4: , 'avg': }
        print(
            f"f1_scores: {f_score_dict_test}\n precision: {precision_test}\n recall_dict: {recall_dict_test}\n auc: {auc_test}\n"
        )
        print(
            f"apk: {apk_test}, ark: {ark_test}, F1k: {F1k_test}, apks: {apks_test}, arks: {arks_test}\n"
        )
        result_row += get_result_as_csv(
            y_test_bank_mask,
            y_test_bank,
            test_results_dict,
            f_score_dict_test,
            precision_test,
            recall_dict_test,
        )

        # Write the results to csv
        results_writer.writerow(result_row)
        results_csv.flush()

        with open(out_name, "a+") as f:
            f.write(
                f"-----------SenderID {bankID} Bank {id_to_bank[bankID]} -------------\n"
            )
            f.write(f"train: \n")
            f.write(f"{df_train}\n")
            f.write(
                f"f1_scores: {f_score_dict_train}\n precision: {precision_train}\n recall_dict: {recall_dict_train}\n auc: {auc_train}\n"
            )
            f.write(f"val: \n")
            f.write(f"{df_val}\n")
            f.write(
                f"f1_scores: {f_score_dict_val}\n precision: {precision_val}\n recall_dict: {recall_dict_val}\n auc: {auc_val}\n"
            )
            f.write(f"test: \n")
            f.write(f"{df_test}\n")
            f.write(
                f"f1_scores: {f_score_dict_test}\n precision: {precision_test}\n recall_dict: {recall_dict_test}\n auc: {auc_test}\n"
            )
    with open(out_name, "a+") as f:
        f.write(
            f"--------------------------------------------------------------------------------\n\n"
        )
    results_csv.close()
    with open(
        f"{FILE_PATH}/ModelChcks/PerBank/{args.model}_{Folder_prefix}_model_{args.MultClass}_NT{args.NumTrees}_{args.NormSub}_{args.ClassWeights}_GeneralModel{args.GeneralModel}_PerBank.pkl",
        "wb",
    ) as f:
        pkl.dump(classifiers, f)
if args.PerClassTrain == "False":
    # Take time from here
    start_time = time.time()
    if args.HyperTune == "True":
        print("Starting hyperparameter tuning...")
        # best_model, best_params = hyperparameterTuning(args.model, X_train, y_train_mask, X_val, y_val_mask, seed=(None if args.seed == -1 else args.seed), hypertuning_subsample=args.hypertuning_subsample)
        best_model, best_params = hyperparameterTuning(
            args.model,
            X_train,
            y_train_mask,
            X_test,
            y_test_mask,
            seed=(None if args.seed == -1 else args.seed),
            hypertuning_subsample=args.hypertuning_subsample,
            minutes_hyperparameter_search=args.minutes_hyperparameter_search
        )
    elif args.HyperTune == "Best":
        if args.model == "XGB":
            best_params = OrderedDict(
                [
                    ("colsample_bytree", 0.8661735495383152),
                    ("learning_rate", 0.2321174391714492),
                    ("max_depth", 3),
                    ("n_estimators", 353),
                    ("subsample", 0.8959290000765973),
                ]
            )
            print(f"Using parameters {best_params}")
            best_model = xgb.XGBClassifier(random_state=args.seed, **best_params)
        elif args.model == "RF":
            best_params = OrderedDict(
                [
                    ("max_depth", 19),
                    ("min_samples_leaf", 1),
                    ("min_samples_split", 10),
                    ("n_estimators", 154),
                ]
            )
            print(f"Using parameters {best_params}")
            best_model = SkModelWrapper(
                RandomForestClassifier(random_state=args.seed, **best_params)
            )
            # best_params = best_model.get_params()
    else:
        scale_pos_weight = np.sum(y_train_mask == 0) / np.sum(y_train_mask == 1)
        print(f"scale_pos_weight: {scale_pos_weight}")
        best_model = getModel(
            model=args.model,
            model_path=args.Load,
            ClassWeights=args.ClassWeights,
            NumTrees=args.NumTrees,
            max_depth=args.max_depth,
            seed=args.seed,
            scale_pos_weight=scale_pos_weight,
            n_features=X_train.shape[1],
            modelargs=args.modelargs,
        )

    print(f"Fitting {args.model} on entire dataset...")
    if args.Load is None:
        best_model.fit(X_train, y_train_mask)
    pred_path = f"{FILE_PATH}/Predictions/{args.prefix}_{args.model}_{Folder_prefix}_model_{args.MultClass}_NT{args.NumTrees}_MD{args.max_depth}_{args.NormSub}_{args.ClassWeights}_{args.HyperTune}/"
    os.makedirs(pred_path, exist_ok=True)
    # Evaluate the model on the train set
    (
        train_results_dict,
        df_train,
        f_score_dict_train,
        precision_train,
        recall_dict_train,
        auc_train,
        apk_train,
        ark_train,
        F1k_train,
        apks_train,
        arks_train,
    ) = get_results(
        X_train,
        y_train,
        y_train_mask,
        args.model,
        best_model,
        Type="Train",
        save_path=pred_path,
    )
    print(f"Train Results: {df_train}")
    print(
        f"f1_scores: {f_score_dict_train}\n precision: {precision_train}\n recall_dict: {recall_dict_train}\n auc: {auc_train}\n"
    )
    print(
        f"apk: {apk_train}, ark: {ark_train}, F1k: {F1k_train}, apks: {apks_train}, arks: {arks_train}\n"
    )

    # Evaluate the model on the validation set
    (
        val_results_dict,
        df_val,
        f_score_dict_val,
        precision_val,
        recall_dict_val,
        auc_val,
        apk_val,
        ark_val,
        F1k_val,
        apks_val,
        arks_val,
    ) = get_results(
        X_val,
        y_val,
        y_val_mask,
        args.model,
        best_model,
        Type="Val",
        save_path=pred_path,
    )
    print(f"Val Results: {df_val}")
    print(
        f"f1_scores: {f_score_dict_val}\n precision: {precision_val}\n recall_dict: {recall_dict_val}\n auc: {auc_val}\n"
    )
    print(
        f"apk: {apk_val}, ark: {ark_val}, F1k: {F1k_val}, apks: {apks_val}, arks: {arks_val}\n"
    )

    # Evaluate the model on the test set
    (
        test_results_dict,
        df_test,
        f_score_dict,
        precision,
        recall_dict,
        auc_test,
        apk_test,
        ark_test,
        F1k_test,
        apks_test,
        arks_test,
    ) = get_results(
        X_test,
        y_test,
        y_test_mask,
        args.model,
        best_model,
        Type="Test",
        save_path=pred_path,
    )
    print(f"Test Results: {df_test}")
    print(
        f"f1_scores: {f_score_dict}\n precision: {precision}\n recall_dict: {recall_dict}\n auc: {auc_test}\n"
    )
    print(
        f"apk: {apk_test}, ark: {ark_test}, F1k: {F1k_test}, apks: {apks_test}, arks: {arks_test}\n"
    )

    print(f"Time taken: {time.time() - start_time}")

    # Save the model
    print("Saving model...")
    if args.Load is None:
        model_path = f"{FILE_PATH}/ModelChcks/{args.prefix}_{args.model}_{Folder_prefix}_model_{args.MultClass}_NT{args.NumTrees}_MD{args.max_depth}_{args.NormSub}_{args.ClassWeights}_{args.HyperTune}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if args.model.split("_")[0].lower() == "pyod":
            pass
            # joblib.dump(best_model, model_path + '.joblib')
        else:
            with open(model_path + ".pkl", "wb") as f:
                pkl.dump(best_model, f)

    if os.path.dirname(args.OutFile):
        os.makedirs(os.path.dirname(args.OutFile), exist_ok=True)
    else:
        print("Error in outfile directory")
        
    with open(args.OutFile, "w") as f:
        f.write(
            f"{args.prefix} {args.model}, data_path={data_path}, ClassWeights={args.ClassWeights}, train_shape={X_train.shape}, n_features={X_train.shape[1]}, Load={args.Load}, seed={args.seed}, Multi={args.MultClass}, modelargs={args.modelargs} \n"
        )
        f.write(
            f"NumTrees={args.NumTrees}, maxDepth={args.max_depth}, NormSub={args.NormSub}, HyperTune {args.HyperTune}, hypertuning_subsample={args.hypertuning_subsample},  \n"
        )
        f.write(
            f"Included Features={args.IncludedFeatures} Removed features={args.RemoveFeatures} \n"
        )
        if args.HyperTune != "False":
            f.write(f"Best params: {best_params}\n")
        f.write(f"train: \n")
        f.write(f"{df_train}\n")
        f.write(
            f"f1_scores: {f_score_dict_train}\n precision: {precision_train}\n recall_dict: {recall_dict_train}\n auc: {auc_train}\n"
        )
        f.write(
            f"apk: {apk_train}, ark: {ark_train}, apks: {apks_train}, arks: {arks_train}\n"
        )
        f.write(f"val: \n")
        f.write(f"{df_val}\n")
        f.write(
            f"f1_scores: {f_score_dict_val}\n precision: {precision_val}\n recall_dict: {recall_dict_val}\n auc: {auc_val}\n"
        )
        f.write(f"apk: {apk_val}, ark: {ark_val}, apks: {apks_val}, arks: {arks_val}\n")
        f.write(f"test: \n")
        f.write(f"{df_test}\n")
        f.write(
            f"f1_scores: {f_score_dict}\n precision: {precision}\n recall_dict: {recall_dict}\n auc: {auc_test}\n"
        )
        f.write(
            f"apk: {apk_test}, ark: {ark_test}, apks: {apks_test}, arks: {arks_test}\n"
        )
        f.write(f"Time taken: {time.time() - start_time}\n")
        f.write(
            f"--------------------------------------------------------------------------------\n\n"
        )
else:
    # Get unique classes from y_train
    unique_classes = np.unique(y_train)

    # Create and train a Random Forest classifier for each class
    classifiers = {}
    for class_label in unique_classes:
        binary_y_train = np.where(y_train == class_label, 1, 0)
        # clf = getModel(model=args.model, model_path=args.Load, ClassWeights=args.ClassWeights, NumTrees=args.NumTrees, max_depth=args.max_depth, seed=args.seed)
        scale_pos_weight = np.sum(binary_y_train == 0) / np.sum(binary_y_train == 1)
        clf = getModel(
            model=args.model,
            model_path=args.Load,
            ClassWeights=args.ClassWeights,
            NumTrees=args.NumTrees,
            max_depth=args.max_depth,
            seed=args.seed,
            scale_pos_weight=scale_pos_weight,
            max_features=None,
            n_features=X_train.shape[1],
            modelargs=args.modelargs,
        )
        # Create binary labels where the current class is 1 and others are 0
        clf.fit(X_train, binary_y_train)
        classifiers[class_label] = clf

    with open("outputs_PerClass_FullFeats.txt", "a+") as f:
        f.write(
            f"{args.model}, f1_scoring, data_path={data_path}, ClassWeights={args.ClassWeights}, train_shape={X_train.shape}, n_features={X_train.shape[1]}, Load={args.Load}, seed={args.seed}, Multi={args.MultClass}, NumTrees={args.NumTrees}, maxDepth={args.max_depth}, NormSub={args.NormSub}, hypertuning_subsample={args.hypertuning_subsample} \n"
        )
    with open(
        f"ModelChcks/FullFeats_{args.model}_{Folder_prefix}_model_{args.MultClass}_NT{args.NumTrees}_{args.NormSub}_{args.ClassWeights}_PerClass_WithRank.pkl",
        "wb",
    ) as f:
        pkl.dump(classifiers, f)
    # Evaluate and write results for each class
    for class_label, clf in classifiers.items():
        print(f"-----------Class {class_label}-------------")
        # Evaluate on train set
        train_predictions = clf.predict(X_train)
        binary_y_train = np.where(y_train == class_label, 1, 0)
        df_train = pd.DataFrame(
            classification_report(binary_y_train, train_predictions, output_dict=True)
        ).T
        print(f"Train Results: {df_train}")

        # Evaluate on validation set
        val_predictions = clf.predict(X_val)
        binary_y_val = np.where(y_val == class_label, 1, 0)
        df_val = pd.DataFrame(
            classification_report(binary_y_val, val_predictions, output_dict=True)
        ).T
        print(f"Val Results: {df_val}")

        # Evaluate on test set
        test_predictions = clf.predict(X_test)
        binary_y_test = np.where(y_test == class_label, 1, 0)
        df_test = pd.DataFrame(
            classification_report(binary_y_test, test_predictions, output_dict=True)
        ).T
        print(f"Test Results: {df_test}")

        with open("outputs_PerClass_FullFeats.txt", "a+") as f:
            f.write(f"-----------Class {class_label}-------------\n")
            f.write(f"train: \n")
            f.write(f"{df_train}\n")
            f.write(f"val: \n")
            f.write(f"{df_val}\n")
            f.write(f"test: \n")
            f.write(f"{df_test}\n")
    with open("outputs_PerClass_FullFeats.txt", "a+") as f:
        f.write(
            f"--------------------------------------------------------------------------------\n\n"
        )
