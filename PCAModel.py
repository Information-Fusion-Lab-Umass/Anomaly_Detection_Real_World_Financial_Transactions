from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA as sklearn_PCA

from sklearn.metrics import precision_recall_curve
from wpca import PCA, WPCA, EMPCA

class Pca():
    def __init__(self, method: str, n_selected_components: Optional[Union[int, float]] = None):
        """
        Args:
            method: scoring method (either "reconstruction" or "mahalanobis").
            n_selected_components: number of components to consider in scoring (the
              highest-variance axes for "reconstruction", the lowest-variance axes for "mahalanobis").
        """
        print('n_selected_components', n_selected_components)
        self.method_ = method
        self.n_selected_components_ = n_selected_components
        self.threshold = None

    def fit(self, X_train, y_train=None, X_val=None, y_val=None, weights=None):
        if weights is not None:
            self.pca_ = WPCA()
            self.pca_.fit(X_train, weights=weights)
        else:
            self.pca_ = sklearn_PCA()
            self.pca_.fit(X_train)
        self.decision_scores_ = self.decision_function(X_train)

    def _get_n_selected_components(self):
        """Returns the number of components based on `self.n_selected_components_`."""
        if self.n_selected_components_ == -1:
            n_selected_components = self.pca_.components_.shape[0]
        elif isinstance(self.n_selected_components_, int):
            n_selected_components = self.n_selected_components_
        else:
            cumulative_variance = np.cumsum(self.pca_.explained_variance_ratio_)
            n_selected_components = (
                np.where(cumulative_variance > self.n_selected_components_)[0][0] + 1
            )
        return n_selected_components

    def transform_windows(self, X):
        transformed_X = None
        n_selected_components = self._get_n_selected_components()
        if self.method_ == "reconstruction":
            # select `n_selected_components` with the largest variance and place them as columns
            V = self.pca_.components_[:n_selected_components].T
            # transformed X in the reduced component space (projection)
            transformed_X = X.dot(V)
        elif self.method_ == "mahalanobis":
            # select `n_selected_components` with the smallest variance and place them as columns
            V = self.pca_.components_[-n_selected_components:].T
            lambdas = self.pca_.explained_variance_[-n_selected_components:]
            # transformed X in the reduced component space (whitening)
            transformed_X = X.dot(V) / np.sqrt(np.maximum(lambdas, 1e-7))
        return transformed_X

    def decision_function(self, X):
        n_selected_components = self._get_n_selected_components()
        if self.method_ == "reconstruction":
            # select `n_selected_components` with the largest variance and place them as columns
            V = self.pca_.components_[:n_selected_components].T
            # transformed/projected X expressed in the original space
            projected_X = X.dot(V).dot(V.T)
            # squared norm without computing the square roots for numerical stability
            window_scores = np.sum(np.square(X - projected_X), axis=1)
        elif self.method_ == "mahalanobis":
            # select `n_selected_components` with the smallest variance and place them as columns
            V = self.pca_.components_[-n_selected_components:].T
            lambdas = self.pca_.explained_variance_[-n_selected_components:]
            # transformed X in the reduced component space
            transformed_X = X.dot(V)
            # squared norm without computing square roots for numerical stability
            window_scores = np.sum(np.square(transformed_X) / lambdas, axis=1)
        else:
            raise NotImplementedError(
                'Only "reconstruction" and "mahalanobis" methods are supported.'
            )
        return window_scores
    
    def set_threshold(self, X_val, y_val):
        """Returns the threshold value for the window scoring function."""
        val_scores = self.decision_function(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        F1s = 2 * (precision * recall) / (precision + recall)
        F1s = np.nan_to_num(F1s)
        best_thr = thresholds[np.argmax(F1s)]
        self.threshold = best_thr
        return val_scores
    
    def predict(self, X):
        if self.threshold is None:
            raise ValueError('Threshold not set. Call set_threshold() first.')
        scores = self.decision_function(X)
        preds = (scores >= self.threshold).astype(int)
        return preds
    
    def finetune(self, X, y, X_val=None, y_val=None, **params):
        if params['weights'] is not None:
            self.pca_ = WPCA()
            self.pca_.fit(X, weights=params['weights'])
        else:
            self.pca_ = sklearn_PCA()
            self.pca_.fit(X)
        self.set_threshold(X_val, y_val)
        return self
