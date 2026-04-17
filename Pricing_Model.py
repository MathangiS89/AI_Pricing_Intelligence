# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:57:43 2026

@author: srima
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.inspection import partial_dependence
import numpy as np

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def get_model_slope(model, X_train, target_feature='price'):
    """
    Extracts the average slope (dQ/dP) from a fitted model using PDP.
    Used as a proxy for coefficients in tree-based models.
    """
    # Generate the PDP values
    pdp_results = partial_dependence(model, X_train, [target_feature], grid_resolution=50)
    prices = pdp_results['values'][0]
    predicted_volumes = pdp_results['average'][0]
    
    # Calculate the gradient (slope) at each point across the price grid
    slopes = np.gradient(predicted_volumes, prices)
    return np.mean(slopes)
