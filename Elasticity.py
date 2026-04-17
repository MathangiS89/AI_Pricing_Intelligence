# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:10:17 2026

@author: srima
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def get_model_raw_metrics(model, X_train, price_col='price'):
    """
    Extracts raw slope (dQ/dP), avg price, and avg volume using sklearn.
    Works for Linear, XGBoost, and Random Forest.
    """
    try:
        # 1. Create a price range for probing based on training data
        p_min, p_max = X_train[price_col].min(), X_train[price_col].max()
        probe_prices = np.linspace(p_min, p_max, 100).reshape(-1, 1)
        X_probe = pd.DataFrame(probe_prices, columns=[price_col])
        
        # 2. Get predictions from the model
        predicted_volumes = model.predict(X_probe)
        
        # 3. Fit a simple linear line to those predictions to get the global slope
        interpreter = LinearRegression()
        interpreter.fit(probe_prices, predicted_volumes)
        
        return {
            "raw_slope": float(interpreter.coef_[0]),
            "avg_price": float(X_train[price_col].mean()),
            "avg_volume": float(predicted_volumes.mean()),
            "confidence": float(interpreter.score(probe_prices, predicted_volumes))
        }
    except Exception as e:
        raise RuntimeError(f"Slope extraction failed: {str(e)}")