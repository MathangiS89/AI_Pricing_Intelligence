# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:10:17 2026

@author: srima
"""

def extract_linear_elasticity(model, feature_names):
    return dict(zip(feature_names, model.coef_))
