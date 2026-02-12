# AI Pricing Intelligence Framework

## Overview

This project is a modular, multi-entity pricing elasticity modeling framework designed to evaluate competitive pricing impact at site level.

It supports:

- Multi-entity modeling
- Automated model selection (Linear, Random Forest, XGBoost)
- Synthetic data generation
- Batch result output
- Scalable structure for future MLOps integration

---

## Project Structure

Data_Preparation_Pipeline.py
Pricing_Model.py
Model_Evaluation_Selection.py
Elasticity.py
run_pipeline.py
generate_synthetic_data.py
sample_data.csv
requirements.txt


---

## Key Features

- Entity-level pricing elasticity modeling
- Automated best-model selection using R² & MAPE
- Synthetic dataset generator for reproducibility
- Batch output file generation
- Production-ready modular structure

---

## How to Run

1. Generate synthetic data: python generate_synthetic_data.py
2. Run modeling pipeline:python run_pipeline.py

Results are saved as:Model_Results.csv


---

## Version

v1.0 – Multi-Entity Pricing Elasticity Framework

---




