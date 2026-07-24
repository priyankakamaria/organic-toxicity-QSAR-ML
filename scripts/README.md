# Machine Learning-Based QSAR Prediction of Aquatic Toxicity of Organic Compounds

## Overview

This project develops an explainable QSAR framework for predicting aquatic toxicity using molecular descriptors, fingerprints and machine learning algorithms.

## Workflow

1. Dataset collection and curation
2. SMILES standardization
3. EC50 to pEC50 conversion
4. Molecular descriptor calculation
5. Feature filtering
6. Feature selection
7. Machine learning model development
8. Model validation
9. Applicability domain analysis
10. SHAP interpretation

## Molecular Features

- RDKit descriptors
- Morgan fingerprints (ECFP4/ECFP6)
- MACCS keys

## Machine Learning Models

- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- SVR

## Validation

- 5-fold cross-validation
- External test validation
- Y-randomization
- Williams plot
