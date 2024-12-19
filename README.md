# AutoML-Model
## AutoML Pipeline for Regression Models

This repository contains an AutoML pipeline designed to automate data preprocessing, feature engineering, feature reduction, model training, and evaluation for regression tasks. The pipeline allows for easy integration of various machine learning algorithms and performs hyperparameter tuning to optimize model performance.

## Overview:

The pipeline follows these key steps:

## Load Configuration and Data:

Loads configuration settings (hyperparameters, feature handling methods) from a JSON file.
Loads dataset from a CSV file.

## Feature Handling:

Handles missing values based on user-defined strategies (e.g., imputing with the mean or custom values).
Feature Generation:

Creates new features based on predefined interaction or polynomial relationships between existing features.
Feature Reduction:

Reduces the number of features using methods like tree-based feature importance, PCA (Principal Component Analysis), or correlation with the target.

## Model Preparation:

Prepares regression models based on user-configured algorithms (e.g., Random Forest, Decision Tree, Linear Regression, SVR).
Hyperparameters for each model are tuned using GridSearchCV for optimal performance.

## Model Evaluation:

Evaluates trained models on test data using metrics such as:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R-Squared (R²)

## Train/Test Split:

Splits data into training and testing sets to evaluate the performance of the models.

## Supported Algorithms:

Random Forest Regressor
Linear Regression
Decision Tree Regressor
Support Vector Regressor (SVR)

## Key Features:

Automatic Feature Engineering: Automatically generates interaction and polynomial features based on the configuration file.

Hyperparameter Tuning: Uses GridSearchCV for hyperparameter optimization, enabling selection of the best model configuration.

Flexible Configuration: The pipeline can be easily adapted to different datasets by modifying the configuration file and specifying different algorithms and hyperparameters.

Requirements:
Python 3.x

Libraries: pandas, numpy, scikit-learn, json

## Usage:

Configuration File: Provide a configuration JSON file (algoparams.json) that defines how features should be handled, which models to train, and their hyperparameters.

Dataset: Provide a dataset file (e.g., CSV) to train and test the models.

Run the Pipeline: Execute the pipeline by running the run_autopipeline() function with paths to the configuration file and dataset.
