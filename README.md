# Bird Species Distribution Modeling: Predicting Habitat Shifts with AI

## Overview

This research project explores how climate change is reshaping bird migration patterns and habitats. By integrating satellite imagery and climate variables, we propose a hybrid AI pipeline to predict bird species presence across varying environments.  

We leverage:
- Convolutional Neural Networks (CNNs) implemented from scratch and in PyTorch for spatial feature extraction from satellite images.
- Tabular classifiers (Random Forest, Gradient Boosting) using our own implementation and scikit-learn, combined with our own custom preprocessing and feature engineering pipeline.
- Real-world ecological data, including satellite data and bird occurrence labels from the eBird database.

## Approach

We designed two parallel models:
1. CNN-based Vision Model (custom implementation compared with PyTorch):
 Uses satellite images to identify features like forestation, water bodies, and urbanization that influence bird migration patterns using ResNet models.


2. Tabular Feature Model (custom implementation compared with scikit-learn):
Uses structured environmental variables (e.g., temperature, elevation, latitude/longitude) to predict presence/absence using Random Forests and Gradient Boosting classifiers.

These models were evaluated individually and comparatively to study their effectiveness in modeling species distribution.

## Key Contributions

- A CNN trained on landscape imagery for species presence classification
- A tabular model using ecological features for probabilistic bird occurrence prediction
- A comparative evaluation showing strong predictive performance (~85% average accuracy)

## Repository Structure

The research project features the following components:

- `notebooks`: Directory containing the image classification and tabular models
- `images`: Directory containing images of species distribution on a map as well as feature importance

## Technologies Used

- Python 3
- PyTorch
- scikit-learn
- NumPy, Pandas, Matplotlib
- Sentinel, eBird

## Reproducibility

In order to use the repository you can clone it then request an eBird API key as that’s our primary data collection method. Set the API key as EBIRD_API_TOKEN then you can proceed to run both the CNN and the tabular model. 
