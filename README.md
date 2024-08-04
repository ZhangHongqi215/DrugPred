
# DrugPred

DrugPred: An Ensemble Learning Model Based on ESM2 for Predicting Potential Druggable Proteins

## Introduction

DrugPred is an ensemble learning model designed to predict potential druggable proteins using evolutionary scale modeling (ESM2) and amino acid composition (AAC) as features. The model integrates these features into a multidimensional and diverse feature space, enabling high accuracy in predicting drug targets. This repository contains the code and resources to run the DrugPred model, along with explanations and guidance on how to use the IPython notebook provided.

## Features

- **ESM2 Feature Extraction**: Uses deep learning to study the sequence-structure-function relationship of protein sequences.
- **AAC Feature Extraction**: Translates protein sequences into amino acid percentages.
- **Ensemble Learning**: Combines multiple machine learning algorithms to improve prediction accuracy and stability.
- **Model Interpretability**: Utilizes t-SNE and SHAP techniques to explain model predictions.

## Installation

To run the code in this repository, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Usage

The main code for the DrugPred model is contained in the `DrugPred.ipynb` notebook. Here is a step-by-step guide to using the notebook:

### Step 1: Load the Data

Load the dataset containing protein sequences and their labels (drug target or not). The dataset should be formatted as described in the notebook.

### Step 2: Feature Extraction

Run the cells responsible for extracting features from the protein sequences using ESM2 and AAC methods. The notebook includes detailed comments explaining each step.


### Step 3: Model Training

Train the ensemble learning model using the extracted features. The notebook demonstrates how to train individual models (SVM, NN, CatBoost, XGBoost) and then combine their predictions using an SVM-based voting mechanism.


### Step 4: Model Evaluation

Evaluate the performance of the model using various metrics such as accuracy, precision, recall, F1 score, and MCC. The notebook provides code for generating ROC and PRC curves as well.


### Step 5: Model Interpretability

Use t-SNE and SHAP to visualize and interpret the model's predictions. The notebook includes examples of how to apply these techniques.


## Acknowledgements

This work was supported by the National Natural Science Foundation of China and the Sichuan Province Postdoctoral Research Project Special Support Foundation.

