
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

```python
# Example code for feature extraction using ESM2
from esm import FastaBatchedDataset, pretrained

model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
batch_converter = alphabet.get_batch_converter()

data = FastaBatchedDataset.from_file("path_to_fasta_file")
batch_labels, batch_strs, batch_tokens = batch_converter(data)
results = model(batch_tokens, repr_layers=[33], return_contacts=False)
```

### Step 3: Model Training

Train the ensemble learning model using the extracted features. The notebook demonstrates how to train individual models (SVM, NN, CatBoost, XGBoost) and then combine their predictions using an SVM-based voting mechanism.

```python
# Example code for training individual models
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# Train models
svm_model = SVC(probability=True).fit(X_train, y_train)
xgb_model = XGBClassifier().fit(X_train, y_train)
catboost_model = CatBoostClassifier().fit(X_train, y_train)
nn_model = MLPClassifier().fit(X_train, y_train)
```

### Step 4: Model Evaluation

Evaluate the performance of the model using various metrics such as accuracy, precision, recall, F1 score, and MCC. The notebook provides code for generating ROC and PRC curves as well.

```python
# Example code for model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"MCC: {mcc}")
```

### Step 5: Model Interpretability

Use t-SNE and SHAP to visualize and interpret the model's predictions. The notebook includes examples of how to apply these techniques.

```python
# Example code for t-SNE visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_test)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test)
plt.title("t-SNE visualization")
plt.show()
```

## Authors

- **Hong-Qi Zhang** - Model design, training, and project management
- **Shang-Hua Liu** - Background research, data organization, and model validation
- **Jun-Wen Yu** - Model validation
- **Rui Li** - Figure preparation
- **Dong-Xin Ye** - Validation and software
- **Yan-Ting Jin** - Supervision and writing
- **Cheng-Bing Huang** - Supervision and writing
- **Ke-Jun Deng** - Funding acquisition, supervision, and writing

## Acknowledgements

This work was supported by the National Natural Science Foundation of China and the Sichuan Province Postdoctoral Research Project Special Support Foundation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
