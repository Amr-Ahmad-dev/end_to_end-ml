# Chronic Disease Risk Prediction

<p align="center">
  <strong>An end-to-end machine learning pipeline for health-data processing, visualization, classification, and reusable prediction.</strong>
</p>

<p align="center">
  <a href='https://www.linkedin.com/in/amrahmadsalah/'>
    <img src="https://img.shields.io/badge/LinkedIn-Amr%20Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href='https://youtu.be/1UKmyQ_MMMc'>
    <img src="https://img.shields.io/badge/LinkedIn-Amr%20Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="YouTube">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Experimental-yellow?style=for-the-badge" alt="Experimental">
</p>

---

## Objective

This project was built to explore the complete path from health-related tabular data to a reusable machine-learning prediction module.

The objective was not simply to train a classifier.

The project combines:

**Data Understanding → Visualization → Cleaning → Preprocessing → Feature Engineering → Class Balancing → Model Training → Evaluation → Model Serialization → Prediction Interface**

The resulting system separates the experimentation pipeline from the final prediction module, allowing the trained model to be used independently of the complete data-processing workflow.

> **Disclaimer:** This is an experimental machine-learning project for educational and engineering purposes. It is **not a medical diagnostic system**, and its predictions should not be used for medical decisions.

---

# Pipeline Overview

```text
                         RAW HEALTH DATA
                               │
                               ▼
                    ┌─────────────────────┐
                    │     data.py         │
                    │   Pipeline Class    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
          Inspection      Visualization      Cleaning
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                       Preprocessing
                               │
                               ▼
                     Feature Engineering
                               │
                               ▼
                     Class-Balanced Training
                               │
                               ▼
                         Evaluation
                               │
                               ▼
                     ┌─────────────────┐
                     │ module.joblib   │
                     │  Trained Model  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   module.py     │
                     │ Prediction API  │
                     └────────┬────────┘
                              │
                              ▼
                     Interactive Application
```

---

# What the Project Does

The dataset contains a mixture of binary and continuous health-related variables.

The pipeline was designed to examine and transform the data before it reached the model.

The workflow includes:

* dataset inspection
* feature-type handling
* distribution analysis
* correlation analysis
* data cleaning
* preprocessing
* feature scaling
* feature engineering
* class-imbalance handling
* visualization before and after processing
* model training
* classification evaluation
* model serialization
* reusable prediction

One of the main goals was to make preprocessing observable.

Instead of simply transforming the dataset and sending the result to a model, the pipeline generates visualizations before and after processing so the effect of the transformation can be examined directly.

---

# Repository Structure

```text
.
├── disease/
│   └── module.joblib
│
├── data.py
├── visualization.py
├── module.py
│
├── preprocessed.png
├── processed.png
│
└── README.md
```

### `data.py`

The main interface for the complete data-processing and machine-learning pipeline.

The pipeline class in this file is the primary interface for:

* cleaning data
* preprocessing data
* creating the model
* training the model
* evaluating the model
* visualizing data before processing
* visualizing data after processing

This is the file to use when reproducing or modifying the complete workflow.

---

### `visualization.py`

Contains the visualization functionality used by the pipeline.

It is responsible for examining the dataset visually and generating the before/after comparisons used during preprocessing.

The file is primarily consumed by the pipeline rather than being the main interface for users.

---

### `module.py`

The direct interface to the finished chronic-disease prediction module.

This is the file to use when the goal is simply to interact with the trained model rather than reproduce the complete data-processing workflow.

---

### `disease/module.joblib`

Serialized trained machine-learning model.

This is the actual model artifact consumed by `module.py`.

---

# Minimal Usage

If you only want to use the finished model, you only need:

```text
module.py
disease/module.joblib
```

The complete data-processing and visualization files are not required for basic model interaction.

Conceptually:

```text
module.py
     │
     ▼
module.joblib
     │
     ▼
prediction
```

This separation allows the final model to be integrated into another application without requiring the entire experimental pipeline.

---

# Full Pipeline vs. Standalone Module

There are two distinct ways to interact with the project.

### Full Pipeline

```text
data.py
   +
visualization.py
   +
dataset
```

Use this when you want to:

* inspect the dataset
* reproduce preprocessing
* visualize transformations
* modify the pipeline
* retrain the model
* experiment with the workflow

### Standalone Prediction

```text
module.py
   +
disease/module.joblib
```

Use this when you only want to:

* load the trained model
* provide a user's feature values
* obtain a prediction
* integrate the model into an application

This distinction is intentional.

**The development pipeline and the prediction interface are separate components.**

---

# Data Transformation

The original data was already substantially structured and contained no missing values.

```text
Dataset
253,680 rows
20 columns

Missing rows:
0

Missing values:
0%
```

The pipeline nevertheless performs additional processing because clean data does not necessarily mean model-ready data.

The dataset contained:

* binary variables
* continuous variables
* different feature scales
* overlapping feature distributions
* correlated variables
* class imbalance

Consequently, preprocessing focused on transforming the representation of the data rather than simply filling missing values.

---

# Training Data

The final run reported:

| Dataset                |    Rows |
| ---------------------- | ------: |
| Original training rows | 183,120 |
| Clean training rows    | 137,456 |
| Removed training rows  |  45,664 |
| Test rows              |  45,781 |

Approximately **24.9% of the original training rows were removed during the cleaning stage**.

The test set was kept separate from the training data.

> The README intentionally does not label the removed rows as "missing data" because the original dataset contained zero missing rows. The exact removal criteria should be documented in `data.py` if this repository is intended for reproducibility.

---

# Class Imbalance

The test set contained:

```text
Negative: 34,761
Positive: 11,020
```

This represents an imbalanced classification problem.

Because of this imbalance, accuracy was not treated as the only meaningful evaluation criterion.

The training process incorporated class weighting so that the minority/positive class received greater importance during optimization.

The decision was deliberately oriented toward **high recall**.

In other words:

> Missing a positive case was considered more costly than generating additional false positives.

This produced a high sensitivity model, but at the cost of substantially lower precision.

---

# Model Performance

Final evaluation on the held-out test set:

| Metric               |      Score |
| -------------------- | ---------: |
| Accuracy             | **61.89%** |
| Balanced Accuracy    | **71.90%** |
| Precision            | **37.89%** |
| Recall               | **91.21%** |
| Specificity          | **52.60%** |
| F1 Score             | **53.54%** |
| ROC-AUC              | **82.23%** |
| PR-AUC               | **57.68%** |
| Log Loss             | **0.5609** |
| Matthews Correlation | **0.3794** |

### Confusion Matrix

```text
                    Predicted
                  0          1

Actual 0       18,285     16,476
Actual 1          969     10,051
```

The classifier correctly identified:

**10,051 of 11,020 positive cases.**

That corresponds to approximately:

**91.21% recall.**

However, it also produced:

**16,476 false positives.**

This explains why the model has high recall but relatively low precision.

The result is therefore better described as a **high-sensitivity classifier** rather than a highly accurate classifier.

---

# Interpreting the Results

The model demonstrates a meaningful ability to separate the classes:

**ROC-AUC = 0.8223**

while the selected classification threshold produces:

**Recall = 91.21%**

at the cost of:

**Precision = 37.89%**

This illustrates an important machine-learning tradeoff.

Increasing sensitivity can substantially increase the number of false positives.

Therefore, the model's performance cannot be summarized responsibly by reporting only its 91% recall.

The appropriate interpretation depends on the intended cost of false negatives versus false positives.

---

# Before & After Visualization

The pipeline stores visualizations of the data before and after preprocessing.

### Before Processing

<p align="center">
  <img src="https://github.com/Amr-Ahmad-dev/end_to_end-ml/blob/main/preprocessing/feature_distributions_histograms.png" alt="Dataset before processing" width="850">
</p>

### After Processing

<p align="center">
  <img src="https://github.com/Amr-Ahmad-dev/end_to_end-ml/blob/main/processed/feature_distributions_histograms.png" alt="Dataset after processing" width="850">
</p>

The repository also contains the complete visualization workflow used to inspect the transformations.

---

# Key Engineering Idea

The main architectural decision was to avoid coupling the final prediction interface to the entire experimentation pipeline.

The development workflow is:

```text
Data
 ↓
Analysis
 ↓
Cleaning
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Training
 ↓
Evaluation
```

while the final application workflow is:

```text
User Input
 ↓
module.py
 ↓
module.joblib
 ↓
Prediction
```

This makes the final model easier to reuse.

The application does not need to know how the dataset was originally explored or how the training experiment was conducted.

---

# What I Learned

### Data quality is not the same as model readiness

A dataset can contain zero missing values and still require substantial preprocessing.

### Accuracy is not enough

With imbalanced classes, accuracy can hide poor minority-class performance.

This project made the tradeoff between precision and recall particularly clear.

### Preprocessing cannot manufacture predictive signal

If the underlying features have weak relationships with the target, transformations can improve representation but cannot create information that does not exist.

### Feature engineering changes the learning problem

Combining related variables into a target representation can alter the statistical structure of the classification problem and therefore needs to be treated as part of the modeling decision.

### Visualization is part of validation

The before/after visualization stage makes preprocessing effects inspectable instead of treating preprocessing as a black box.

### Reusable modules matter

The final `module.py + module.joblib` interface separates model consumption from model development.

---

# Limitations

This project should be interpreted as an engineering and machine-learning experiment.

Important limitations include:

* the dataset provides limited predictive information for the constructed target
* the classes are imbalanced
* the chosen threshold favors recall over precision
* the model produces a substantial number of false positives
* the target construction is experimental
* performance is specific to this dataset and evaluation procedure
* no clinical validation was performed
* no claim of clinical usefulness is being made
* the model should not be used for diagnosis, treatment, or medical decision-making

The reported metrics should therefore be understood as **experimental model-performance measurements**, not medical risk estimates.

---

# Future Work

Potential improvements include:

* cross-validation
* stronger baseline comparisons
* threshold optimization
* probability calibration
* independent external validation
* systematic hyperparameter optimization
* feature-selection analysis
* leakage auditing
* model explainability
* experiment tracking
* automated testing
* API deployment
* containerization

---

# Technologies

<p align="center">

<img src="https://img.shields.io/badge/Python-blue?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square">
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square">
<img src="https://img.shields.io/badge/Joblib-Model%20Serialization-green?style=flat-square">

</p>

---

# Author

## Amr Ahmad

Computer Science student interested in:

**Software Engineering · Python · Data Science · Machine Learning · AI · Automation**

<p align="center">
  <a href='https://www.linkedin.com/in/amrahmadsalah/'>
    <img src="https://img.shields.io/badge/Connect%20with%20me%20on%20LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
</p>

---

<p align="center">
  <sub>Built as an experimental project to understand and implement an end-to-end machine-learning workflow.</sub>
</p>
