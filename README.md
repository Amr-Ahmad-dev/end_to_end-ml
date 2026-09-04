# Chronic Disease Risk Prediction Module

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Status-Experimental-yellow?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <strong>An end-to-end data processing and machine learning pipeline for experimental chronic-disease risk classification.</strong>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/amrahmadsalah/">
    <img src="https://img.shields.io/badge/LinkedIn-Amr%20Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
</p>

---

## Objective

The objective of this project was to build a complete, reusable machine-learning workflow that starts with health-related data and ends with a directly usable prediction module.

Rather than focusing only on training a model, the project was designed around the complete pipeline:

**Data → Understanding → Visualization → Preprocessing → Feature Engineering → Model → Evaluation → Reusable Module → User Interface**

The project was also an experiment in understanding how preprocessing decisions, feature representation, class imbalance, and evaluation metrics affect a classification system.

> **Important:** This project is an experimental machine-learning project and is **not a medical diagnostic or clinical decision-making system**. The dataset, target construction, and model performance have significant limitations.

---

## What I Built

The project contains two layers.

### 1. Full Data & ML Pipeline

The complete pipeline allows the data to be:

* inspected and understood
* cleaned
* transformed
* visualized before preprocessing
* processed and visualized again
* used for feature engineering
* used to create/train the model
* evaluated using classification metrics
* stored for later use

The goal was to make these operations accessible through a single pipeline interface rather than scattering the workflow across notebooks and independent scripts.

### 2. Standalone Prediction Module

The final trained model is packaged as:

```text
disease/module.joblib
```

The standalone module can be used independently of the complete data-processing workflow.

This means an application does not need to reproduce the entire experimentation pipeline simply to make a prediction.

---

# Project Architecture

```text
                         ┌──────────────────────┐
                         │       Raw Data       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      data.py         │
                         │   Pipeline Interface │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          Data Cleaning      Visualization       Preprocessing
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Model Training     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Evaluation & Storage │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   module.joblib      │
                         │   Trained Model      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     module.py        │
                         │ Prediction Interface │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   User Application   │
                         └──────────────────────┘
```

---

# Repository Structure

```text
.
├── disease/
│   └── module.joblib
│
├── visualization.py
├── data.py
├── module.py
│
├── preprocessed.png
├── processed.png
│
└── README.md
```

## File Responsibilities

### `data.py`

The main development and experimentation interface.

This is where the pipeline is controlled.

It provides access to the workflow responsible for:

* cleaning the dataset
* preprocessing
* feature engineering
* model creation
* model evaluation
* visualization
* comparing the dataset before and after processing

The pipeline class in this file is the primary class to interact with when reproducing the complete workflow.

---

### `visualization.py`

Contains the visualization functionality used by the pipeline.

It is responsible for generating visual representations of the data before and after preprocessing.

The purpose is not simply to create plots, but to make the effect of preprocessing observable.

For example:

```text
             BEFORE PROCESSING
                    │
                    ▼
             ┌─────────────┐
             │ Raw Dataset │
             └──────┬──────┘
                    │
              Visualization
                    │
                    ▼
             ───────────────
                    │
                    ▼
             Data Processing
                    │
                    ▼
             ───────────────
                    │
                    ▼
              AFTER PROCESSING
```

The project includes representative before/after visualizations showing these transformations.

---

### `module.py`

The production-facing interaction layer for the trained model.

If you only want to use the finished chronic-disease model rather than reproduce the entire data-processing workflow, this is the file you need.

It loads:

```text
disease/module.joblib
```

and provides the interface required to interact with the trained model.

---

### `disease/module.joblib`

The serialized trained machine-learning model.

This is the core artifact of the finished module.

If you only want to integrate the trained model into another application, you do **not** need the complete experimentation pipeline.

You only need:

```text
module.py
disease/module.joblib
```

---

# Two Ways to Use the Project

## Full Pipeline

Use this if you want to reproduce or study the complete workflow.

```text
data.py
    +
visualization.py
    +
dataset
```

This gives access to the data-processing and visualization pipeline.

It is intended for experimentation, analysis, and understanding how the model was constructed.

---

## Standalone Model

Use this if you only want to make predictions.

```text
module.py
    +
disease/module.joblib
```

This is the minimal version required to interact with the finished model.

The distinction is intentional:

> **Development pipeline ≠ deployed prediction module**

The preprocessing and experimentation code is separated from the final model interface so that the trained model can be consumed independently.

---

# Data Processing Pipeline

The workflow follows a structured sequence:

```text
1. Data Acquisition
        ↓
2. Data Understanding
        ↓
3. Data Cleaning
        ↓
4. Data Visualization
        ↓
5. Feature Engineering
        ↓
6. Preprocessing
        ↓
7. Class-Imbalance Handling
        ↓
8. Model Training
        ↓
9. Model Evaluation
        ↓
10. Model Serialization
        ↓
11. Prediction Interface
```

A major purpose of the project was to make the transformation of the dataset observable rather than treating preprocessing as an invisible step before model training.

---

# Data Characteristics

The dataset contains a mixture of health-related variables, including binary and continuous features.

Several characteristics required consideration during preprocessing and modeling:

* different feature scales
* heterogeneous feature types
* overlapping feature distributions
* class imbalance
* relatively weak relationships between some features and the target
* correlations between variables
* experimentally constructed target representation

The data was therefore examined before preprocessing and again after processing.

---

# Class Imbalance

The target classes were not evenly distributed.

Because of this, accuracy alone was not considered sufficient for evaluating the classifier.

The training process therefore incorporated class balancing so that errors involving the less represented class were given greater importance.

The model was intentionally oriented toward **recall** rather than maximizing raw accuracy.

This reflects the experimental objective of reducing false negatives.

---

# Model Performance

The current experimental model produced approximately:

| Metric   |  Result |
| -------- | ------: |
| Accuracy | **67%** |
| Recall   | **88%** |

Recall was prioritized because the experiment placed greater importance on identifying positive cases than on minimizing every false positive.

However, these numbers should not be interpreted independently.

Model performance should be evaluated using the complete set of relevant classification metrics, including:

* precision
* recall
* F1-score
* specificity
* confusion matrix
* ROC-AUC / PR-AUC where appropriate

The reported performance is specific to this dataset, preprocessing procedure, target definition, model, and evaluation methodology.

It should not be interpreted as evidence of clinical predictive validity.

---

# Before & After Processing

The pipeline stores visualizations of the dataset before and after processing.

### Before Processing

<p align="center">
  <img src="preprocessed.png" alt="Data before processing" width="800">
</p>

### After Processing

<p align="center">
  <img src="processed.png" alt="Data after processing" width="800">
</p>

These visualizations are included to demonstrate the transformation performed by the preprocessing pipeline rather than simply reporting the final model score.

---

# Example Workflow

For the complete pipeline:

```python
from data import Pipeline

pipeline = Pipeline(...)

# Clean and process the data
pipeline.clean()

# Create/train the model
pipeline.create_model()

# Visualize the data
pipeline.visualize()
```

The exact constructor and method arguments depend on the implementation in `data.py`.

For direct interaction with the finished model:

```python
from module import ChronicDiseaseModule

model = ChronicDiseaseModule(...)

prediction = model.predict(...)
```

Refer to `module.py` for the exact interface.

---

# Design Philosophy

The project was built around a simple principle:

> **A machine-learning model is only one component of a larger data system.**

The workflow therefore separates:

```text
Data acquisition
      ↓
Data understanding
      ↓
Data processing
      ↓
Visualization
      ↓
Modeling
      ↓
Prediction
```

This makes it possible to inspect the data transformation process, experiment with preprocessing decisions, evaluate the model, and finally expose the trained model through a reusable interface.

The final interface is intentionally lightweight. Its purpose is to demonstrate that the trained module can be integrated into an application rather than to serve as a polished commercial frontend.

---

# What This Project Taught Me

This project reinforced several practical lessons about machine learning:

**1. Preprocessing does not create information.**

If the underlying features have weak relationships with the target, better preprocessing can improve representation but cannot manufacture predictive signal.

**2. Accuracy can be misleading with imbalanced classes.**

A classifier can achieve respectable accuracy while performing poorly on the class that matters.

**3. Feature engineering can change the problem representation.**

Combining related variables into a meaningful target representation can alter the learning problem and its statistical properties.

**4. Visualization should be part of the pipeline.**

Visualizing the data before and after processing makes preprocessing decisions easier to inspect and validate.

**5. A trained model should be separated from experimentation code.**

Once the model is trained, applications should be able to consume the resulting artifact without reproducing the entire development workflow.

---

# Limitations

This project has important limitations.

* It is an experimental machine-learning project.
* The dataset has limited predictive relationships with the constructed target.
* The classes are imbalanced.
* Model performance is dataset-specific.
* The target construction is experimental.
* The model has not undergone clinical validation.
* The model should not be used for medical diagnosis or treatment decisions.
* High recall does not imply clinical usefulness.
* Further validation on independent data would be required before making stronger claims about generalization.

These limitations are part of the experiment rather than something the project attempts to hide.

---

# Future Improvements

Potential improvements include:

* stronger validation methodology
* independent test data
* cross-validation
* systematic hyperparameter optimization
* threshold optimization
* calibration analysis
* additional imbalance-handling techniques
* feature-selection analysis
* comparison against multiple baseline models
* more rigorous leakage detection
* improved experiment tracking
* model explainability
* API deployment
* automated testing
* containerization

---

# Technologies

```text
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
```

Additional libraries may be required depending on the implementation of the current modules.

---

# Author

## Amr Ahmad

Computer Science student focused on building practical systems across:

**Python • Data Science • Machine Learning • Automation • Backend Development • AI**

<p align="center">
  <a href="YOUR_LINKEDIN_URL">
    <img src="https://img.shields.io/badge/Connect%20with%20me%20on%20LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="Connect with me on LinkedIn">
  </a>
</p>

---

<p align="center">
  <sub>Experimental project developed for learning, engineering practice, and exploration of end-to-end machine-learning workflows.</sub>
</p>
