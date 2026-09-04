import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
import joblib as jp

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    r2_score
)

# External file call for data visualization
from visualization import Visualization as visuals


class ChronicDiseasePipeline:

    def __init__(
        self,
        target="chronic_disease",
        test_size=0.2,
        random_state=42,
        threshold=0.35
    ):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.threshold = threshold

        self.log_cols = [
            "BMI",
            "MentHlth",
            "PhysHlth"
        ]

        self.iqr_cols = [
            "BMI",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "Age",
            "Education",
            "Income"
        ]

        self.model = None

    def clean_data(self, df):
        """
        Remove missing rows and duplicate rows without modifying
        the original DataFrame.
        """
        data = df.copy()

        data = data.dropna()
        data = data.drop_duplicates()

        return data

    def split_data(self, df):
        """
        Separate features from the target and create train/test sets.
        """
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def remove_training_outliers(
        self,
        X_train,
        y_train,
        multiplier=2.0
    ):
        """
        Calculate IQR limits using training data only and remove
        outlier rows from the training set.
        """
        available_iqr_cols = [
            column
            for column in self.iqr_cols
            if column in X_train.columns
        ]

        Q1 = X_train[available_iqr_cols].quantile(0.25)
        Q3 = X_train[available_iqr_cols].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outlier_mask = (
            (X_train[available_iqr_cols] < lower_bound) |
            (X_train[available_iqr_cols] > upper_bound)
        ).any(axis=1)

        keep_mask = ~outlier_mask

        X_train_clean = X_train.loc[keep_mask].copy()
        y_train_clean = y_train.loc[keep_mask].copy()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        return X_train_clean, y_train_clean

    def build_preprocessor(self, X_train):
        """
        Log-transform selected features, then standard-scale them.
        Standard-scale every other feature, including binary features.
        """
        log_cols = [
            column
            for column in self.log_cols
            if column in X_train.columns
        ]

        other_cols = [
            column
            for column in X_train.columns
            if column not in log_cols
        ]

        preprocessor = ColumnTransformer([
            (
                "log_then_scale",
                Pipeline([
                    (
                        "log_transform",
                        FunctionTransformer(
                            np.log1p,
                            feature_names_out="one-to-one"
                        )
                    ),
                    ("scaler", StandardScaler())
                ]),
                log_cols
            ),
            (
                "scale_all_other_features",
                StandardScaler(),
                other_cols
            )
        ])

        return preprocessor

    def build_model(self, X_train):
        """
        Build the complete preprocessing and classification pipeline.
        """
        preprocessor = self.build_preprocessor(X_train)

        model = Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    C=0.1,
                    penalty="l2",
                    max_iter=1000,
                    random_state=self.random_state
                )
            )
        ])

        return model

    def fit(self, df):
        """
        Run the complete training process.
        """
        # 1. Send the initial raw DataFrame to visuals module
        visuals.visuals(df,folder_name="preprocessing")

        data = self.clean_data(df)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = self.split_data(data)

        (
            self.X_train_clean,
            self.y_train_clean
        ) = self.remove_training_outliers(
            self.X_train,
            self.y_train
        )

        # 2. Reconstruct cleaned training DataFrame and send to visuals module
        cleaned_df = pd.concat([self.X_train_clean, self.y_train_clean], axis=1)
        visuals.visuals(cleaned_df,folder_name="processed")

        self.model = self.build_model(self.X_train_clean)

        self.model.fit(
            self.X_train_clean,
            self.y_train_clean
        )

        print("Original training rows:", len(self.X_train))
        print("Clean training rows:", len(self.X_train_clean))
        print("Removed training rows:", len(self.X_train) - len(self.X_train_clean))
        print("Test rows:", len(self.X_test))

        model_artifact = {
            "model": self.model,
            "threshold": self.threshold,
            "feature_columns": self.X_train_clean.columns.tolist(),
            "target": self.target
        }

        jp.dump(
            model_artifact,
            "chronic_disease_model.joblib"
        )

        return self

    def evaluate(self):
        """
        Evaluate the trained model on the untouched test set.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate().")

        y_probability = self.model.predict_proba(
            self.X_test
        )[:, 1]

        y_pred = (
            y_probability >= self.threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            self.y_test,
            y_pred
        ).ravel()

        specificity = tn / (tn + fp)

        results = {
            "r-2": r2_score(self.y_test, y_pred),
            "accuracy": accuracy_score(self.y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(
                self.y_test,
                y_pred
            ),
            "precision": precision_score(
                self.y_test,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                self.y_test,
                y_pred,
                zero_division=0
            ),
            "specificity": specificity,
            "f1_score": f1_score(
                self.y_test,
                y_pred,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                self.y_test,
                y_probability
            ),
            "pr_auc": average_precision_score(
                self.y_test,
                y_probability
            ),
            "log_loss": log_loss(
                self.y_test,
                y_probability
            ),
            "matthews_correlation": matthews_corrcoef(
                self.y_test,
                y_pred
            )
        }

        for name, value in results.items():
            print(f"{name}: {value:.4f}")

        print("\nConfusion matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        print("\nClassification report:")
        print(
            classification_report(
                self.y_test,
                y_pred,
                zero_division=0
            )
        )

        return results

    def predict(self, new_data):
        """
        Predict the class for new raw input data.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        probabilities = self.model.predict_proba(new_data)[:, 1]

        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        return predictions

    def predict_probability(self, new_data):
        """
        Return the probability of chronic_disease = 1.
        """
        if self.model is None:
            raise RuntimeError(
                "Call fit() before predict_probability()."
            )

        return self.model.predict_proba(new_data)[:, 1]