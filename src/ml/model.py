
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import joblib

# Optional imports for ML - handle if not installed
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score,
        log_loss, brier_score_loss
    )
except (ImportError, Exception):
    # Catch ImportError (package missing) and other exceptions (e.g. XGBoostError due to missing libomp)
    xgb = None

class InsiderScoringModel:
    """
    XGBoost ensemble for insider probability scoring.

    Architecture:
    - Multiple XGBoost models with different hyperparameters
    - Calibrated probabilities using Platt scaling
    - Optional rule-based overrides for edge cases
    """

    def __init__(self):
        self.models = []
        self.calibrators = []
        self.feature_columns = None
        self.is_fitted = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_models: int = 5,
        test_size: float = 0.2,
        calibration_size: float = 0.1
    ) -> Dict[str, float]:
        """
        Train ensemble of XGBoost models.

        Returns evaluation metrics.
        """
        if not xgb:
            raise ImportError("XGBoost and scikit-learn are required for training.")

        self.feature_columns = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Further split for calibration
        # Ensure we have enough data for split
        if len(X_train) > 20:
             X_train_final, X_cal, y_train_final, y_cal = train_test_split(
                X_train, y_train, test_size=calibration_size,
                stratify=y_train, random_state=42
            )
        else:
             # Fallback for very small datasets
             X_train_final, X_cal, y_train_final, y_cal = X_train, X_train, y_train, y_train


        # Define hyperparameter variations
        param_variations = [
            {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100},
            {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200},
            {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 300},
            {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 150, "subsample": 0.8},
            {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200, "colsample_bytree": 0.8},
        ]

        # Train models
        self.models = []
        self.calibrators = []
        # Calculate scale_pos_weight for imbalance
        pos_count = y_train_final.sum()
        neg_count = len(y_train_final) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        for i, params in enumerate(param_variations[:n_models]):
            model = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                early_stopping_rounds=10,
                scale_pos_weight=scale_pos_weight,
                random_state=42 + i
            )
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            self.models.append(model)

            # Calibrate
            # Check if calibration data has sufficient classes
            if len(np.unique(y_cal)) < 2:
                 # Fallback: Skip calibration if one class missing in split
                 self.calibrators.append(model)
            else:
                try:
                    calibrator = CalibratedClassifierCV(
                        model, method="sigmoid", cv="prefit"
                    )
                    calibrator.fit(X_cal, y_cal)
                    self.calibrators.append(calibrator)
                except Exception:
                    # Fallback if calibration fails (e.g. convergence issues)
                    self.calibrators.append(model)

        self.is_fitted = True

        # Evaluate
        return self._evaluate(X_test, y_test)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Returns ensemble average of calibrated probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Ensure columns match
        X_subset = X[self.feature_columns]

        # Get predictions from all calibrators
        probas = np.zeros((len(X_subset), len(self.calibrators)))
        for i, calibrator in enumerate(self.calibrators):
            probas[:, i] = calibrator.predict_proba(X_subset)[:, 1]

        # Ensemble average
        return probas.mean(axis=1)

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance.
        """
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba > 0.5).astype(int)
        
        # Handle cases with single class in test set to avoid errors
        if len(np.unique(y_test)) < 2:
             return {"status": "insufficient_classes_for_metrics"}

        return {
            "auc_roc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "log_loss": log_loss(y_test, y_proba),
            "brier_score": brier_score_loss(y_test, y_proba),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance averaged across ensemble.
        """
        if not self.is_fitted:
             raise ValueError("Model not fitted")
             
        importances = np.zeros(len(self.feature_columns))
        valid_models = 0
        
        for model in self.models:
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_
                valid_models += 1

        if valid_models > 0:
            importances /= valid_models

        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances
        }).sort_values("importance", ascending=False)

    def save(self, path: Path):
        """Save model to disk."""
        joblib.dump({
            "models": self.models,
            "calibrators": self.calibrators,
            "feature_columns": self.feature_columns,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "InsiderScoringModel":
        """Load model from disk."""
        data = joblib.load(path)
        model = cls()
        model.models = data["models"]
        model.calibrators = data["calibrators"]
        model.feature_columns = data["feature_columns"]
        model.is_fitted = True
        return model
