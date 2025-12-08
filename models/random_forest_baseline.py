"""
Random Forest baseline for active learning comparison.

This model uses ensemble variance as a proxy for uncertainty, which is known
to be poorly calibrated (chases noise rather than epistemic uncertainty).
This serves as a baseline to demonstrate the superiority of Neural Process
calibrated uncertainty for active learning.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional
import pandas as pd


class RandomForestBaseline:
    """
    Random Forest baseline with ensemble variance as uncertainty.

    This baseline demonstrates the failure mode of using uncalibrated
    uncertainty for active learning:
    - Ensemble variance captures aleatoric noise, not epistemic uncertainty
    - Model gets stuck chasing high-variance noise in the data
    - Learning curve plateaus or declines as poor samples are selected
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest baseline.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth (None = unlimited)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=True  # Enable for variance estimation
        )

        self.is_fitted = False
        self.target_mean = 0.0
        self.target_std = 1.0

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from dataframe.

        Args:
            df: DataFrame with 'embedding_patch', 'longitude', 'latitude'

        Returns:
            Feature matrix (n_samples, n_features)
        """
        # Stack embeddings
        embeddings = np.stack(df['embedding_patch'].values)

        # Flatten if needed (patch_size x patch_size x channels -> flat)
        if embeddings.ndim > 2:
            n_samples = embeddings.shape[0]
            embeddings = embeddings.reshape(n_samples, -1)

        # Add spatial features (normalized coordinates)
        coords = df[['longitude', 'latitude']].values

        # Concatenate embedding + coordinates
        features = np.concatenate([embeddings, coords], axis=1)

        return features

    def fit(self, train_df: pd.DataFrame) -> dict:
        """
        Train the Random Forest on the training data.

        Args:
            train_df: Training DataFrame with 'embedding_patch', 'agbd', 'longitude', 'latitude'

        Returns:
            Training metrics dictionary
        """
        X_train = self._prepare_features(train_df)
        y_train = train_df['agbd'].values

        # Normalize targets (use log space like ANP)
        y_train_log = np.log1p(y_train)
        self.target_mean = y_train_log.mean()
        self.target_std = y_train_log.std()
        y_train_normalized = (y_train_log - self.target_mean) / self.target_std

        # Fit model
        self.rf.fit(X_train, y_train_normalized)
        self.is_fitted = True

        # Compute training metrics
        y_pred_normalized = self.rf.predict(X_train)
        y_pred_log = y_pred_normalized * self.target_std + self.target_mean
        y_pred = np.expm1(y_pred_log)

        # Metrics in original space
        mse = np.mean((y_train - y_pred) ** 2)
        mae = np.mean(np.abs(y_train - y_pred))

        # Metrics in log space
        mse_log = np.mean((y_train_log - y_pred_log) ** 2)
        rmse_log = np.sqrt(mse_log)

        return {
            'train_loss': mse_log,  # Use log-space MSE as loss
            'train_rmse': rmse_log,
            'train_mae': mae,
            'n_train': len(train_df)
        }

    def predict(
        self,
        test_df: pd.DataFrame,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict on test data with ensemble variance as uncertainty.

        Args:
            test_df: Test DataFrame
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            (predictions, uncertainties) or just predictions
            - predictions: AGBD predictions in original space (Mg/ha)
            - uncertainties: Ensemble variance (standard deviation across trees)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X_test = self._prepare_features(test_df)

        if return_uncertainty:
            # Get predictions from each tree
            tree_predictions = np.array([
                tree.predict(X_test) for tree in self.rf.estimators_
            ])  # Shape: (n_estimators, n_samples)

            # Mean and std across trees (in normalized log space)
            y_pred_normalized = tree_predictions.mean(axis=0)
            y_std_normalized = tree_predictions.std(axis=0)  # Ensemble variance

            # Convert to original space
            y_pred_log = y_pred_normalized * self.target_std + self.target_mean
            y_pred = np.expm1(y_pred_log)

            # Uncertainty: std in log space (this is the "bad" uncertainty)
            uncertainties = y_std_normalized * self.target_std

            return y_pred, uncertainties
        else:
            y_pred_normalized = self.rf.predict(X_test)
            y_pred_log = y_pred_normalized * self.target_std + self.target_mean
            y_pred = np.expm1(y_pred_log)
            return y_pred, None

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluate on test data.

        Args:
            test_df: Test DataFrame with 'agbd' ground truth

        Returns:
            Dictionary of test metrics
        """
        y_true = test_df['agbd'].values
        y_true_log = np.log1p(y_true)

        y_pred, _ = self.predict(test_df, return_uncertainty=False)
        y_pred_log = np.log1p(y_pred)

        # Metrics in log space (for fair comparison with ANP)
        mse_log = np.mean((y_true_log - y_pred_log) ** 2)
        rmse_log = np.sqrt(mse_log)
        mae_log = np.mean(np.abs(y_true_log - y_pred_log))

        # R² in log space
        ss_res = np.sum((y_true_log - y_pred_log) ** 2)
        ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
        r2_log = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'test_rmse': rmse_log,
            'test_mae': mae_log,
            'test_r2': r2_log
        }
