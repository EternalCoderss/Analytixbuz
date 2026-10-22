"""
Step 6: Automatic Feature Engineering.
Datetime extraction, derived ratios, interactions, and stateful preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schemas.models import EngineeredFeatureInfo, FeatureEngineeringResult


class FeatureEngineer:
    """
    Stateful feature engineering transformer.
    Encapsulates all preprocessing so single-row What-If perturbations
    execute identically to training data.
    """

    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.engineered_info: list[EngineeredFeatureInfo] = []
        self._original_feature_count: int = 0
        self._target_col: Optional[str] = None
        self._id_columns: list[str] = []
        self._datetime_columns: list[str] = []
        self._top_corr_features: list[str] = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame,
                      target_col: Optional[str] = None,
                      id_columns: Optional[list[str]] = None,
                      datetime_columns: Optional[list[str]] = None,
                      task_type: str = "Classification") -> tuple[np.ndarray, np.ndarray, FeatureEngineeringResult]:
        """
        Engineer features and fit the preprocessing pipeline.
        Returns (X_transformed, y, result).
        """
        df = df.copy()
        self._target_col = target_col
        self._id_columns = id_columns or []
        self._datetime_columns = datetime_columns or []
        self.engineered_info = []

        # Count original features (excluding target and IDs)
        exclude = set(self._id_columns)
        if target_col:
            exclude.add(target_col)
        self._original_feature_count = len([c for c in df.columns if c not in exclude])

        # --- Datetime Engineering ---
        df = self._engineer_datetime(df)

        # --- Drop ID and datetime columns ---
        drop_cols = [c for c in self._id_columns + self._datetime_columns if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # --- Extract target ---
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].values
            df = df.drop(columns=[target_col])

        # --- Derived Ratios & Interactions ---
        if target_col and y is not None:
            df = self._engineer_interactions(df, y)

        # --- Identify column types ---
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # --- Build preprocessing pipeline ---
        transformers = []
        if self.numeric_features:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", numeric_pipeline, self.numeric_features))

        if self.categorical_features:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        if not transformers:
            # Edge case: no features at all
            X = np.zeros((len(df), 1))
            self.feature_names_out = ["placeholder"]
            self._fitted = True
            result = self._build_result(df)
            return X, y if y is not None else np.array([]), result

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        X = self.preprocessor.fit_transform(df)
        self.feature_names_out = self._get_feature_names()
        self._fitted = True

        result = self._build_result(df)
        return X, y if y is not None else np.array([]), result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted pipeline."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer has not been fitted yet.")

        df = df.copy()

        # Apply same datetime engineering
        df = self._engineer_datetime(df)

        # Drop columns
        drop_cols = [c for c in self._id_columns + self._datetime_columns if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Drop target
        if self._target_col and self._target_col in df.columns:
            df = df.drop(columns=[self._target_col])

        # Apply same interactions
        df = self._apply_interactions(df)

        # Ensure all expected columns exist, fill missing with NaN
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = np.nan
        for col in self.categorical_features:
            if col not in df.columns:
                df[col] = "Missing"

        return self.preprocessor.transform(df)

    def _engineer_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from datetime columns."""
        for col in self._datetime_columns:
            if col not in df.columns:
                continue
            try:
                dt_series = pd.to_datetime(df[col], errors="coerce")
                prefix = col
                df[f"{prefix}_month"] = dt_series.dt.month.fillna(0).astype(int)
                df[f"{prefix}_weekday"] = dt_series.dt.weekday.fillna(0).astype(int)
                df[f"{prefix}_quarter"] = dt_series.dt.quarter.fillna(0).astype(int)
                df[f"{prefix}_day_of_year"] = dt_series.dt.dayofyear.fillna(0).astype(int)
                df[f"{prefix}_is_weekend"] = (dt_series.dt.weekday >= 5).astype(int)

                for feat in ["month", "weekday", "quarter", "day_of_year", "is_weekend"]:
                    self.engineered_info.append(EngineeredFeatureInfo(
                        name=f"{prefix}_{feat}",
                        source="datetime_extraction",
                        description=f"Extracted {feat} from '{col}'",
                    ))
            except Exception:
                pass
        return df

    def _engineer_interactions(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Create ratio and interaction features from top correlated features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            self._top_corr_features = []
            return df

        # Find top 3 features most correlated with target
        correlations = []
        y_series = pd.Series(y)
        for col in numeric_cols:
            try:
                corr = df[col].corr(y_series)
                if not np.isnan(corr):
                    correlations.append((abs(corr), col))
            except Exception:
                pass

        correlations.sort(reverse=True)
        top_features = [c for _, c in correlations[:3]]
        self._top_corr_features = top_features

        # Create pairwise ratios and interactions
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                fa, fb = top_features[i], top_features[j]
                ratio_name = f"{fa}_div_{fb}"
                product_name = f"{fa}_x_{fb}"

                if ratio_name not in df.columns:
                    df[ratio_name] = df[fa] / (df[fb].abs() + 1e-5)
                    self.engineered_info.append(EngineeredFeatureInfo(
                        name=ratio_name, source="ratio",
                        description=f"Ratio of '{fa}' / |'{fb}'| + ε",
                    ))

                if product_name not in df.columns:
                    df[product_name] = df[fa] * df[fb]
                    self.engineered_info.append(EngineeredFeatureInfo(
                        name=product_name, source="interaction",
                        description=f"Interaction: '{fa}' × '{fb}'",
                    ))

        return df

    def _apply_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same interaction features to new data (for transform)."""
        top_features = [f for f in self._top_corr_features if f in df.columns]

        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                fa, fb = top_features[i], top_features[j]
                ratio_name = f"{fa}_div_{fb}"
                product_name = f"{fa}_x_{fb}"

                if ratio_name not in df.columns:
                    df[ratio_name] = df[fa] / (df[fb].abs() + 1e-5)
                if product_name not in df.columns:
                    df[product_name] = df[fa] * df[fb]

        return df

    def _get_feature_names(self) -> list[str]:
        """Extract feature names from the fitted preprocessor."""
        names = []
        if self.preprocessor is None:
            return names
        try:
            names = list(self.preprocessor.get_feature_names_out())
        except Exception:
            # Fallback: construct names manually
            for name, trans, cols in self.preprocessor.transformers_:
                if name == "remainder":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    names.extend(trans.get_feature_names_out())
                else:
                    names.extend(cols)
        # Clean sklearn prefixes
        cleaned = []
        for n in names:
            for prefix in ["num__", "cat__"]:
                if n.startswith(prefix):
                    n = n[len(prefix):]
                    break
            cleaned.append(n)
        return cleaned

    def _build_result(self, df: pd.DataFrame) -> FeatureEngineeringResult:
        """Build the feature engineering result."""
        total = len(self.feature_names_out)
        engineered_count = len(self.engineered_info)
        return FeatureEngineeringResult(
            original_feature_count=self._original_feature_count,
            engineered_feature_count=engineered_count,
            total_feature_count=total,
            engineered_features=self.engineered_info,
            feature_names=self.feature_names_out,
        )
