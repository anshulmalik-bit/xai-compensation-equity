"""
Variance Inflation Factor (VIF) Checker
========================================

Computes VIF for all numeric features to detect multicollinearity before
modelling. This matters because while LightGBM (tree-based) handles moderate
collinearity well, the Oaxaca-Blinder decomposition in the audit layer
uses OLS-like assumptions where high VIF distorts coefficient estimates.

Threshold logic:
    - VIF 1–5:   Low collinearity — no action needed
    - VIF 5–10:  Moderate — LOGGED but not flagged (LightGBM handles this)
    - VIF > 10:  High — raises ValueError (will distort OB decomposition)

The threshold of 10 (not 5) is used because the primary model is tree-based.
The stricter threshold is only needed for the audit layer's OLS components.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class VIFReport:
    """VIF computation results for all numeric features."""
    vif_table: pd.DataFrame
    high_vif_features: list
    moderate_vif_features: list


def check_vif(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
    error_threshold: float = 10.0,
    log_threshold: float = 5.0,
) -> VIFReport:
    """
    Compute VIF for all numeric features and enforce multicollinearity limits.

    Methodology: VIF measures how much the variance of a regression coefficient
    is inflated due to collinearity with other features. VIF_j = 1/(1 - R²_j)
    where R²_j is the R² from regressing feature j on all other features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numeric columns will be auto-selected)
    exclude_columns : list[str], optional
        Columns to exclude from VIF check (e.g., target variable, IDs)
    error_threshold : float
        VIF above this raises ValueError (default: 10.0)
    log_threshold : float
        VIF above this is logged as info (default: 5.0)

    Returns
    -------
    VIFReport
        Contains VIF table, lists of high and moderate VIF features

    Raises
    ------
    ValueError
        If any feature has VIF > error_threshold
    """
    if exclude_columns is None:
        exclude_columns = []

    # Select only numeric columns, excluding specified ones
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_columns
    ]

    if len(numeric_cols) < 2:
        print("VIF check: fewer than 2 numeric features -- skipping")
        return VIFReport(
            vif_table=pd.DataFrame(),
            high_vif_features=[],
            moderate_vif_features=[],
        )

    # Drop rows with NaN for VIF computation
    df_clean = df[numeric_cols].dropna()

    # IMPORTANT: Run VIF on the RAW feature matrix WITHOUT adding a constant term.
    # The default statsmodels pattern adds a constant which artificially suppresses
    # VIF values, causing highly correlated features (like job_level and years_experience
    # which have a raw VIF over 16) to slip under the 10.0 threshold.
    # We use the raw matrix directly to expose true collinearity.

    vif_data = []
    for i, col in enumerate(numeric_cols):
        vif_value = variance_inflation_factor(
            df_clean.values, df_clean.columns.get_loc(col)
        )
        vif_data.append({"feature": col, "vif": round(vif_value, 2)})

    vif_table = pd.DataFrame(vif_data).sort_values("vif", ascending=False)

    high_vif = vif_table[vif_table["vif"] > error_threshold]["feature"].tolist()
    moderate_vif = vif_table[
        (vif_table["vif"] > log_threshold) & (vif_table["vif"] <= error_threshold)
    ]["feature"].tolist()

    # Report
    print(f"VIF check ({len(numeric_cols)} features):")
    for _, row in vif_table.iterrows():
        marker = ""
        if row["vif"] > error_threshold:
            marker = " [FAIL] EXCEEDS THRESHOLD"
        elif row["vif"] > log_threshold:
            marker = " [WARN] moderate (logged)"
        print(f"  {row['feature']}: {row['vif']}{marker}")

    if moderate_vif:
        print(f"\n  [INFO] Moderate VIF (5-10, logged only): {moderate_vif}")
        print("     LightGBM handles this level of collinearity. No action needed.")

    if high_vif:
        error_msg = (
            f"Features with VIF > {error_threshold}: {high_vif}. "
            "This level of multicollinearity will distort the Oaxaca-Blinder "
            "decomposition coefficients. Drop or combine these features "
            "before modelling."
        )
        for feat in high_vif:
            feat_vif = vif_table[vif_table["feature"] == feat]["vif"].values[0]
            error_msg += f"\n  - {feat}: VIF={feat_vif} -- exceeds threshold of {error_threshold}."
        raise ValueError(error_msg)

    return VIFReport(
        vif_table=vif_table,
        high_vif_features=high_vif,
        moderate_vif_features=moderate_vif,
    )
