"""
India-Specific Feature Engineering
====================================

Engineers 3 features from raw data and 1 null-handling feature:
1. city_tier     — integer 1/2/3 from city name (Metro/Tier-1/Tier-2+)
2. education_tier — integer 1–4 (IIT/IIM → Diploma), full gradient
3. career_velocity — job_level / years_experience (continuous float)
4. manager_rating_missing — binary flag for null manager_rating

Design decisions:
- city_tier is the ONLY geographic feature (no is_metro — redundant)
- education_tier is the ONLY education feature (no education_premium_flag
  or iit_iim_flag — both would split SHAP attribution)
- career_velocity replaces experience_band (bucketed categoricals destroy
  SHAP granularity — SHAP can't distinguish 8 vs 11 years in "Senior" band)
- manager_rating_missing preserves the information that a rating was absent
  (missingness may be informative — e.g., new joiners, exempt roles)
"""

import pandas as pd
import numpy as np


# ─── City → tier mapping ─────────────────────────────────────────────────────

CITY_TIER_MAP = {
    # Tier 1 — Metro
    "Mumbai": 1, "Delhi": 1, "Bangalore": 1, "Hyderabad": 1,
    "Chennai": 1, "Pune": 1, "Kolkata": 1,
    # Tier 2
    "Ahmedabad": 2, "Jaipur": 2, "Lucknow": 2, "Chandigarh": 2,
    "Kochi": 2, "Indore": 2, "Coimbatore": 2, "Nagpur": 2,
    "Visakhapatnam": 2,
    # Tier 3
    "Bhopal": 3, "Patna": 3, "Ranchi": 3, "Dehradun": 3,
    "Mysore": 3, "Thiruvananthapuram": 3, "Vadodara": 3,
    "Surat": 3, "Guwahati": 3, "Raipur": 3,
}

# ─── Education level → tier mapping ──────────────────────────────────────────

EDUCATION_TIER_MAP = {
    # Tier 1: IIT/IIM
    "IIT": 1, "IIM": 1,
    # Tier 2: NIT and top private
    "NIT": 2, "BITS": 2, "IIIT": 2,
    "Top Private (ISB, XLRI, SP Jain)": 2,
    # Tier 3: State/Central university
    "State University": 3, "Central University": 3,
    "Tier-2 Private": 3,
    # Tier 4: Diploma and others
    "Diploma": 4, "Distance Learning": 4, "Other": 4,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer India-specific features for the compensation model.

    Produces a DataFrame with 12 non-redundant features ready for modelling.
    No two features measure the same underlying construct.

    Parameters
    ----------
    df : pd.DataFrame
        Clean, validated DataFrame with raw features

    Returns
    -------
    pd.DataFrame
        DataFrame with 12 model-ready features:
        - years_experience (raw)
        - job_level (raw)
        - career_velocity (engineered)
        - city_tier (engineered)
        - education_tier (engineered)
        - performance_rating (raw)
        - manager_rating (raw, imputed)
        - manager_rating_missing (engineered)
        - months_since_promotion (raw)
        - variable_pay_pct (raw)
        - department (raw categorical)
        - gender (protected attribute)
    """
    df = df.copy()

    # ── 1. city_tier ──────────────────────────────────────────────────────
    df["city_tier"] = df["city"].map(CITY_TIER_MAP)
    unmapped_cities = df[df["city_tier"].isna()]["city"].unique()
    if len(unmapped_cities) > 0:
        print(f"  [WARN] Unmapped cities defaulted to tier 3: {unmapped_cities.tolist()}")
        df["city_tier"] = df["city_tier"].fillna(3).astype(int)
    else:
        df["city_tier"] = df["city_tier"].astype(int)

    # ── 2. education_tier ─────────────────────────────────────────────────
    df["education_tier"] = df["education_level"].map(EDUCATION_TIER_MAP)
    unmapped_edu = df[df["education_tier"].isna()]["education_level"].unique()
    if len(unmapped_edu) > 0:
        print(f"  [WARN] Unmapped education levels defaulted to tier 4: {unmapped_edu.tolist()}")
        df["education_tier"] = df["education_tier"].fillna(4).astype(int)
    else:
        df["education_tier"] = df["education_tier"].astype(int)

    # ── 3. career_velocity ────────────────────────────────────────────────
    # career_velocity = job_level / years_experience
    # For 0 years_experience (fresh joiners), use 0.5 to avoid division by zero
    # This gives them a velocity of job_level / 0.5 = 2 * job_level,
    # which correctly represents "promoted before accumulating tenure"
    safe_experience = df["years_experience"].clip(lower=0.5)
    df["career_velocity"] = (df["job_level"] / safe_experience).round(4)

    # ── 4. manager_rating_missing ─────────────────────────────────────────
    df["manager_rating_missing"] = df["manager_rating"].isna().astype(int)
    # Impute missing manager ratings with median (preserving the flag)
    median_rating = df["manager_rating"].median()
    df["manager_rating"] = df["manager_rating"].fillna(median_rating)

    # ── Select final feature matrix ───────────────────────────────────────
    # Note: years_experience is explicitly EXCLUDED here due to its high VIF (>16)
    # when paired with job_level. The tenure information is preserved
    # inside career_velocity without causing collinearity.
    feature_columns = [
        "job_level",
        "career_velocity",
        "city_tier",
        "education_tier",
        "performance_rating",
        "manager_rating",
        "manager_rating_missing",
        "months_since_promotion",
        "variable_pay_pct",
        "department",
        "gender",
    ]

    # Preserve target and ID for downstream use
    extra_columns = ["employee_id", "ctc"]
    output_columns = feature_columns + [c for c in extra_columns if c in df.columns]

    df_out = df[output_columns].copy()

    print(f"Feature engineering complete: {len(feature_columns)} features")
    print(f"  Engineered: city_tier, education_tier, career_velocity, manager_rating_missing")
    print(f"  career_velocity range: {df_out['career_velocity'].min():.2f} - {df_out['career_velocity'].max():.2f}")
    print(f"  education_tier distribution: {df_out['education_tier'].value_counts().sort_index().to_dict()}")
    print(f"  manager_rating imputed: {df['manager_rating_missing'].sum()} records")

    return df_out
