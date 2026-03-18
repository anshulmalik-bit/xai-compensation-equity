"""
Job Title Standardiser
=======================

Uses fuzzy matching (rapidfuzz) to map raw job titles to canonical forms
from the canonical_job_titles.json lookup table. This prevents the model
from treating "Sr. Software Eng" and "Senior Software Engineer" as
different features.

Methodology: Token-set ratio matching with a configurable similarity
threshold (default 80). Unmatched titles are logged for human review
rather than silently dropped.
"""

import json
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz


def _load_canonical_titles() -> dict[str, str]:
    """Load the canonical job title mappings from JSON."""
    json_path = Path(__file__).parent.parent / "data" / "canonical_job_titles.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"canonical_job_titles.json not found at {json_path}. "
            "This file must exist before running the title standardiser."
        )
    with open(json_path, "r") as f:
        return json.load(f)


def standardise_titles(
    df: pd.DataFrame,
    column: str = "job_title",
    threshold: int = 80,
) -> pd.DataFrame:
    """
    Standardise job titles using fuzzy matching against canonical forms.

    For each unique title in the dataset, finds the best match in the
    canonical lookup table using token-set ratio. Titles with a match
    score >= threshold are replaced; titles below threshold are kept
    unchanged and logged as unmatched.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a job_title column
    column : str
        Name of the column to standardise (default: "job_title")
    threshold : int
        Minimum similarity score (0-100) for a match (default: 80)

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised job titles in the specified column.
        Original column is preserved as {column}_original.
    """
    canonical_map = _load_canonical_titles()
    variants = list(canonical_map.keys())

    df = df.copy()
    df[f"{column}_original"] = df[column]

    unique_titles = df[column].unique()
    title_mapping = {}
    unmatched = []

    for title in unique_titles:
        # First check exact match in canonical values (already canonical)
        if title in canonical_map.values():
            title_mapping[title] = title
            continue

        # Then check exact match in variants
        if title in canonical_map:
            title_mapping[title] = canonical_map[title]
            continue

        # Fuzzy match
        match = process.extractOne(
            title, variants, scorer=fuzz.token_set_ratio
        )
        if match and match[1] >= threshold:
            title_mapping[title] = canonical_map[match[0]]
        else:
            title_mapping[title] = title  # Keep original
            unmatched.append((title, match[1] if match else 0))

    df[column] = df[column].map(title_mapping)

    matched_count = len(unique_titles) - len(unmatched)
    print(f"Title standardisation: {matched_count}/{len(unique_titles)} unique titles matched")
    if unmatched:
        print(f"  Unmatched ({len(unmatched)}):")
        for title, score in unmatched[:5]:
            print(f"    '{title}' (best score: {score})")

    return df
