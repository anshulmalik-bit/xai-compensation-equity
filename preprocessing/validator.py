"""
Schema Validation Gate
=======================

Reads a CSV file, validates every row against the EmployeeRecord pydantic schema,
and separates valid from invalid records. Invalid records are reported with
actionable error messages so the data source can be corrected.

This is the first step in the preprocessing pipeline — no data passes downstream
without passing this gate.
"""

import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from data.schema import EmployeeRecord


@dataclass
class ValidationResult:
    """Result of schema validation containing valid data and error details."""
    dataframe: pd.DataFrame
    valid_count: int = 0
    invalid_count: int = 0
    errors: list = field(default_factory=list)


def validate_dataset(filepath: str | Path) -> ValidationResult:
    """
    Validate every row in the CSV against the EmployeeRecord schema.

    Methodology: Each row is independently validated. A single invalid field
    rejects the entire row (strict mode). This prevents partially valid
    records from corrupting downstream analysis — better to lose 1% of
    records than carry forward data quality issues.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file to validate

    Returns
    -------
    ValidationResult
        Contains the cleaned DataFrame (valid rows only), counts, and
        detailed error list for invalid rows

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}. "
            "Run generate_india_compensation_dataset.py first."
        )

    df = pd.read_csv(filepath)
    valid_rows = []
    errors = []

    for idx, row in df.iterrows():
        try:
            # Convert NaN to None for Optional fields (pydantic needs None, not NaN)
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            record = EmployeeRecord(**row_dict)
            valid_rows.append(row)
        except Exception as e:
            errors.append({
                "row_index": idx,
                "employee_id": row.get("employee_id", "UNKNOWN"),
                "error": str(e),
            })

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    result = ValidationResult(
        dataframe=valid_df,
        valid_count=len(valid_rows),
        invalid_count=len(errors),
        errors=errors,
    )

    print(f"Validation: {result.valid_count} valid, {result.invalid_count} invalid")
    if errors:
        print(f"  First 3 errors:")
        for err in errors[:3]:
            print(f"    Row {err['row_index']} ({err['employee_id']}): {err['error'][:120]}")

    return result
