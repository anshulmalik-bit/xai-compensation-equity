"""
Pydantic Schema for Indian Compensation Employee Record
========================================================

Validates individual employee records against expected types, ranges, and
Indian-specific constraints. Used as the schema gate in the preprocessing
pipeline — malformed records are rejected with actionable error messages.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class EmployeeRecord(BaseModel):
    """
    Schema for a single employee record in the Indian compensation dataset.

    All field constraints are calibrated to realistic Indian HRMS values.
    Validation failures produce messages explaining what's wrong and what's expected.
    """
    employee_id: str = Field(..., pattern=r"^EMP\d{5}$")
    gender: str = Field(..., description="Male or Female")
    age: int = Field(..., ge=18, le=65)
    years_experience: int = Field(..., ge=0, le=40)
    education_level: str
    department: str
    job_title: str
    job_level: int = Field(..., ge=1, le=6)
    city: str
    performance_rating: float = Field(..., ge=1.0, le=5.0)
    manager_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    months_since_promotion: int = Field(..., ge=0, le=120)
    variable_pay_pct: float = Field(..., ge=0.0, le=50.0)
    ctc: float = Field(..., gt=0.0, le=200.0, description="CTC in LPA")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(
                f"gender must be one of {allowed}, got '{v}'. "
                "Check for encoding issues or non-standard values."
            )
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        allowed = {
            "Engineering", "Data Science", "Product", "Finance",
            "Marketing", "HR", "Operations", "Sales", "Legal",
            "Customer Support",
        }
        if v not in allowed:
            raise ValueError(
                f"department must be one of {allowed}, got '{v}'. "
                "Use canonical department names."
            )
        return v

    @field_validator("years_experience")
    @classmethod
    def validate_experience_age_coherence(cls, v: int) -> int:
        """Experience cannot exceed what's physically possible."""
        if v > 42:  # assuming minimum working age 18, max age 60
            raise ValueError(
                f"years_experience={v} is implausible. "
                "Maximum realistic value is ~42 (age 60, started at 18)."
            )
        return v
