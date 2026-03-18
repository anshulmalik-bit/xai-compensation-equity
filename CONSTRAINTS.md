# XAI Compensation Equity — Constraints & Critical Gates

## Architecture Constraints

### Data Layer
- Synthetic data must use **documented bias coefficients**: β_gender = -0.07 in the generation function
- Salary distribution must follow **log-normal** — use `log(CTC)` as the target variable throughout
- **3 engineered features**: `city_tier` (int 1–3), `education_tier` (int 1–4), `career_velocity` (float = job_level / years_experience)
- **1 null-handling feature**: `manager_rating_missing` (binary)
- **11 total model features**, each measuring a distinct construct
- **BANNED features** (redundant, do not create or include): `years_experience` (redundant with job_level/career_velocity), `is_metro`, `experience_band`, `industry_vertical`, `education_premium_flag`, `iit_iim_flag`
- Data must be calibrated to **Keka 2025-26** Indian compensation benchmarks

### Modelling Layer
- Three separate LightGBM models for **P10, P50, P90** quantile regression — not a single model
- Optuna for hyperparameter tuning — not grid search
- Target variable is **log(salary)**, never raw salary

### XAI/Audit Layer
- SHAP must use **TreeExplainer** (not KernelExplainer) for LightGBM
- Oaxaca-Blinder must run on **SHAP-adjusted residuals**, not on raw predictions or test targets

### Deployment Layer
- Streamlit app must be **3 pages exactly** (Salary Predictor, Pay Equity Auditor, Compliance Report)
- PDF and Streamlit figures must use the **same underlying computation** — no separate calculation paths

---

## 🔴 Critical User Supervision Gates

These gates require **human verification** and cannot be delegated. The agent must pause and request user review at each gate.

### Gate 1: Data Correlation Check (after Week 1 deliverable)
- **What to check**: Gender correlation with log(salary) after controlling for experience
- **Expected range**: -0.20 to -0.35
- **Why it matters**: If the signal is too weak, the entire audit layer produces a trivially small unexplained gap and the project's thesis collapses silently. If too strong, the model is unrealistic.
- **How to verify**: Run correlation matrix on generated dataset, check `corr(gender, log_salary)` with experience held constant

### Gate 2: Oaxaca-Blinder Input Verification (after Week 6 deliverable)
- **What to check**: Open `oaxaca_audit.py`, find the line where `OaxacaBlinder` is instantiated
- **Required**: `endog` parameter must be `shap_adjusted_salary`
- **Reject if**: `endog` is `y_pred`, `y_test`, `salary`, or any other variable
- **Why it matters**: Running OB on raw predictions bypasses the entire SHAP-adjustment methodology, making Track A and Track B disconnected

### Gate 3: Numbers Reconciliation (after Week 7 deliverable)
- **What to check**: Every figure in the generated PDF must exactly match what the Streamlit app displays
- **Why it matters**: A discrepancy means a state management bug — the PDF and Streamlit are computing from different data paths
- **How to verify**: Generate PDF, open Streamlit app, compare every numeric value side by side

---

## Code Quality Constraints
- All functions must have docstrings explaining the statistical methodology, not just what the code does
- VIF checker: threshold is **10.0** (not 5.0) — LightGBM handles moderate collinearity. VIF 5–10 logged only. VIF > 10 raises `ValueError`.
- Error messages must be actionable (e.g., "VIF for X is 12.3 — exceeds threshold of 10.0. Drop or combine this feature before modelling.")
- No silent failures — pipeline must raise clear errors on malformed input
- README must be business-first: no technical jargon until paragraph 3
