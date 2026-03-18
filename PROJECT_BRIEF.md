# XAI Compensation Equity — Project Brief

## Business Problem
Indian companies face increasing regulatory pressure (Equal Remuneration Act, Digital Personal Data Protection Act 2023, MeitY AI guidelines) to demonstrate that their compensation practices do not discriminate by gender, caste-proxy (education tier), or geography. Most HR analytics teams lack the tooling to produce auditable, explainable evidence of pay equity — or inequity.

## What This Project Builds
A **production-ready compensation equity auditing system** that:
1. Predicts salary bands using quantile regression (P10/P50/P90) on Indian HRMS-style data
2. Explains individual predictions via SHAP (Track A — individual-level)
3. Decomposes group-level pay gaps via SHAP-adjusted Oaxaca-Blinder (Track B — structural bias)
4. Auto-generates a compliance PDF mapped to Indian regulatory frameworks
5. Deploys as a 3-page Streamlit app on Community Cloud

## Target Audience
- MBA dual-specialisation portfolio (Finance & Business Analytics)
- Technical interviews at Deloitte, EY, KPMG, GCC People Analytics teams
- The project must survive a "show me the live app and explain every number" interview format

## Technology Stack
- **Data**: Python, Pandas, Pydantic, RapidFuzz
- **Modelling**: LightGBM (quantile regression), Optuna, scikit-learn
- **XAI**: SHAP (TreeExplainer), statsmodels (Oaxaca-Blinder)
- **Deployment**: Streamlit, ReportLab (PDF), GitHub, Streamlit Community Cloud
- **Synthetic Data**: Calibrated to Keka 2025-26 Indian compensation benchmarks

## Key Metrics
| Metric | Target |
|---|---|
| R² (P50 model) | > 0.86 |
| RMSE | < 1.5 LPA |
| Compa-ratio accuracy | > 80% |
| Gender correlation with log(salary) | -0.20 to -0.35 (after controls) |
| Unexplained gap p-value | < 0.05 |

## Project Positioning
- **IS**: A production-ready auditing architecture; data source is the only thing that changes for real deployment
- **IS NOT**: A replacement for enterprise tools (Mercer/Aon), proof of discrimination, or an academic paper
