# XAI Compensation Equity Audit

Companies cannot fix pay disparities they cannot see. While most organizations measure "unadjusted" pay gaps by simply comparing average salaries between men and women, this basic math fails to separate structural biases from legitimate differences in seniority, performance, and role. When leadership lacks a precise, defensible method to isolate actual discrimination from valid compensation drivers, equity remediation becomes expensive guesswork, exposing the business to both talent attrition and regulatory liability.

This system provides a mathematically rigorous, auditable software layer that surgically splits an organization's compensation gap into "explained" (legitimate) and "unexplained" (bias) components. It ingests standard HR data, simulates a complex organizational structure, and detects specific biased pathways like subjective manager evaluations. The final output is an enterprise-grade compliance report and web dashboard that calculates the exact financial liability and lists the legally mandated remediation budget to correct the discrimination without causing reverse-bias.

The methodology resolves a longstanding mathematical conflict in HR analytics: traditional linear models (OLS) fail to understand complex human compensation (like performance multiplying with seniority), but modern non-linear AI accurately captures these dynamics while acting as a "black box" that regulators reject. This system solves the paradox by training a highly accurate non-linear model to capture the complex real-world data, then deploying Explainable AI (SHAP) to unpack the model's logic for every single employee. Finally, the system executes an econometric Oaxaca-Blinder decomposition purely on those isolated SHAP residuals, rigorously separating fair pay differences from structural bias.

The generated compliance artifacts directly map to the Indian regulatory landscape. The platform flags violations under the Equal Remuneration Act (1976), which mandates identical pay for identical work. By processing sensitive compensation information locally and structurally decoupling algorithmic explanations, it anticipates the privacy mandates of the Digital Personal Data Protection (DPDP) Act, 2023. Additionally, the explicit explainability layer adheres to the Ministry of Electronics and Information Technology (MeitY) guidelines on AI transparency and algorithmic fairness.

## Installation

```bash
git clone https://github.com/yourusername/xai-compensation-equity.git
cd xai-compensation-equity
pip install -r requirements.txt
```

## Usage

1. **Generate the Data**: `python data/generate_india_compensation_dataset.py`
2. **Train the Models**: `python models/train_models.py`
3. **Run the XAI Explainer**: `python models/explain.py`
4. **Execute the Audit**: `python audit/oaxaca_blinder.py`
5. **View the Dashboard/Report**: `streamlit run app.py`

## Dataset Description

The project relies on a synthetically engineered 10,000-employee dataset mirroring Indian tech-sector compensation structures. It includes canonical HR features such as job tier, education level, performance ratings, and geographic tier. The dataset is specifically structurally designed to contain complex non-linear compensation interactions and an explicitly injected direct wage penalty for female employees to test the detection bounds of the pipeline.

## Known Limitations

- **Synthetic Data Calibration**: This project uses highly engineered synthetic data rather than live organizational payrolls. The feature variance, correlations, and multiplier thresholds were intentionally manipulated to guarantee the audit scripts successfully catch structural boundary cases.
- **Intentional 7.7% Remediation Stress-Test**: The output remediation cost equates to approximately 7.7% of the total synthetic payroll. This is vastly higher than real-world equity corrections (which typically range from 0.5% to 2.0%) and is a deliberate mathematical consequence of injecting an extreme 28.4% direct pay penalty across 40% of the simulated workforce to stress-test the model detection bounds.
- **Not Legal Proof of Discrimination**: An "unexplained gap" identified by the statistical decomposition is a powerful indicator of liability risk, not absolute legal proof of explicit discrimination. Unmeasured legitimate variables (e.g., historical negotiation baselines, specialized technical certifications) may reside hidden within the unexplained variance.
