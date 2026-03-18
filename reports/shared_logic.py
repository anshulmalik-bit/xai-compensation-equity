import pandas as pd
import numpy as np

def compute_audit_results():
    """
    Computes all final business, compliance, and gap metrics required for
    the PDF and Streamlit dashboard to maintain perfect state reconciliation.
    """
    df = pd.read_csv("data/shap_results.csv")
    
    # 1. Basic Distributions
    males = df[df["gender"] == "Male"]
    females = df[df["gender"] == "Female"]
    
    mean_obs_m = males["actual_log_salary"].mean()
    mean_obs_f = females["actual_log_salary"].mean()
    total_gap_log = mean_obs_m - mean_obs_f
    
    # 2. Oaxaca-Blinder SHAP-adjusted Gap
    shap_gender_diff = males["shap_gender_effect"].mean() - females["shap_gender_effect"].mean()
    residual_diff = males["residual_log_salary"].mean() - females["residual_log_salary"].mean()
    
    unexplained_gap_log = shap_gender_diff + residual_diff
    explained_gap_log = total_gap_log - unexplained_gap_log
    
    total_pct_gap = (np.exp(total_gap_log) - 1) * 100
    explained_pct = (np.exp(explained_gap_log) - 1) * 100
    unexplained_pct = (np.exp(unexplained_gap_log) - 1) * 100
    
    female_avg_raw = np.exp(females["actual_log_salary"]).mean()
    unexplained_gap_inr = (np.exp(unexplained_gap_log) - 1) * female_avg_raw * 100000
    
    # 3. Manager Rating Bias Vector
    manager_diff_points = females["manager_rating"].mean() - males["manager_rating"].mean()
    manager_shap_diff = males["shap_manager_effect"].mean() - females["shap_manager_effect"].mean()
    manager_inr_penalty = (np.exp(manager_shap_diff) - 1) * female_avg_raw
    manager_inr_annual = manager_inr_penalty * len(females) * 5 * 100000 # LPA to INR
    
    # 4. Remediation Strategy / ER Act Liability
    df["predicted_p50_raw"] = np.exp(df["predicted_log_salary"])
    df["actual_raw"] = np.exp(df["actual_log_salary"])
    
    male_stats = df[df["gender"] == "Male"].groupby("job_level")["actual_raw"].agg(["median", "std"]).to_dict('index')
    
    affected_gaps = []
    
    for _, row in df[df["gender"] == "Female"].iterrows():
        jl = row["job_level"]
        if jl in male_stats:
            m_med = male_stats[jl]["median"]
            m_std = male_stats[jl]["std"]
            threshold = m_med - m_std
            if row["predicted_p50_raw"] < threshold:
                affected_gaps.append(m_med - row["actual_raw"])
                
    affected_gaps = np.array(affected_gaps)
    affected_count = len(affected_gaps)
    remediation_cost_lpa = np.sum(affected_gaps)
    
    # Bootstrap CI for remediation cost (1000 iterations)
    np.random.seed(42)
    boot_sums = []
    if affected_count > 0:
        for _ in range(1000):
            sample = np.random.choice(affected_gaps, size=affected_count, replace=True)
            boot_sums.append(np.sum(sample))
        ci_lower_lpa = np.percentile(boot_sums, 2.5)
        ci_upper_lpa = np.percentile(boot_sums, 97.5)
    else:
        ci_lower_lpa, ci_upper_lpa = 0.0, 0.0
    
    return {
        "unexplained_pct": unexplained_pct,
        "explained_pct": explained_pct,
        "total_pct": total_pct_gap,
        "unexplained_gap_inr": unexplained_gap_inr,
        "manager_diff": manager_diff_points,
        "manager_inr_annual": manager_inr_annual,
        "affected_employees": affected_count * 5,
        "remediation_cost_inr": remediation_cost_lpa * 5 * 100000,
        "remediation_lower_inr": ci_lower_lpa * 5 * 100000,
        "remediation_upper_inr": ci_upper_lpa * 5 * 100000
    }
