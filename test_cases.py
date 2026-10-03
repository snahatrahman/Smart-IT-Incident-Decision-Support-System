# -*- coding: utf-8 -*-
"""
Standalone Test-Case Demonstration:
Evaluates real-world incident test cases through the trained Decision Support Pipeline.
Usage:
    python test_cases.py
"""
import os
import joblib
import pandas as pd
import numpy as np
from src.config import OUTPUT_DIR, DISCOVERED_RULES_PATH, DECISION_DASHBOARD_PATH

def load_system():
    models_path = os.path.join(OUTPUT_DIR, "trained_models.joblib")
    if not os.path.exists(models_path):
        print(f"Error: {models_path} not found. Running run_pipeline.py first is required.")
        return None, None

    artifacts = joblib.load(models_path)
    rules_df = pd.read_csv(DISCOVERED_RULES_PATH) if os.path.exists(DISCOVERED_RULES_PATH) else pd.DataFrame()
    return artifacts, rules_df

def predict_single_ticket(ticket_dict, artifacts, rules_df):
    clf_breach = artifacts["clf_breach"]
    clf_route = artifacts["clf_route"]
    encoder = artifacts["encoder"]
    group_encoder = artifacts["group_encoder"]
    intake_features = artifacts["intake_features"]
    cat_cols = artifacts["INTAKE_CATEGORICAL"]

    df = pd.DataFrame([ticket_dict])

    # Ensure all required sparse indicator flags exist
    for sp in artifacts["SPARSE_COLS"]:
        flag = f"has_{sp}"
        if flag not in df.columns:
            df[flag] = 0

    # Ensure temporal features exist
    if "open_hour" not in df.columns:
        df["open_hour"] = 9
    if "open_dayofweek" not in df.columns:
        df["open_dayofweek"] = 1

    # Encode categorical features
    X_cat = encoder.transform(df[cat_cols].astype(str))
    X = df[intake_features].copy()
    X[cat_cols] = X_cat

    # Predict Breach Probability
    breach_prob = clf_breach.predict_proba(X)[0, 1]

    # Predict Recommended Specialist Group
    pred_grp_idx = clf_route.predict(X)[0]
    rec_group = group_encoder.inverse_transform([[pred_grp_idx]])[0, 0]

    # Determine Tier & Action
    if breach_prob >= 0.70:
        tier = "HIGH RISK"
        badge = "[CRITICAL ALERT]"
        action = f"URGENT: Bypass Level 1 Triage -> Dispatch Directly to {rec_group}"
    elif breach_prob >= 0.40:
        tier = "MEDIUM"
        badge = "[ELEVATED RISK]"
        action = f"MONITOR: Route to {rec_group} with proactive 24h milestone reminder"
    else:
        tier = "LOW RISK"
        badge = "[NORMAL WORKLOAD]"
        action = f"STANDARD: Route to {rec_group} (Eligible for automated self-heal / Fast-Track)"

    # Match Explainable Rules
    explanation = "Standard low-complexity ticket characteristics."
    cat_val = ticket_dict.get("category", "")
    pri_val = ticket_dict.get("priority", "")
    if not rules_df.empty:
        matches = rules_df[rules_df["Antecedent"].str.contains(cat_val, na=False)]
        if not matches.empty:
            r = matches.iloc[0]
            explanation = f"Matches Rule: IF {{{r['Antecedent']}}} => SLA Breach ({r['Confidence']*100:.0f}% confidence, {r['Lift']}x lift)"
        elif breach_prob >= 0.70:
            explanation = f"Historical risk pattern: {cat_val} with {pri_val} priority triggers delays."

    return {
        "Breach_Probability": breach_prob,
        "Risk_Tier": tier,
        "Badge": badge,
        "Recommended_Group": rec_group,
        "Recommended_Action": action,
        "Explainable_Reason": explanation
    }

def print_decision_card(case_num, case_name, ticket_data, prediction):
    print("\n" + "="*85)
    print(f" TEST CASE #{case_num}: {case_name.upper()}")
    print("="*85)
    print(" [INCOMING TICKET ATTRIBUTES]")
    print(f"   * Category:    {ticket_data.get('category', 'N/A'):<22} | Subcategory: {ticket_data.get('subcategory', 'N/A')}")
    print(f"   * Priority:    {ticket_data.get('priority', 'N/A'):<22} | Impact/Urgency: {ticket_data.get('impact', 'N/A')} / {ticket_data.get('urgency', 'N/A')}")
    print(f"   * Location:    {ticket_data.get('location', 'N/A'):<22} | Contact Type:  {ticket_data.get('contact_type', 'N/A')}")
    print(f"   * Open Time:   Hour {ticket_data.get('open_hour', 9):02d}:00 (Weekday {ticket_data.get('open_dayofweek', 1)})")
    print("-" * 85)
    print(" [AI DECISION SUPPORT OUTPUT]")
    print(f"   * SLA Breach Probability : {prediction['Breach_Probability']*100:.1f}%  -->  {prediction['Badge']} {prediction['Risk_Tier']}")
    print(f"   * Smart Routing Target   : {prediction['Recommended_Group']}")
    print(f"   * Recommended Action     : {prediction['Recommended_Action']}")
    print(f"   * Explainable Root-Cause : {prediction['Explainable_Reason']}")
    print("="*85)

def main():
    print("\n" + "#"*85)
    print("    SMART IT INCIDENT DECISION SUPPORT - REAL-TIME TEST CASE RUNNER")
    print("#"*85)

    artifacts, rules_df = load_system()
    if artifacts is None:
        return

    # Test Case 1: Severe Enterprise Database Degradation
    tc1 = {
        "category": "Category 19",
        "subcategory": "Subcategory 215",
        "priority": "1 - Critical",
        "impact": "1 - High",
        "urgency": "1 - High",
        "contact_type": "Phone",
        "location": "Location 143",
        "has_cmdb_ci": 1,
        "has_problem_id": 1,
        "has_rfc": 0,
        "has_vendor": 0,
        "has_caused_by": 0,
        "open_hour": 14,
        "open_dayofweek": 2
    }
    pred1 = predict_single_ticket(tc1, artifacts, rules_df)
    print_decision_card(1, "Critical Production Database Outage", tc1, pred1)

    # Test Case 2: Routine Employee Password Reset / Desktop Access
    tc2 = {
        "category": "Category 53",
        "subcategory": "Subcategory 175",
        "priority": "3 - Moderate",
        "impact": "2 - Medium",
        "urgency": "2 - Medium",
        "contact_type": "Self-service",
        "location": "Location 165",
        "has_cmdb_ci": 0,
        "has_problem_id": 0,
        "has_rfc": 0,
        "has_vendor": 0,
        "has_caused_by": 0,
        "open_hour": 10,
        "open_dayofweek": 1
    }
    pred2 = predict_single_ticket(tc2, artifacts, rules_df)
    print_decision_card(2, "Routine Employee Password Reset", tc2, pred2)

    # Test Case 3: Third-Party External Vendor Network Gateway Glitch
    tc3 = {
        "category": "Category 42",
        "subcategory": "Subcategory 224",
        "priority": "2 - High",
        "impact": "2 - Medium",
        "urgency": "1 - High",
        "contact_type": "Email",
        "location": "Location 108",
        "has_cmdb_ci": 0,
        "has_problem_id": 0,
        "has_rfc": 0,
        "has_vendor": 1,
        "has_caused_by": 0,
        "open_hour": 16,
        "open_dayofweek": 4
    }
    pred3 = predict_single_ticket(tc3, artifacts, rules_df)
    print_decision_card(3, "External Vendor Network Gateway Incident", tc3, pred3)

    # Test Case 4: Real Unseen Ticket from Test Queue
    if os.path.exists(DECISION_DASHBOARD_PATH):
        real_queue = pd.read_csv(DECISION_DASHBOARD_PATH)
        real_high = real_queue[real_queue["Risk_Tier"] == "HIGH RISK"].iloc[0]
        print("\n" + "="*85)
        print(f" TEST CASE #4: REAL PRODUCTION TICKET FROM UNSEEN TEST QUEUE ({real_high['Ticket_ID']})")
        print("="*85)
        print(f"   * Ticket ID              : {real_high['Ticket_ID']}")
        print(f"   * Category / Priority    : {real_high['Category']} / {real_high['Priority']}")
        print(f"   * Predicted Breach Risk  : {real_high['Breach_Risk']}  -->  {real_high['Risk_Tier']}")
        print(f"   * Recommended Action     : {real_high['Recommended_Action']}")
        print(f"   * Root-Cause Explanation : {real_high['Explainable_Reason']}")
        print("="*85)

    print("\nTest case demonstration completed successfully.\n")

if __name__ == "__main__":
    main()
