import pandas as pd
from src.config import DECISION_DASHBOARD_PATH, ID_COL

def run_phase6(test_df, rules_df):
    print("\n" + "="*70)
    print(">>> PHASE 6: OPERATIONAL DECISION SUPPORT (DISPATCHER QUEUE)")
    print("="*70)

    print("Synthesizing predictive risk scores with symbolic root-cause rules...")

    dashboard_records = []

    for _, row in test_df.iterrows():
        ticket_id = row[ID_COL]
        risk_score = row["predicted_breach_risk"]
        cat = row["category"]
        pri = row["priority"]
        loc = row["location"]
        rec_group = row["recommended_group"]

        # Determine Risk Tier
        if risk_score >= 0.70:
            tier = "HIGH RISK"
            action = f"URGENT: Bypass L1 -> Dispatch direct to {rec_group}"
        elif risk_score >= 0.40:
            tier = "MEDIUM"
            action = f"MONITOR: Route to {rec_group} with 24h milestone alert"
        else:
            tier = "LOW RISK"
            action = f"STANDARD: Route to {rec_group} (Fast-track)"

        # Match with discovered rules for explainability
        explanation = "Standard ticket characteristics."
        if not rules_df.empty:
            cat_match = rules_df[rules_df["Antecedent"].str.contains(str(cat), na=False)]
            if not cat_match.empty:
                top_rule = cat_match.iloc[0]
                explanation = f"Matches Rule: {{{top_rule['Antecedent']}}} (Historical Confidence: {top_rule['Confidence']*100:.0f}%)"
            elif risk_score >= 0.70:
                explanation = f"High risk pattern: {cat} with {pri} priority."

        dashboard_records.append({
            "Ticket_ID": ticket_id,
            "Category": cat,
            "Priority": pri,
            "Breach_Risk": f"{risk_score*100:.1f}%",
            "Risk_Tier": tier,
            "Recommended_Action": action,
            "Explainable_Reason": explanation
        })

    queue_df = pd.DataFrame(dashboard_records)
    queue_df = queue_df.sort_values(by="Risk_Tier", ascending=True).reset_index(drop=True)
    queue_df.to_csv(DECISION_DASHBOARD_PATH, index=False)

    print(f"Generated Live Decision Support Queue for {len(queue_df):,} incoming tickets.")
    print(f"Saved to: {DECISION_DASHBOARD_PATH}\n")

    # Display Top 10 sample tickets across tiers
    sample_high = queue_df[queue_df["Risk_Tier"] == "HIGH RISK"].head(4)
    sample_med = queue_df[queue_df["Risk_Tier"] == "MEDIUM"].head(3)
    sample_low = queue_df[queue_df["Risk_Tier"] == "LOW RISK"].head(3)
    sample_display = pd.concat([sample_high, sample_med, sample_low])

    print("="*105)
    print("                      SAMPLE REAL-TIME DISPATCHER & MANAGER DASHBOARD")
    print("="*105)
    print(f"{'Ticket ID':<12} | {'Risk %':<8} | {'Tier':<10} | {'Recommended Action':<38} | {'Explainable Reason'}")
    print("-" * 105)
    for _, r in sample_display.iterrows():
        print(f"{r['Ticket_ID']:<12} | {r['Breach_Risk']:<8} | {r['Risk_Tier']:<10} | {r['Recommended_Action'][:36]:<38} | {r['Explainable_Reason'][:50]}")
    print("="*105)

    return queue_df

if __name__ == "__main__":
    from src.p1_preprocess import run_phase1
    from src.p2_process_mining import run_phase2
    from src.p3_profiling_clustering import run_phase3
    from src.p4_association_rules import run_phase4
    from src.p5_predictive_modeling import run_phase5
    traces, cases = run_phase1()
    feats = run_phase2(traces)
    enriched = run_phase3(cases, feats)
    rules = run_phase4(enriched)
    _, _, test_df = run_phase5(enriched)
    run_phase6(test_df, rules)
