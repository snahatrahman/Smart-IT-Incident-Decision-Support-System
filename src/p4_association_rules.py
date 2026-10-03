import itertools
import pandas as pd
import numpy as np
from src.config import DISCOVERED_RULES_PATH, TARGET_BREACH

def run_phase4(enriched_df):
    print("\n" + "="*70)
    print(">>> PHASE 4: ROOT-CAUSE PATTERN & ASSOCIATION RULE MINING")
    print("="*70)

    print("Discretizing continuous metrics into categorical transaction itemsets...")
    trans_df = pd.DataFrame()

    # Discretize handoffs
    trans_df["Handoffs"] = pd.cut(
        enriched_df["total_handoffs"],
        bins=[-1, 1, 3, 100],
        labels=["Handoffs: 0-1", "Handoffs: 2-3", "Handoffs: 4+"]
    ).astype(str)

    # Discretize idle waiting time
    trans_df["WaitTime"] = pd.cut(
        enriched_df["total_wait_hours"],
        bins=[-1, 2, 24, 10000],
        labels=["Wait: Low (<2h)", "Wait: Med (2-24h)", "Wait: High (>24h)"]
    ).astype(str)

    # Ping-pong flag
    trans_df["PingPong"] = enriched_df["is_ping_pong"].map({1: "PingPong: Yes", 0: "PingPong: No"})

    # Priority
    trans_df["Priority"] = "Priority: " + enriched_df["priority"].astype(str)

    # Top categories (group infrequent categories into 'Other')
    top_cats = enriched_df["category"].value_counts().head(8).index
    trans_df["Category"] = enriched_df["category"].apply(lambda c: f"Cat: {c}" if c in top_cats else "Cat: Other")

    # Target consequent
    trans_df["Target"] = enriched_df[TARGET_BREACH].map({1: "Outcome: SLA Breached", 0: "Outcome: SLA Met"})

    n_cases = len(trans_df)
    target_val = "Outcome: SLA Breached"
    base_support_target = (trans_df["Target"] == target_val).mean()
    print(f"Dataset baseline SLA breach rate: {base_support_target*100:.1f}%\n")

    # Feature columns to combine for antecedents
    feature_cols = ["Category", "Priority", "Handoffs", "WaitTime", "PingPong"]

    rules_list = []

    # 1-item and 2-item antecedents
    for r in [1, 2]:
        for col_combo in itertools.combinations(feature_cols, r):
            grouped = trans_df.groupby(list(col_combo))
            for val_combo, group in grouped:
                ant_count = len(group)
                if ant_count < 50:  # Minimum frequency threshold
                    continue

                # Support of Antecedent
                support_ant = ant_count / n_cases

                # Co-occurrence with target
                both_count = (group["Target"] == target_val).sum()
                support_both = both_count / n_cases

                # Confidence = P(Target | Antecedent)
                confidence = both_count / ant_count if ant_count > 0 else 0

                # Lift = Confidence / Baseline Target Rate
                lift = confidence / base_support_target if base_support_target > 0 else 0

                # Filter for strong rules: high confidence and meaningful positive lift
                if confidence >= 0.60 and lift > 1.25 and support_both >= 0.01:
                    ant_str = " AND ".join(val_combo) if isinstance(val_combo, tuple) else val_combo
                    rules_list.append({
                        "Antecedent": ant_str,
                        "Consequent": target_val,
                        "Support": round(support_both, 4),
                        "Confidence": round(confidence, 4),
                        "Lift": round(lift, 2),
                        "Frequency": both_count
                    })

    rules_df = pd.DataFrame(rules_list)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by=["Confidence", "Lift"], ascending=False).reset_index(drop=True)
        rules_df.to_csv(DISCOVERED_RULES_PATH, index=False)
        print(f"Discovered {len(rules_df)} high-confidence failure rules (saved to {DISCOVERED_RULES_PATH})")
        print("\n--- Top 5 Discovered Root-Cause Association Rules ---")
        for idx, row in rules_df.head(5).iterrows():
            print(f"Rule #{idx+1}: IF {{{row['Antecedent']}}} => {{{row['Consequent']}}}")
            print(f"         Confidence: {row['Confidence']*100:.1f}% | Lift: {row['Lift']}x | Cases: {row['Frequency']:,}\n")
    else:
        print("No rules met the strict thresholds. Exporting empty rule store.")
        rules_df = pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Frequency"])
        rules_df.to_csv(DISCOVERED_RULES_PATH, index=False)

    return rules_df

if __name__ == "__main__":
    from src.p1_preprocess import run_phase1
    from src.p2_process_mining import run_phase2
    from src.p3_profiling_clustering import run_phase3
    traces, cases = run_phase1()
    feats = run_phase2(traces)
    enriched = run_phase3(cases, feats)
    run_phase4(enriched)
