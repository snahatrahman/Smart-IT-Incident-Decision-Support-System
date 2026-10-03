import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.config import (
    ENRICHED_CASES_PATH, OUTLIERS_AUDIT_PATH, ID_COL,
    RANDOM_STATE, OUTLIER_CONTAMINATION, TARGET_BREACH
)

def run_phase3(cases_df, process_features):
    print("\n" + "="*70)
    print(">>> PHASE 3: CASE FEATURE STORE, ANOMALY PROFILING & CLUSTERING")
    print("="*70)

    # 1. Merge into unified Case Feature Store
    print(f"Merging intake attributes ({len(cases_df):,} cases) with process features...")
    enriched_df = pd.merge(cases_df, process_features, on=ID_COL, how="inner")
    print(f"Unified Case Feature Store created: {len(enriched_df):,} rows x {len(enriched_df.columns)} columns")

    # 2. Anomaly Profiling via Isolation Forest
    # We inspect extreme values of resolution time, events, handoffs, and waiting hours.
    print(f"\nProfiling pathological incidents using Isolation Forest (Top {OUTLIER_CONTAMINATION*100:.0f}%)...")
    anomaly_feature_cols = [
        "resolution_time_hrs", "total_events", "total_handoffs",
        "total_wait_hours", "ping_pong_count"
    ]
    
    # Fill any potential NaN with 0 for scaling
    X_anom = enriched_df[anomaly_feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_anom_scaled = scaler.fit_transform(X_anom)

    iso_forest = IsolationForest(
        contamination=OUTLIER_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=100
    )
    # -1 for outliers, 1 for inliers
    preds = iso_forest.fit_predict(X_anom_scaled)
    enriched_df["is_pathological"] = (preds == -1).astype(int)

    outliers_df = enriched_df[enriched_df["is_pathological"] == 1].copy()
    outliers_df.to_csv(OUTLIERS_AUDIT_PATH, index=False)
    
    print(f"Identified {len(outliers_df):,} pathological tickets.")
    print(f"  -> Pathological SLA Breach Rate: {outliers_df[TARGET_BREACH].mean()*100:.1f}% "
          f"(vs normal: {enriched_df[enriched_df['is_pathological'] == 0][TARGET_BREACH].mean()*100:.1f}%)")
    print(f"  -> Pathological Avg Resolution Time: {outliers_df['resolution_time_hrs'].mean():.1f} hrs "
          f"(vs normal: {enriched_df[enriched_df['is_pathological'] == 0]['resolution_time_hrs'].mean():.1f} hrs)")
    print(f"  -> [CRUCIAL RULE APPLIED]: All {len(outliers_df):,} outliers are RETAINED for root-cause analysis!")

    # 3. Workload Segmentation (Clustering)
    print("\nSegmenting workload into 3 operational archetypes via K-Means...")
    cluster_features = ["resolution_time_hrs", "total_handoffs", "total_wait_hours"]
    X_clust = scaler.fit_transform(enriched_df[cluster_features].fillna(0).values)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    enriched_df["workload_cluster"] = kmeans.fit_predict(X_clust)

    # Label clusters intuitively based on average resolution time and handoffs
    cluster_means = enriched_df.groupby("workload_cluster")[cluster_features].mean()
    print("\nCluster Characteristics:")
    print(cluster_means.round(1).to_string())

    # Save complete enriched feature store
    enriched_df.to_csv(ENRICHED_CASES_PATH, index=False)
    print(f"\nUnified Case Feature Store saved to: {ENRICHED_CASES_PATH}")

    return enriched_df

if __name__ == "__main__":
    from src.p1_preprocess import run_phase1
    from src.p2_process_mining import run_phase2
    traces, cases = run_phase1()
    feats = run_phase2(traces)
    run_phase3(cases, feats)
