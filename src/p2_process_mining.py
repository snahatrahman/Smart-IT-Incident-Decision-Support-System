import os
import pandas as pd
import numpy as np
from src.config import (
    CLEANED_TRACES_PATH, OUTPUT_DIR, ID_COL, STATE_COL,
    GROUP_COL, TIMESTAMP_COL
)

def run_phase2(traces_df=None):
    print("\n" + "="*70)
    print(">>> PHASE 2: GLOBAL PROCESS MINING & BOTTLENECK DISCOVERY")
    print("="*70)

    if traces_df is None:
        print(f"Loading event traces from: {CLEANED_TRACES_PATH}")
        traces_df = pd.read_csv(CLEANED_TRACES_PATH)
        traces_df[TIMESTAMP_COL] = pd.to_datetime(traces_df[TIMESTAMP_COL])

    print(f"Analyzing {len(traces_df):,} event records across {traces_df[ID_COL].nunique():,} cases...")

    # Ensure chronological sorting
    traces_df = traces_df.sort_values(by=[ID_COL, TIMESTAMP_COL]).copy()

    # 1. State Transitions (Directly-Follows Graph)
    traces_df["next_state"] = traces_df.groupby(ID_COL)[STATE_COL].shift(-1)
    traces_df["next_group"] = traces_df.groupby(ID_COL)[GROUP_COL].shift(-1)
    traces_df["next_timestamp"] = traces_df.groupby(ID_COL)[TIMESTAMP_COL].shift(-1)

    # Compute step duration in hours
    traces_df["step_duration_hrs"] = (
        (traces_df["next_timestamp"] - traces_df[TIMESTAMP_COL]).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)

    # Top state transitions
    state_transitions = (
        traces_df.dropna(subset=["next_state"])
        .groupby([STATE_COL, "next_state"])
        .agg(
            frequency=(ID_COL, "count"),
            avg_duration_hrs=("step_duration_hrs", "mean")
        )
        .reset_index()
        .sort_values(by="frequency", ascending=False)
    )

    print("\n--- Top 5 Discovered State Transitions (DFG) ---")
    print(state_transitions.head(5).to_string(index=False))

    # Save bottleneck transition map
    bottlenecks_path = os.path.join(OUTPUT_DIR, "process_bottlenecks.csv")
    state_transitions.to_csv(bottlenecks_path, index=False)
    print(f"Full process transitions saved to: {bottlenecks_path}")

    # 2. Ping-Pong Loop Detection
    # A ping-pong loop occurs when a ticket bounces back to a previously visited group:
    # e.g., Group 70 -> Group 24 -> Group 70
    traces_df["prev_group"] = traces_df.groupby(ID_COL)[GROUP_COL].shift(1)
    traces_df["prev_prev_group"] = traces_df.groupby(ID_COL)[GROUP_COL].shift(2)

    traces_df["is_ping_pong_step"] = (
        (traces_df[GROUP_COL] == traces_df["prev_prev_group"]) &
        (traces_df[GROUP_COL] != traces_df["prev_group"]) &
        traces_df[GROUP_COL].notna() &
        traces_df["prev_prev_group"].notna()
    ).astype(int)

    # 3. Holding State Dwell Time Calculation
    # Waiting states: Awaiting User Info, Awaiting Vendor, Awaiting Problem, Awaiting Evidence
    waiting_states = ["Awaiting User Info", "Awaiting Vendor", "Awaiting Problem", "Awaiting Evidence"]
    traces_df["is_waiting_state"] = traces_df[STATE_COL].isin(waiting_states).astype(int)
    traces_df["waiting_dwell_hrs"] = traces_df["step_duration_hrs"] * traces_df["is_waiting_state"]

    # Precompute vectorized indicators for fast aggregation
    traces_df["is_handoff"] = (
        traces_df["next_group"].notna() &
        (traces_df[GROUP_COL] != traces_df["next_group"])
    ).astype(int)

    traces_df["active_dwell_hrs"] = (
        traces_df["step_duration_hrs"] * (traces_df[STATE_COL] == "Active")
    ).fillna(0.0)

    # 4. Aggregate Engineered Process Features per Case (Ticket)
    print("\nExtracting behavioral trace metrics per ticket...")
    process_features = traces_df.groupby(ID_COL).agg(
        total_events=(STATE_COL, "count"),
        unique_groups=(GROUP_COL, "nunique"),
        total_handoffs=("is_handoff", "sum"),
        ping_pong_count=("is_ping_pong_step", "sum"),
        total_wait_hours=("waiting_dwell_hrs", "sum"),
        total_active_hours=("active_dwell_hrs", "sum")
    ).reset_index()

    process_features["is_ping_pong"] = (process_features["ping_pong_count"] > 0).astype(int)

    print(f"Extracted {len(process_features.columns) - 1} behavioral features for {len(process_features):,} tickets")
    print(f"Ping-pong rework detected in {process_features['is_ping_pong'].sum():,} tickets "
          f"({process_features['is_ping_pong'].mean()*100:.1f}%)")
    print(f"Average waiting dwell time: {process_features['total_wait_hours'].mean():.1f} hours")

    return process_features

if __name__ == "__main__":
    run_phase2()
