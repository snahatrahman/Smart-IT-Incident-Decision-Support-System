import pandas as pd
import numpy as np
from src.config import ( DATA_PATH, CLEANED_TRACES_PATH, ID_COL, STATE_COL, GROUP_COL, TIMESTAMP_COL, RESOURCE_COL, INTAKE_CATEGORICAL, SPARSE_COLS, TARGET_SLA, TARGET_BREACH)

def run_phase1():
    print("\n" + "="*70)
    print(">>> PHASE 1: DATA CLEANING & DUAL-STREAM SEPARATION")
    print("="*70)

    print(f"Loading raw logs from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Raw rows loaded: {len(df):,} | Unique cases: {df[ID_COL].nunique():,}")

    # 1. Clean missing representation: '?' -> NaN
    df = df.replace("?", np.nan)

    # 2. Remove corrupted system states (e.g. '-100')
    initial_len = len(df)
    df = df[df[STATE_COL] != "-100"].copy()
    print(f"Filtered {initial_len - len(df)} corrupted rows with state '-100'")

    # 3. Create binary presence flags for sparse attributes (>98% missing)
    for col in SPARSE_COLS:
        flag_col = f"has_{col}"
        df[flag_col] = df[col].notna().astype(int)

    # 4. Parse timestamps (%d/%m/%Y %H:%M)
    date_cols = ["opened_at", TIMESTAMP_COL, "resolved_at", "closed_at"]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], format="%d/%m/%Y %H:%M", errors="coerce")

    # 5. Sort chronologically by update timestamp within each case
    df = df.sort_values(by=[ID_COL, TIMESTAMP_COL]).reset_index(drop=True)

    # -------------------------------------------------------------
    # Stream A: Event Trace Log (For Process Mining)
    # -------------------------------------------------------------
    print("\nBuilding Stream A: Event Trace Stream...")
    trace_cols = [ID_COL, TIMESTAMP_COL, STATE_COL, GROUP_COL, RESOURCE_COL]
    traces_df = df[trace_cols].copy()
    traces_df.to_csv(CLEANED_TRACES_PATH, index=False)
    print(f"Stream A saved to: {CLEANED_TRACES_PATH} ({len(traces_df):,} events)")

    # -------------------------------------------------------------
    # Stream B: Case Intake Matrix (1 row per ticket at t=0)
    # -------------------------------------------------------------
    print("Building Stream B: Case Intake Matrix (Zero Future Leakage)...")

    # Group by ticket ID to get initial intake attributes and final outcome
    first_updates = df.groupby(ID_COL).first().reset_index()
    last_updates = df.groupby(ID_COL).last().reset_index()

    cases_df = pd.DataFrame()
    cases_df[ID_COL] = first_updates[ID_COL]
    cases_df["opened_at"] = first_updates["opened_at"]
    cases_df["open_hour"] = first_updates["opened_at"].dt.hour
    cases_df["open_dayofweek"] = first_updates["opened_at"].dt.dayofweek

    # Static intake attributes
    for col in INTAKE_CATEGORICAL + ["caller_id"]:
        cases_df[col] = first_updates[col].fillna("Unknown")

    # Sparse indicator flags (known at intake/initial ticket)
    for col in SPARSE_COLS:
        flag_col = f"has_{col}"
        cases_df[flag_col] = first_updates[flag_col]

    # Ground-truth targets from the FINAL state of the ticket
    # made_sla: In the dataset, True means Met SLA, False means Breached SLA
    final_sla_str = last_updates[TARGET_SLA].astype(str).str.strip().str.lower()
    cases_df[TARGET_SLA] = (final_sla_str == "true")
    cases_df[TARGET_BREACH] = (~cases_df[TARGET_SLA]).astype(int)

    # Final resolution timestamps & duration in hours
    res_time = last_updates["resolved_at"].fillna(last_updates["closed_at"])
    cases_df["resolution_time_hrs"] = (res_time - cases_df["opened_at"]).dt.total_seconds() / 3600.0
    cases_df["resolution_time_hrs"] = cases_df["resolution_time_hrs"].clip(lower=0.0)

    # Final resolving assignment group (target for smart routing)
    cases_df["final_resolving_group"] = last_updates[GROUP_COL].fillna("Unknown")

    print(f"Stream B constructed: {len(cases_df):,} cases")
    print(f"SLA Target Distribution: {cases_df[TARGET_BREACH].value_counts(normalize=True).to_dict()} (1 = Breached)")

    return traces_df, cases_df

if __name__ == "__main__":
    run_phase1()
