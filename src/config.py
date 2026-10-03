# -*- coding: utf-8 -*-
"""
Configuration file: Defines project paths, column sets, and parameters.
Keeps all settings in one easy-to-read location.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "incident_event_log.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processed intermediate data paths
CLEANED_TRACES_PATH = os.path.join(OUTPUT_DIR, "cleaned_traces.csv")
ENRICHED_CASES_PATH = os.path.join(OUTPUT_DIR, "enriched_case_features.csv")
OUTLIERS_AUDIT_PATH = os.path.join(OUTPUT_DIR, "pathological_outliers.csv")
DISCOVERED_RULES_PATH = os.path.join(OUTPUT_DIR, "root_cause_rules.csv")
EVALUATION_METRICS_PATH = os.path.join(OUTPUT_DIR, "model_evaluation_metrics.txt")
DECISION_DASHBOARD_PATH = os.path.join(OUTPUT_DIR, "live_dispatcher_queue.csv")

# Column definitions
ID_COL = "number"
STATE_COL = "incident_state"
GROUP_COL = "assignment_group"
TIMESTAMP_COL = "sys_updated_at"
RESOURCE_COL = "assigned_to"

# Features available strictly at ticket creation (Intake - time zero)
INTAKE_CATEGORICAL = [
    "category",
    "subcategory",
    "contact_type",
    "location",
    "impact",
    "urgency",
    "priority",
]

# Sparse indicator columns (presence flag)
SPARSE_COLS = ["cmdb_ci", "problem_id", "rfc", "vendor", "caused_by"]

# Target labels
TARGET_SLA = "made_sla"          # Ground-truth: True (Met SLA) vs False (Breached SLA)
TARGET_BREACH = "is_breached"    # 1 if breached (False), 0 if met (True)

# Modeling parameters
RANDOM_STATE = 42
TEMPORAL_SPLIT_RATIO = 0.8       # 80% older tickets for training, 20% newer for testing
OUTLIER_CONTAMINATION = 0.03     # Top 3% most extreme pathological tickets
