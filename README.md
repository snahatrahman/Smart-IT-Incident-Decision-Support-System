# Smart IT Incident Decision Support System

An end-to-end IT Service Management (ITSM) decision support framework that leverages **Process Mining**, **Predictive Machine Learning (GBDT)**, and **Symbolic Association Rule Mining** to eliminate SLA breaches, detect ping-pong rework bottlenecks, and optimize specialist routing at intake.

---

## Key Highlights

- **SLA Breach Risk Classifier**: Predicts failure risk at ticket creation (\(t=0\)) with strict chronological temporal validation (zero lookahead leakage), achieving **~0.76 ROC-AUC**.
- **Smart Routing Recommender**: Predicts the exact resolving specialist group with **~75.7% top-1 accuracy**, eliminating 1–3 intermediate triage handoffs.
- **Explainable Root-Cause Rules**: Discovers high-confidence IF-THEN heuristic rules (\(\text{Confidence} \ge 80\%\), \(\text{Lift} > 2\times\)) to provide dispatchers and managers with transparent rationales.
- **Process Mining & Bottleneck Discovery**: Directly-Follows Graphs (DFG), ping-pong bounce detection, and holding-state dwell time quantification.
- **Audit-Retained Anomaly Profiling**: Flags pathological outliers via Isolation Forest without discarding them, ensuring resilience against catastrophic delays.

---

## Quickstart Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full 6-Phase Pipeline
```bash
python run_pipeline.py
```

### 3. Run Real-Time Incident Test Cases
```bash
python test_cases.py
```

---

## Documentation & Walkthrough

For a complete, in-depth guide covering the 6-phase pipeline architecture, system requirements, step-by-step setup on Windows/Linux/macOS, output artifact schemas, operational playbooks, and troubleshooting FAQs, refer to:

👉 **[WALKTHROUGH.txt](WALKTHROUGH.txt)**

For the visual architecture diagram and detailed methodology documentation, open:

👉 **[DM_pipeline_corrected.html](DM_pipeline_corrected.html)**

---

## Repository Structure

```text
Smart-IT-Incident-Decision-Support-System/
├── .gitignore                    # Excludes raw data, generated models & caches
├── README.md                     # Project overview and quickstart
├── WALKTHROUGH.txt               # In-depth operational and execution walkthrough
├── requirements.txt              # Package dependencies
├── run_pipeline.py               # Master runner for Phases 1–6
├── test_cases.py                 # Real-time incident test evaluator
├── DM_pipeline_corrected.html    # Standalone visual architecture & diagram
│
├── incident_event_log.csv        # [Raw Data - Git Ignored]
│
├── src/                          # Modular Python source code
│   ├── config.py                 # Configuration paths and hyperparameters
│   ├── p1_preprocess.py          # Phase 1: Cleaning & Dual-Stream Separation
│   ├── p2_process_mining.py      # Phase 2: Process Mining & Bottlenecks
│   ├── p3_profiling_clustering.py# Phase 3: Case Feature Store & Anomaly Profiling
│   ├── p4_association_rules.py   # Phase 4: Association Rule Mining
│   ├── p5_predictive_modeling.py # Phase 5: GBDT Predictive Modeling
│   └── p6_decision_support.py    # Phase 6: Dispatcher Queue & Dashboard
│
└── outputs/                      # Generated pipeline deliverables
    ├── .gitkeep                  # Preserves directory in git
    ├── cleaned_traces.csv
    ├── process_bottlenecks.csv
    ├── enriched_case_features.csv
    ├── pathological_outliers.csv
    ├── root_cause_rules.csv
    ├── trained_models.joblib
    ├── model_evaluation_metrics.txt
    └── live_dispatcher_queue.csv
```

---

## License & Notes
- Dataset: ServiceNow benchmark incident event log.
- Raw data files and trained binary models are excluded from Git commits via `.gitignore`.
## Dataset
Download the incident event log from the UCI repository (https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log) and save it as incident_event_log.csv in the project root before running run_pipeline.py.
