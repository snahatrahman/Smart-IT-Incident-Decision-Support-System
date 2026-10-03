# -*- coding: utf-8 -*-
"""
Master Execution Script: Runs the entire 6-Phase Smart IT Incident Pipeline.
Usage:
    python run_pipeline.py
"""
import time
import os
from src.p1_preprocess import run_phase1
from src.p2_process_mining import run_phase2
from src.p3_profiling_clustering import run_phase3
from src.p4_association_rules import run_phase4
from src.p5_predictive_modeling import run_phase5
from src.p6_decision_support import run_phase6
from src.config import OUTPUT_DIR

def main():
    start_total = time.time()
    print("\n" + "#"*75)
    print("      SMART IT INCIDENT DECISION SUPPORT PIPELINE - MASTER RUNNER")
    print("#"*75)

    # Phase 1: Data Preprocessing & Dual-Stream Separation
    t0 = time.time()
    traces_df, cases_df = run_phase1()
    print(f">> Phase 1 completed in {time.time() - t0:.1f}s")

    # Phase 2: Process Mining & Bottleneck Discovery
    t0 = time.time()
    process_features = run_phase2(traces_df)
    print(f">> Phase 2 completed in {time.time() - t0:.1f}s")

    # Phase 3: Case Feature Integration, Anomaly Profiling & Clustering
    t0 = time.time()
    enriched_df = run_phase3(cases_df, process_features)
    print(f">> Phase 3 completed in {time.time() - t0:.1f}s")

    # Phase 4: Root-Cause Pattern & Association Rule Mining
    t0 = time.time()
    rules_df = run_phase4(enriched_df)
    print(f">> Phase 4 completed in {time.time() - t0:.1f}s")

    # Phase 5: Temporal Predictive Modeling (ML)
    t0 = time.time()
    clf_breach, clf_route, test_df = run_phase5(enriched_df)
    print(f">> Phase 5 completed in {time.time() - t0:.1f}s")

    # Phase 6: Operational Decision Support Dashboard
    t0 = time.time()
    queue_df = run_phase6(test_df, rules_df)
    print(f">> Phase 6 completed in {time.time() - t0:.1f}s")

    total_duration = time.time() - start_total
    print("\n" + "#"*75)
    print(f"      PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_duration:.1f}s!")
    print("#"*75)
    print(f"\nGenerated Output Artifacts in: {OUTPUT_DIR}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  - {fname:<32} ({size_kb:,.1f} KB)")
    print("\nAll pipeline tasks are complete and ready for review.")

if __name__ == "__main__":
    main()
