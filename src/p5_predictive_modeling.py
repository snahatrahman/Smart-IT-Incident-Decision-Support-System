import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from src.config import (
    EVALUATION_METRICS_PATH, OUTPUT_DIR, RANDOM_STATE, TEMPORAL_SPLIT_RATIO,
    TARGET_BREACH, INTAKE_CATEGORICAL, SPARSE_COLS
)

def run_phase5(enriched_df):
    print("\n" + "="*70)
    print(">>> PHASE 5: TEMPORAL PREDICTIVE MODELING (ML ENSEMBLE)")
    print("="*70)

    # 1. Enforce strict chronological order to avoid lookahead leakage
    enriched_df = enriched_df.sort_values(by="opened_at").reset_index(drop=True)
    n_total = len(enriched_df)
    n_train = int(n_total * TEMPORAL_SPLIT_RATIO)

    train_df = enriched_df.iloc[:n_train].copy()
    test_df = enriched_df.iloc[n_train:].copy()

    print(f"Total dataset: {n_total:,} tickets")
    print(f"  -> Temporal Train Set: {len(train_df):,} tickets ({train_df['opened_at'].min().strftime('%Y-%m-%d')} to {train_df['opened_at'].max().strftime('%Y-%m-%d')})")
    print(f"  -> Temporal Test Set:  {len(test_df):,} tickets ({test_df['opened_at'].min().strftime('%Y-%m-%d')} to {test_df['opened_at'].max().strftime('%Y-%m-%d')})")

    # Features strictly available at ticket intake (t=0)
    intake_features = ["open_hour", "open_dayofweek"] + INTAKE_CATEGORICAL + [f"has_{c}" for c in SPARSE_COLS]

    # Categorical encoding using OrdinalEncoder (safe for HistGradientBoosting)
    cat_cols_indices = [i for i, c in enumerate(intake_features) if c in INTAKE_CATEGORICAL]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    X_train_cat = encoder.fit_transform(train_df[INTAKE_CATEGORICAL].astype(str))
    X_test_cat = encoder.transform(test_df[INTAKE_CATEGORICAL].astype(str))

    X_train = train_df[intake_features].copy()
    X_test = test_df[intake_features].copy()

    X_train[INTAKE_CATEGORICAL] = X_train_cat
    X_test[INTAKE_CATEGORICAL] = X_test_cat

    y_train = train_df[TARGET_BREACH].values
    y_test = test_df[TARGET_BREACH].values

    # -------------------------------------------------------------
    # Model 1: Intake SLA Breach Risk Classifier
    # -------------------------------------------------------------
    print("\nTraining Model 1: Intake SLA Breach Classifier (HistGradientBoosting)...")
    clf_breach = HistGradientBoostingClassifier(
        categorical_features=cat_cols_indices,
        random_state=RANDOM_STATE,
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=25
    )
    clf_breach.fit(X_train, y_train)

    y_pred_proba = clf_breach.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, target_names=["Met SLA", "Breached SLA"])

    print(f"\n[Model 1 Performance on Unseen Future Tickets]")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(report)

    # -------------------------------------------------------------
    # Model 2: Smart Routing Recommender
    # -------------------------------------------------------------
    print("Training Model 2: Smart Routing Recommender (Predicting Resolving Group)...")
    # Identify top 12 most frequent resolving groups
    top_groups = train_df["final_resolving_group"].value_counts().head(12).index.tolist()
    
    # Filter train/test for known top groups
    train_mask = train_df["final_resolving_group"].isin(top_groups)
    test_mask = test_df["final_resolving_group"].isin(top_groups)

    group_encoder = OrdinalEncoder()
    y_train_grp = group_encoder.fit_transform(train_df.loc[train_mask, ["final_resolving_group"]]).ravel()
    y_test_grp = group_encoder.transform(test_df.loc[test_mask, ["final_resolving_group"]]).ravel()

    clf_route = HistGradientBoostingClassifier(
        categorical_features=cat_cols_indices,
        random_state=RANDOM_STATE,
        max_iter=100
    )
    clf_route.fit(X_train.loc[train_mask], y_train_grp)
    y_pred_grp = clf_route.predict(X_test.loc[test_mask])

    routing_acc = accuracy_score(y_test_grp, y_pred_grp)
    print(f"Top-1 Smart Routing Accuracy: {routing_acc*100:.1f}% across top {len(top_groups)} specialist groups\n")

    # Save metrics report
    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("=== SMART IT INCIDENT DECISION SUPPORT - MODEL EVALUATION ===\n\n")
        f.write(f"Validation Strategy: Strict Chronological Temporal Split ({int(TEMPORAL_SPLIT_RATIO*100)}% Train / {int((1-TEMPORAL_SPLIT_RATIO)*100)}% Test)\n")
        f.write(f"Train Cases: {len(train_df):,} | Test Cases: {len(test_df):,}\n\n")
        f.write(f"--- Model 1: Intake SLA Breach Risk ---\n")
        f.write(f"ROC-AUC Score: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write(f"--- Model 2: Smart Assignment Group Routing ---\n")
        f.write(f"Top Specialist Groups: {len(top_groups)}\n")
        f.write(f"Direct Route Accuracy: {routing_acc*100:.1f}%\n")

    print(f"Evaluation metrics report saved to: {EVALUATION_METRICS_PATH}")

    # Attach predictions back to test_df for Phase 6 decision dashboard
    test_df["predicted_breach_risk"] = y_pred_proba
    test_df["predicted_breach_flag"] = y_pred

    # Predict suggested group for test cases
    test_pred_grp_indices = clf_route.predict(X_test)
    test_df["recommended_group"] = group_encoder.inverse_transform(test_pred_grp_indices.reshape(-1, 1)).ravel()

    # Serialize model artifacts for instant standalone test-case inference
    import joblib
    models_path = os.path.join(OUTPUT_DIR, "trained_models.joblib")
    model_artifacts = {
        "clf_breach": clf_breach,
        "clf_route": clf_route,
        "encoder": encoder,
        "group_encoder": group_encoder,
        "intake_features": intake_features,
        "INTAKE_CATEGORICAL": INTAKE_CATEGORICAL,
        "SPARSE_COLS": SPARSE_COLS
    }
    joblib.dump(model_artifacts, models_path)
    print(f"Model artifacts serialized to: {models_path}")

    return clf_breach, clf_route, test_df

if __name__ == "__main__":
    from src.p1_preprocess import run_phase1
    from src.p2_process_mining import run_phase2
    from src.p3_profiling_clustering import run_phase3
    traces, cases = run_phase1()
    feats = run_phase2(traces)
    enriched = run_phase3(cases, feats)
    run_phase5(enriched)
