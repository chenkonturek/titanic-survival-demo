"""
Run modeling pipeline for Titanic survival prediction.
Train/Val/Test split: 70/15/15
Models: Logistic Regression, Random Forest, Gradient Boosting
"""
import sys
sys.path.insert(0, '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/02_src')

import pandas as pd
import json
from modeling import (
    split_data, scale_features, train_all_models, select_best_model,
    evaluate_model, plot_feature_importance, plot_roc_curve,
    plot_model_comparison, plot_confusion_matrix, save_model,
    FEATURE_COLS, FIGURES_DIR, MODELS_DIR
)

PROCESSED_PATH = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/00_data/processed/titanic_cleaned.parquet'

if __name__ == '__main__':
    print("=== Phase 3: ML Modeling ===\n")

    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded cleaned data: {df.shape}")

    # --- Data Split ---
    print("\n--- Train/Val/Test Split (70/15/15) ---")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # --- Feature Scaling ---
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    # --- Train All Models ---
    print("\n--- Training Models ---")
    results, fitted_models = train_all_models(
        X_train, y_train, X_val, y_val, X_train_sc, X_val_sc
    )

    # --- Select Best Model ---
    best_name = select_best_model(results)
    best_model = fitted_models[best_name]

    # Determine correct test features (scaled for LR, raw for tree-based)
    if best_name == 'Logistic Regression':
        X_test_best = X_test_sc
    else:
        X_test_best = X_test

    # --- Test Set Evaluation ---
    print("\n--- Test Set Evaluation (Best Model) ---")
    test_metrics = evaluate_model(best_model, X_test_best, y_test, label='Test')

    # --- Print Full Comparison Table ---
    print("\n--- Validation Metrics Summary ---")
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 57)
    for name, res in results.items():
        m = res['val']
        print(f"{name:<25} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>6.4f}")

    print(f"\n--- Test Metrics ({best_name}) ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Save Intermediate Results ---
    results_summary = {
        'best_model': best_name,
        'validation_metrics': {k: {m: round(v, 4) for m, v in res['val'].items()}
                                for k, res in results.items()},
        'test_metrics': {k: round(v, 4) for k, v in test_metrics.items()},
    }
    intermediate_path = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/00_data/intermediate/model_results.json'
    with open(intermediate_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {intermediate_path}")

    # --- Plots ---
    print("\n--- Generating Plots ---")
    plot_model_comparison(results, f'{FIGURES_DIR}/model_comparison.png')

    fi_df = plot_feature_importance(
        best_model, FEATURE_COLS,
        f'{FIGURES_DIR}/feature_importance.png',
        model_name=best_name
    )
    print("\nTop 5 Feature Importances:")
    print(fi_df.head(5).to_string(index=False))

    # Save feature importance CSV for report
    fi_df.to_csv('/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/00_data/intermediate/feature_importances.csv', index=False)

    plot_roc_curve(best_model, X_test_best, y_test,
                   f'{FIGURES_DIR}/roc_curve.png', model_name=best_name)

    plot_confusion_matrix(best_model, X_test_best, y_test,
                          f'{FIGURES_DIR}/confusion_matrix.png', model_name=best_name)

    # --- Save Best Model ---
    print("\n--- Saving Best Model ---")
    save_model(best_model, scaler, f'{MODELS_DIR}/titanic_predictor_v1')

    print("\n=== Modeling Complete ===")
