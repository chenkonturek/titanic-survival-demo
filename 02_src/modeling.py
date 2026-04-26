"""
Model training, evaluation, and feature importance functions for Titanic survival prediction.
Author: ml-model-scientist agent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay
)
import pickle


FEATURE_COLS = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
                'embarked', 'Title', 'FamilySize', 'IsAlone']
TARGET_COL = 'survived'

FIGURES_DIR = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/03_reports/figures'
MODELS_DIR = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/04_models'


def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
               random_state: int = 42):
    """Split into train/val/test sets (70/15/15)."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # First split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Then split train and val from remaining
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_state, stratify=y_trainval
    )
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Standard scale all splits using train statistics."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_val_sc, X_test_sc, scaler


def evaluate_model(model, X, y, label: str = '') -> dict:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob),
    }
    if label:
        print(f"  [{label}] Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")
    return metrics


def train_all_models(X_train, y_train, X_val, y_val, X_train_sc, X_val_sc):
    """Train LR, RF, GB; return results dict and fitted models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                                 random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                         learning_rate=0.05, random_state=42),
    }

    results = {}
    fitted = {}

    # Logistic Regression uses scaled features
    print("\n--- Logistic Regression ---")
    lr = models['Logistic Regression']
    lr.fit(X_train_sc, y_train)
    train_m = evaluate_model(lr, X_train_sc, y_train, 'Train')
    val_m = evaluate_model(lr, X_val_sc, y_val, 'Val')
    results['Logistic Regression'] = {'train': train_m, 'val': val_m}
    fitted['Logistic Regression'] = lr

    # Tree-based models use unscaled features
    for name in ['Random Forest', 'Gradient Boosting']:
        print(f"\n--- {name} ---")
        m = models[name]
        m.fit(X_train, y_train)
        train_m = evaluate_model(m, X_train, y_train, 'Train')
        val_m = evaluate_model(m, X_val, y_val, 'Val')
        results[name] = {'train': train_m, 'val': val_m}
        fitted[name] = m

    return results, fitted


def select_best_model(results: dict) -> str:
    """Return the model name with highest validation accuracy."""
    best = max(results, key=lambda k: results[k]['val']['accuracy'])
    print(f"\nBest model by validation accuracy: {best} "
          f"(val acc = {results[best]['val']['accuracy']:.4f})")
    return best


def plot_feature_importance(model, feature_names: list, save_path: str,
                             model_name: str = 'Best Model') -> pd.DataFrame:
    """Plot and save top-5 feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model has no feature_importances_ or coef_ attribute.")

    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    top5 = fi_df.head(5)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.barh(top5['Feature'][::-1], top5['Importance'][::-1],
                   color=colors[::-1], edgecolor='white')
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top 5 Feature Importances — {model_name}')
    for bar, val in zip(bars, top5['Importance'][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved: {save_path}")
    return fi_df


def plot_roc_curve(model, X_test, y_test, save_path: str, model_name: str = '') -> None:
    """Plot and save ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title(f'ROC Curve — {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {save_path}")


def plot_model_comparison(results: dict, save_path: str) -> None:
    """Grouped bar chart comparing validation metrics across models."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name]['val'][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Validation Set Metrics')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved: {save_path}")


def plot_confusion_matrix(model, X_test, y_test, save_path: str, model_name: str = '') -> None:
    """Plot and save confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: No', 'Predicted: Yes'])
    ax.set_yticklabels(['Actual: No', 'Actual: Yes'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def save_model(model, scaler, path_prefix: str) -> None:
    """Save the best model and scaler to disk."""
    with open(f'{path_prefix}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{path_prefix}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved: {path_prefix}_model.pkl")
    print(f"Scaler saved: {path_prefix}_scaler.pkl")
