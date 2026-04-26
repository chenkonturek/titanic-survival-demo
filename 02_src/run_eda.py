"""
Run EDA pipeline for Titanic survival prediction.
Generates all figures and prints insights summary.
"""
import sys
sys.path.insert(0, '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/02_src')

import pandas as pd
import numpy as np
from visualization import (
    plot_survival_by_gender,
    plot_survival_by_class,
    plot_age_distribution,
    plot_survival_by_family_size,
    plot_fare_distribution,
    plot_correlation_heatmap,
    plot_survival_by_title,
)

PROCESSED_PATH = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/00_data/processed/titanic_cleaned.parquet'
FIGURES_DIR = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/03_reports/figures'

if __name__ == '__main__':
    print("=== Phase 2: Exploratory Data Analysis ===\n")

    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded cleaned data: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # --- Basic Stats ---
    print("--- Survival Overview ---")
    total = len(df)
    survived = df['survived'].sum()
    print(f"Total passengers: {total}")
    print(f"Survived: {survived} ({survived/total*100:.1f}%)")
    print(f"Did not survive: {total - survived} ({(total-survived)/total*100:.1f}%)\n")

    # --- By Gender ---
    print("--- Survival by Gender ---")
    g = df.groupby('sex')['survived'].agg(['sum', 'count', 'mean'])
    g.index = g.index.map({0: 'Male', 1: 'Female'})
    g.columns = ['Survived', 'Total', 'Rate']
    print(g.to_string())
    print()

    # --- By Class ---
    print("--- Survival by Passenger Class ---")
    c = df.groupby('pclass')['survived'].agg(['sum', 'count', 'mean'])
    c.index = c.index.map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    c.columns = ['Survived', 'Total', 'Rate']
    print(c.to_string())
    print()

    # --- Age Stats ---
    print("--- Age Statistics ---")
    print(f"Overall median age: {df['age'].median():.1f}")
    for s, label in [(1, 'Survivors'), (0, 'Non-survivors')]:
        med = df[df['survived'] == s]['age'].median()
        print(f"  {label} median age: {med:.1f}")
    print()

    # --- By Family Size ---
    print("--- Survival by Family Size (IsAlone) ---")
    iso = df.groupby('IsAlone')['survived'].agg(['sum', 'count', 'mean'])
    iso.index = iso.index.map({1: 'Alone', 0: 'With Family'})
    iso.columns = ['Survived', 'Total', 'Rate']
    print(iso.to_string())
    print()

    # --- By Title ---
    print("--- Survival by Title ---")
    title_map = {1: 'Mr', 2: 'Miss', 3: 'Mrs', 4: 'Master', 5: 'Rare'}
    t = df.copy()
    t['Title_label'] = t['Title'].map(title_map)
    tt = t.groupby('Title_label')['survived'].agg(['sum', 'count', 'mean'])
    tt.columns = ['Survived', 'Total', 'Rate']
    print(tt.sort_values('Rate', ascending=False).to_string())
    print()

    # --- Fare ---
    print("--- Fare Statistics ---")
    for s, label in [(1, 'Survivors'), (0, 'Non-survivors')]:
        med = df[df['survived'] == s]['fare'].median()
        print(f"  {label} median fare: £{med:.2f}")
    print()

    # --- Generate Figures ---
    print("--- Generating Figures ---")
    plot_survival_by_gender(df, f'{FIGURES_DIR}/survival_by_gender.png')
    plot_survival_by_class(df, f'{FIGURES_DIR}/survival_by_class.png')
    plot_age_distribution(df, f'{FIGURES_DIR}/age_distribution.png')
    plot_survival_by_family_size(df, f'{FIGURES_DIR}/survival_by_family_size.png')
    plot_fare_distribution(df, f'{FIGURES_DIR}/fare_distribution.png')
    plot_correlation_heatmap(df, f'{FIGURES_DIR}/correlation_heatmap.png')
    plot_survival_by_title(df, f'{FIGURES_DIR}/survival_by_title.png')

    print("\n=== EDA Complete ===")
