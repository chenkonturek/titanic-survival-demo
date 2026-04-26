"""
Visualization helper functions for Titanic EDA.
Author: data-analyst-eda agent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

FIGURES_DIR = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/03_reports/figures'

# Consistent style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = ['#d9534f', '#5bc0de']  # red=died, blue=survived


def plot_survival_by_gender(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart: survival rate by sex."""
    rates = df.groupby('sex')['survived'].mean().reset_index()
    rates['sex_label'] = rates['sex'].map({0: 'Male', 1: 'Female'})

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rates['sex_label'], rates['survived'] * 100,
                  color=['#5b9bd5', '#e07b8f'], edgecolor='white', width=0.5)
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Survival Rate by Gender')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, rates['survived'] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_survival_by_class(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart: survival rate by passenger class."""
    rates = df.groupby('pclass')['survived'].mean().reset_index()
    labels = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
    rates['class_label'] = rates['pclass'].map(labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rates['class_label'], rates['survived'] * 100,
                  color=['#2ca02c', '#ff7f0e', '#d62728'], edgecolor='white', width=0.5)
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Survival Rate by Passenger Class')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, rates['survived'] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_age_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Overlapping histogram: age distribution by survival."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for survived, label, color in [(0, 'Did Not Survive', '#d9534f'), (1, 'Survived', '#5bc0de')]:
        subset = df[df['survived'] == survived]['age']
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title('Age Distribution by Survival')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_survival_by_family_size(df: pd.DataFrame, save_path: str) -> None:
    """Line + bar chart: survival rate by family size."""
    rates = df.groupby('FamilySize')['survived'].agg(['mean', 'count']).reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.bar(rates['FamilySize'], rates['count'], alpha=0.4, color='#aec7e8', label='Count')
    ax2.plot(rates['FamilySize'], rates['mean'] * 100, 'o-', color='#1f77b4',
             linewidth=2, markersize=7, label='Survival Rate')
    ax1.set_xlabel('Family Size')
    ax1.set_ylabel('Passenger Count', color='#aec7e8')
    ax2.set_ylabel('Survival Rate (%)', color='#1f77b4')
    ax2.set_ylim(0, 100)
    plt.title('Survival Rate by Family Size')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fare_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Box plot: fare distribution by survival."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data_survived = df[df['survived'] == 1]['fare']
    data_died = df[df['survived'] == 0]['fare']
    ax.boxplot([data_died, data_survived], labels=['Did Not Survive', 'Survived'],
               patch_artist=True,
               boxprops=dict(facecolor='#d9d9d9'),
               medianprops=dict(color='#d62728', linewidth=2))
    ax.set_ylabel('Fare (£)')
    ax.set_title('Fare Distribution by Survival')
    ax.set_ylim(0, df['fare'].quantile(0.97))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """Correlation heatmap of all features."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, ax=ax, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_survival_by_title(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart: survival rate by Title."""
    title_labels = {1: 'Mr', 2: 'Miss', 3: 'Mrs', 4: 'Master', 5: 'Rare'}
    temp = df.copy()
    temp['Title_label'] = temp['Title'].map(title_labels)
    rates = temp.groupby('Title_label')['survived'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(rates.index, rates.values * 100, color='#6baed6', edgecolor='white', width=0.5)
    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Survival Rate by Title')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, rates.values * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
