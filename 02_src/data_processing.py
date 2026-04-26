"""
Data processing functions for Titanic survival prediction.
Author: data-engineer agent
"""

import pandas as pd
import numpy as np
import seaborn as sns


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load Titanic dataset from seaborn and save to raw_path (immutable after save)."""
    df = sns.load_dataset('titanic')
    df.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path} — shape: {df.shape}")
    return df


def derive_title_from_who(who: str, sex: str) -> int:
    """
    Derive a Title integer code from the seaborn 'who' column.
    The seaborn dataset does not include a Name column, so we use
    'who' (man / woman / child) combined with 'sex' as a proxy.
      1 = Mr    (adult male)
      2 = Miss  (female child)
      3 = Mrs   (adult female)
      4 = Master (male child)
      5 = Rare  (fallback)
    """
    if who == 'man':
        return 1   # Mr
    elif who == 'woman':
        return 3   # Mrs / Miss — adult female
    elif who == 'child':
        if sex == 'female' or sex == 1:
            return 2   # Miss
        else:
            return 4   # Master
    else:
        return 5   # Rare


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full cleaning and feature engineering on raw Titanic DataFrame.
    Returns a cleaned DataFrame ready for modelling.
    """
    df = df.copy()

    # Drop rows where target is missing
    df.dropna(subset=['survived'], inplace=True)

    # --- Feature Engineering (before dropping helper columns) ---
    # Title proxy from 'who' + 'sex'
    df['Title'] = df.apply(
        lambda r: derive_title_from_who(r['who'], r['sex']), axis=1
    )

    # Family features
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # --- Handle Missing Values ---
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['fare'] = df['fare'].fillna(df['fare'].median())

    # --- Encode Categoricals ---
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # --- Drop Redundant / Irrelevant Columns ---
    drop_cols = ['who', 'adult_male', 'embark_town', 'alive', 'alone', 'class', 'deck']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Ensure no nulls remain in key columns
    key_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch',
                'fare', 'embarked', 'FamilySize', 'IsAlone', 'Title']
    df.dropna(subset=[c for c in key_cols if c in df.columns], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def save_processed(df: pd.DataFrame, processed_path: str) -> None:
    """Save cleaned DataFrame to parquet."""
    df.to_parquet(processed_path, index=False)
    print(f"Processed data saved to {processed_path} — shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
