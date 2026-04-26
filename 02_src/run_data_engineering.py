"""
Run data engineering pipeline for Titanic survival prediction.
"""
import sys
sys.path.insert(0, '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/02_src')

from data_processing import load_raw_data, clean_and_engineer, save_processed

RAW_PATH = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/00_data/raw/titanic_raw.csv'
PROCESSED_PATH = '/Users/chen/PycharmProjects/claude-demo/titanic-survival-prediction/00_data/processed/titanic_cleaned.parquet'

if __name__ == '__main__':
    print("=== Phase 1: Data Engineering ===")
    raw_df = load_raw_data(RAW_PATH)
    cleaned_df = clean_and_engineer(raw_df)
    save_processed(cleaned_df, PROCESSED_PATH)
    print("\n=== Data Engineering Complete ===")
    print(f"Final shape: {cleaned_df.shape}")
    print(f"Final columns: {list(cleaned_df.columns)}")
    print(f"\nMissing values per column:\n{cleaned_df.isnull().sum()}")
    print(f"\nSurvival rate: {cleaned_df['survived'].mean():.3f}")
