"""
Main script for running comprehensive EDA on Bank Marketing Dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eda.data_loader import load_data, get_data_info  # noqa: E402
from src.eda.visualizations import (  # noqa: E402
    plot_class_imbalance,
    plot_categorical_conversion,
    plot_numerical_analysis,
    plot_correlation_heatmap,
    plot_seasonality,
    plot_duration_analysis,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_data_leakage_warning():
    """Print data leakage warning."""
    print_section("DATA LEAKAGE WARNING - Duration Variable")
    print("""
[WARNING] IMPORTANT: The 'duration' variable may cause DATA LEAKAGE in a real-world scenario.

REASON:
- Duration is the length of the last contact call (in seconds)
- This information is only available AFTER the call has been completed
- In a real-world prediction scenario, we need to predict BEFORE making the call
- Therefore, duration should NOT be used as a feature for model training
- Including it would give the model information it wouldn't have in production

RECOMMENDATION:
- Exclude 'duration' from feature engineering for the classification model
- Use it only for post-campaign analysis and understanding call patterns
""")


def main():
    """Main function to run EDA."""
    config_path = "config/config.yaml"

    # 1. Data Loading
    print_section("1. LOADING DATA")
    df = load_data(config_path=config_path)

    data_info = get_data_info(df)
    print(f"Dataset shape: {data_info['shape']}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nBasic statistics:")
    print(df.describe())

    # 2. Class Imbalance Analysis
    print_section("2. CLASS IMBALANCE ANALYSIS")
    target_counts, target_percentages = plot_class_imbalance(df, config_path=config_path)
    print("\nTarget variable distribution:")
    print(target_counts)
    print("\nPercentages:")
    print(target_percentages)

    # 3. Categorical Conversion Rate Analysis
    print_section("3. CATEGORICAL CONVERSION RATE ANALYSIS")
    plot_categorical_conversion(df, config_path=config_path)

    # 4. Numerical Variables & Outliers Analysis
    print_section("4. NUMERICAL VARIABLES & OUTLIERS ANALYSIS")
    plot_numerical_analysis(df, config_path=config_path)

    # 5. Correlation Heatmap
    print_section("5. CORRELATION HEATMAP")
    plot_correlation_heatmap(df, config_path=config_path)

    # 6. Seasonality Analysis
    print_section("6. SEASONALITY ANALYSIS")
    plot_seasonality(df, config_path=config_path)

    # 7. Duration Analysis
    print_section("7. DURATION ANALYSIS")
    plot_duration_analysis(df, config_path=config_path)

    # Data Leakage Warning
    print_data_leakage_warning()

    # Summary
    print_section("EDA COMPLETE - SUMMARY")
    print(f"\n[OK] Dataset loaded: {data_info['shape'][0]} rows, {data_info['shape'][1]} columns")
    print("[OK] Class imbalance analysis completed")
    print("[OK] Categorical conversion rates analyzed")
    print("[OK] Numerical variables and outliers examined")
    print("[OK] Correlation heatmap generated")
    print("[OK] Seasonality trends identified")
    print("[OK] Duration analysis completed with data leakage warning")
    print("\nAll visualizations saved successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
