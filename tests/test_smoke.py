import pytest
import os
import yaml
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path to ensure imports work correctly during testing
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import actual application logic
from src.eda.data_loader import load_config
from src.preprocessing.main import load_and_preprocess_data

# ==========================================
# CONSTANTS & MOCK DATA
# ==========================================

# Updated mock data with enough samples for stratified splitting (at least 2-3 of each class)
MOCK_CSV_CONTENT = """age;job;marital;education;default;balance;housing;loan;contact;day;month;duration;campaign;pdays;previous;poutcome;y
58;management;married;tertiary;no;2143;yes;no;cellular;5;may;261;1;-1;0;unknown;no
44;technician;single;secondary;no;29;yes;no;cellular;5;may;151;1;-1;0;unknown;no
33;entrepreneur;married;secondary;no;2;yes;yes;cellular;5;may;76;1;-1;0;failure;yes
35;management;married;tertiary;no;231;yes;no;cellular;5;may;120;1;-1;0;success;yes
28;blue-collar;single;secondary;no;447;yes;yes;telephone;5;may;80;1;-1;0;unknown;no
42;entrepreneur;divorced;tertiary;yes;2;yes;no;cellular;5;may;380;1;-1;0;other;yes
50;management;married;tertiary;no;100;no;no;cellular;5;may;100;1;-1;0;unknown;no
30;technician;single;secondary;no;500;no;no;cellular;5;may;200;1;-1;0;failure;yes
40;admin;married;secondary;no;1000;yes;no;telephone;5;may;300;1;-1;0;unknown;no
45;blue-collar;married;primary;no;50;yes;no;cellular;5;may;150;1;-1;0;success;yes
"""


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope="session")
def project_config():
    """
    Fixture to load the real project configuration.
    Fails immediately if config.yaml is missing or invalid.
    """
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        pytest.fail(f"Critical: Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_raw_data(tmp_path):
    """
    Fixture to create a temporary raw CSV file with valid structure.
    Returns the path to the temporary file.
    """
    data_dir = tmp_path / "data" / "raw" / "bank"
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "bank-full.csv"
    with open(file_path, "w") as f:
        f.write(MOCK_CSV_CONTENT)

    return file_path


@pytest.fixture
def mock_processed_dir(tmp_path):
    """Fixture to provide a temporary directory for processed output."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


# ==========================================
# SMOKE TESTS
# ==========================================

@pytest.mark.smoke
def test_config_integrity(project_config):
    """
    SMOKE TEST 1: Configuration Validity.
    """
    required_keys = ["data", "output", "eda", "plotting"]
    missing_keys = [key for key in required_keys if key not in project_config]
    assert not missing_keys, f"Config is missing critical sections: {missing_keys}"
    assert "raw" in project_config["data"], "Config missing 'data.raw' path"


@pytest.mark.smoke
def test_raw_data_availability():
    """
    SMOKE TEST 2: Production Data Check.
    Checks if the REAL dataset exists in the project structure.
    """
    real_data_path = PROJECT_ROOT / "data" / "raw" / "bank" / "bank-full.csv"
    if not real_data_path.exists():
        warnings.warn(f"Real dataset not found at {real_data_path}. Ensure it is downloaded.", UserWarning)
    else:
        assert real_data_path.stat().st_size > 0, "Real dataset file is empty!"


@pytest.mark.smoke
def test_preprocessing_pipeline_logic(mock_raw_data, mock_processed_dir):
    """
    SMOKE TEST 3: Preprocessing Pipeline Sanity.
    Runs the ACTUAL `load_and_preprocess_data` function on mock data.
    """
    try:
        train_df, val_df, test_df = load_and_preprocess_data(
            str(mock_raw_data),
            str(mock_processed_dir)
        )
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert (mock_processed_dir / "train.csv").exists()
        assert "target" in train_df.columns, "Target column missing in processed data"
    except Exception as e:
        pytest.fail(f"Preprocessing pipeline crashed: {str(e)}")


@pytest.mark.smoke
def test_eda_functions_execution(mock_raw_data):
    """
    SMOKE TEST 4: EDA Visualization Sanity.
    Actually EXECUTES the visualization functions on mock data to catch runtime errors
    like the 'tick_labels' issue in matplotlib.
    """
    from src.eda.visualizations import plot_duration_analysis, plot_class_imbalance

    # 1. Load mock data into a DataFrame
    # Note: We must use the same delimiter as in the mock data (semicolon)
    df = pd.read_csv(mock_raw_data, sep=';')

    # 2. Run the function that previously failed
    try:
        # We pass save=False to avoid creating files during testing, focusing only on logic
        # If save=True was needed, we would need to mock the config path too.
        plot_duration_analysis(df, save=False)

        # Also test another one to be safe
        plot_class_imbalance(df, save=False)

    except TypeError as e:
        pytest.fail(f"EDA function failed with TypeError (likely API mismatch): {e}")
    except Exception as e:
        pytest.fail(f"EDA function crashed: {e}")
    finally:
        # Close any plots to avoid memory leaks during tests
        plt.close('all')