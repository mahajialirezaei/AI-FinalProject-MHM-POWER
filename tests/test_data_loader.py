"""
Tests for data_loader module
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eda.data_loader import load_data, get_data_info


def test_load_data():
    """Test that data loads correctly."""
    # This test requires actual data file, skip if not available
    try:
        df = load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "y" in df.columns
    except FileNotFoundError:
        pytest.skip("Data file not found - skipping test")


def test_get_data_info():
    """Test data info extraction."""
    # This test requires actual data file, skip if not available
    try:
        df = load_data()
        info = get_data_info(df)

        assert "shape" in info
        assert "columns" in info
        assert "dtypes" in info
        assert isinstance(info["shape"], tuple)
        assert len(info["shape"]) == 2
    except FileNotFoundError:
        pytest.skip("Data file not found - skipping test")
