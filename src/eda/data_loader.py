"""
Data loading utilities
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str = None, delimiter: str = None, config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Load the bank marketing dataset.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. If None, uses config file.
    delimiter : str, optional
        Delimiter for CSV file. If None, uses config file.
    config_path : str
        Path to configuration file.
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if data_path is None or delimiter is None:
        config = load_config(config_path)
        data_path = data_path or config['data']['raw']
        delimiter = delimiter or config['data']['delimiter']
    
    # Ensure path is relative to project root
    if not Path(data_path).is_absolute():
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / data_path
    
    df = pd.read_csv(data_path, delimiter=delimiter)
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    
    Returns:
    --------
    dict
        Dictionary with dataset information
    """
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }

