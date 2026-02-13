"""
Bank Marketing Campaign Prediction UI
Streamlit application for predicting term deposit subscription probability
"""

import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import yaml
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress SHAP warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# Fix for SHAP XGBoost base_score parsing issue (XGBoost 3.1.0+ compatibility)
# This patches SHAP's XGBTreeModelLoader to handle base_score stored as string array '[5E-1]'
try:
    if hasattr(shap.explainers._tree, "XGBTreeModelLoader"):
        XGBTreeModelLoader = shap.explainers._tree.XGBTreeModelLoader

        # Store original __init__ if not already patched
        if not hasattr(XGBTreeModelLoader, "_original_init"):
            XGBTreeModelLoader._original_init = XGBTreeModelLoader.__init__

            def patched_xgb_init(self, model):
                """Patched XGBTreeModelLoader init to fix base_score parsing.

                Fixes XGBoost 3.1.0+ compatibility issue.
                """
                # The issue: XGBoost 3.1.0+ stores base_score as '[5E-1]' instead of '0.5'
                # SHAP's code at line 2104 does:
                #   self.base_score = float(learner_model_param["base_score"])
                # This fails because float('[5E-1]') raises ValueError
                # Solution: Fix the booster config BEFORE calling original_init,
                # ensuring the fix persists

                import xgboost as xgb
                import json as json_lib
                import tempfile
                import os

                # Get the booster
                if isinstance(model, xgb.core.Booster):
                    booster = model
                elif hasattr(model, "get_booster"):
                    booster = model.get_booster()
                else:
                    # Fallback: try original init
                    XGBTreeModelLoader._original_init(self, model)
                    return

                # Check and fix base_score BEFORE SHAP reads it
                config = booster.save_config()
                config_dict = json_lib.loads(config)

                base_score_needs_fix = False
                base_score_float = None

                if "learner" in config_dict and "learner_model_param" in config_dict["learner"]:
                    base_score_param = config_dict["learner"]["learner_model_param"].get(
                        "base_score", None
                    )
                    if isinstance(base_score_param, str) and base_score_param.startswith("["):
                        # Fix it: '[5E-1]' -> '0.5'
                        base_score_str = base_score_param.strip("[]")
                        base_score_float = float(base_score_str)
                        config_dict["learner"]["learner_model_param"]["base_score"] = str(
                            base_score_float
                        )
                        base_score_needs_fix = True

                # If base_score needs fixing, create a new booster with the fixed config
                if base_score_needs_fix:
                    # Save the model (trees) to a temp file
                    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as model_file:
                        booster.save_model(model_file.name)
                        model_path = model_file.name

                    try:
                        # Create a completely new booster
                        new_booster = xgb.Booster()

                        # Load the model first (this loads the tree structure)
                        new_booster.load_model(model_path)

                        # Now load the FIXED config - this should set base_score correctly
                        # The key is to load config AFTER loading the model
                        new_booster.load_config(json_lib.dumps(config_dict))

                        # Verify the fix was applied
                        verify_config = new_booster.save_config()
                        verify_dict = json_lib.loads(verify_config)
                        verify_base = (
                            verify_dict.get("learner", {})
                            .get("learner_model_param", {})
                            .get("base_score", "")
                        )

                        # If verification shows it's still wrong, force reload config again
                        if isinstance(verify_base, str) and verify_base.startswith("["):
                            # Config was overwritten by model load - reload it
                            new_booster.load_config(json_lib.dumps(config_dict))
                            # Verify again
                            verify_config2 = new_booster.save_config()
                            verify_dict2 = json_lib.loads(verify_config2)
                            verify_base2 = (
                                verify_dict2.get("learner", {})
                                .get("learner_model_param", {})
                                .get("base_score", "")
                            )

                            # If still wrong, patch save_config to always return fixed config
                            if isinstance(verify_base2, str) and verify_base2.startswith("["):
                                original_save = new_booster.save_config

                                def fixed_save_config():
                                    cfg = json_lib.loads(original_save())
                                    if "learner" in cfg and "learner_model_param" in cfg["learner"]:
                                        cfg["learner"]["learner_model_param"]["base_score"] = str(
                                            base_score_float
                                        )
                                    return json_lib.dumps(cfg)

                                new_booster.save_config = fixed_save_config

                        # Use the fixed booster
                        booster = new_booster
                    finally:
                        try:
                            os.unlink(model_path)
                        except Exception:
                            pass

                # As a final safety net, patch save_config to always return fixed config
                # This ensures that even if SHAP reads config multiple times,
                # it gets the fixed version
                original_save_config_restore = None
                if base_score_needs_fix and base_score_float is not None:
                    original_save_config_restore = booster.save_config

                    def patched_save_config():
                        """Patched save_config that always returns fixed base_score."""
                        config_str = original_save_config_restore()
                        config_dict = json_lib.loads(config_str)
                        if (
                            "learner" in config_dict
                            and "learner_model_param" in config_dict["learner"]
                        ):
                            base_score_param = config_dict["learner"]["learner_model_param"].get(
                                "base_score", None
                            )
                            if isinstance(base_score_param, str) and base_score_param.startswith(
                                "["
                            ):
                                config_dict["learner"]["learner_model_param"]["base_score"] = str(
                                    base_score_float
                                )
                        return json_lib.dumps(config_dict)

                    booster.save_config = patched_save_config

                # Now call original init with the (possibly fixed) booster
                try:
                    XGBTreeModelLoader._original_init(self, booster)
                finally:
                    # Restore original save_config if we patched it
                    if original_save_config_restore is not None:
                        booster.save_config = original_save_config_restore

            XGBTreeModelLoader.__init__ = patched_xgb_init
except Exception:
    # If patching fails, continue anyway - will try other fixes
    pass

# Fix for LogisticRegression multi_class attribute error
# This patches LogisticRegression to handle missing multi_class attribute
# which can occur due to scikit-learn version mismatches
try:
    from sklearn.linear_model import LogisticRegression

    # Store original methods if not already patched
    if not hasattr(LogisticRegression, "_original_getattribute"):
        LogisticRegression._original_getattribute = LogisticRegression.__getattribute__

        def _patched_getattribute(self, name):
            """Patched __getattribute__ to handle missing multi_class attribute."""
            if name == "multi_class":
                # Check if attribute exists in instance dict
                if "multi_class" not in self.__dict__:
                    # Set default value for binary classification
                    self.__dict__["multi_class"] = "auto"
            # Call original getattribute
            return LogisticRegression._original_getattribute(self, name)

        LogisticRegression.__getattribute__ = _patched_getattribute

        # Also patch __setstate__ to handle unpickling
        if hasattr(LogisticRegression, "__setstate__"):
            LogisticRegression._original_setstate = LogisticRegression.__setstate__

            def _patched_setstate(self, state):
                """Patched __setstate__ to ensure multi_class exists after unpickling."""
                # Call original setstate
                LogisticRegression._original_setstate(self, state)
                # Ensure multi_class exists
                if "multi_class" not in self.__dict__:
                    self.__dict__["multi_class"] = state.get("multi_class", "auto")

            LogisticRegression.__setstate__ = _patched_setstate
except Exception:
    # If patching fails, continue anyway - models might work without it
    pass

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565a0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "models"
PREPROCESSOR_PATH = PROJECT_ROOT / "src" / "models" / "preprocessor.pkl"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_resource
def load_config():
    """Load configuration file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            # Ensure thresholds dictionary exists
            if "model" not in config:
                config["model"] = {}
            if "thresholds" not in config["model"]:
                config["model"]["thresholds"] = {}
            return config
    except Exception as e:
        st.warning(f"Could not load config: {e}")
        return {"model": {"threshold": 0.5, "thresholds": {}}}


@st.cache_resource
def load_preprocessor():
    """Load the preprocessor."""
    try:
        # Try joblib first (preprocessor is saved with pickle but might work with joblib)
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            return preprocessor
        except Exception:
            # Fallback to pickle
            with open(PREPROCESSOR_PATH, "rb") as f:
                preprocessor = pickle.load(f)
            return preprocessor
    except FileNotFoundError:
        st.error(f"❌ Preprocessor not found at {PREPROCESSOR_PATH}")
        st.info("Please run preprocessing first: `python -m src.preprocessing.main`")
        return None
    except Exception as e:
        st.error(f"❌ Error loading preprocessor: {e}")
        return None


def _safe_load_model(model_path):
    """Safely load a model, handling LogisticRegression multi_class attribute errors."""
    from sklearn.linear_model import LogisticRegression

    try:
        # First try normal loading
        return joblib.load(model_path)
    except AttributeError as e:
        if "multi_class" in str(e):
            # Error during unpickling - try with patched LogisticRegression
            # Temporarily patch __setstate__ to handle missing multi_class
            original_setstate = None
            if hasattr(LogisticRegression, "__setstate__"):
                original_setstate = LogisticRegression.__setstate__

                def safe_setstate(self, state):
                    """Safe setstate that adds multi_class if missing."""
                    # Ensure multi_class is in state
                    if "multi_class" not in state:
                        state["multi_class"] = "auto"
                    # Call original setstate
                    if original_setstate:
                        original_setstate(self, state)
                    # Double-check multi_class exists
                    if not hasattr(self, "multi_class"):
                        self.multi_class = "auto"

                LogisticRegression.__setstate__ = safe_setstate

            try:
                model = joblib.load(model_path)
                # Restore original setstate
                if original_setstate:
                    LogisticRegression.__setstate__ = original_setstate
                return model
            except Exception:
                # Restore original setstate before re-raising
                if original_setstate:
                    LogisticRegression.__setstate__ = original_setstate
                raise
        else:
            raise


def _patch_logistic_regression_model(model):
    """Patch LogisticRegression model to add multi_class attribute if missing.

    This fixes compatibility issues with scikit-learn versions where multi_class
    is a parameter but code tries to access it as an attribute.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Handle standalone LogisticRegression
    if isinstance(model, LogisticRegression):
        if not hasattr(model, "multi_class"):
            # Set default value - 'auto' is the default for binary classification
            try:
                model.multi_class = "auto"
            except Exception:
                # If direct assignment fails, try via __dict__
                try:
                    model.__dict__["multi_class"] = "auto"
                except Exception:
                    pass
    # Handle pipelines that might contain LogisticRegression
    elif isinstance(model, Pipeline):
        for step_name, step_obj in model.steps:
            if isinstance(step_obj, LogisticRegression):
                if not hasattr(step_obj, "multi_class"):
                    try:
                        step_obj.multi_class = "auto"
                    except Exception:
                        try:
                            step_obj.__dict__["multi_class"] = "auto"
                        except Exception:
                            pass

    return model


@st.cache_resource
def load_models(config_dict):
    """Load all available models with their thresholds."""
    models = {}
    model_configs = {
        "baseline_logreg": {
            "file": "baseline_logreg.pkl",
            "name": "Baseline Logistic Regression",
            "description": "Phase 1 baseline model",
            "is_pipeline": False,
        },
        "random_forest_model_smote": {
            "file": "random_forest_model_smote.pkl",
            "name": "Random Forest (SMOTE)",
            "description": "Random Forest with SMOTE oversampling",
            "is_pipeline": False,
        },
        "xgboost_model_smote": {
            "file": "xgboost_model_smote.pkl",
            "name": "XGBoost (SMOTE)",
            "description": "XGBoost with SMOTE pipeline",
            "is_pipeline": True,
        },
        "xgboost_weighted": {
            "file": "xgboost_weighted.pkl",
            "name": "XGBoost (Weighted)",
            "description": "XGBoost with class weighting",
            "is_pipeline": False,
        },
        "xgboost_optimized": {
            "file": "xgboost_optimized.pkl",
            "name": "XGBoost Optimized (SMOTE)",
            "description": "Optuna-optimized XGBoost with SMOTE",
            "is_pipeline": True,
        },
        "xgboost_weighted_optimized": {
            "file": "xgboost_weighted_optimized.pkl",
            "name": "Champion Model ⭐",
            "description": "Production-ready optimized weighted XGBoost",
            "is_pipeline": False,
        },
    }

    # Get thresholds from config
    thresholds_dict = config_dict.get("model", {}).get("thresholds", {})
    default_threshold = config_dict.get("model", {}).get("threshold", 0.5)

    for model_key, model_config in model_configs.items():
        model_path = MODELS_DIR / model_config["file"]
        try:
            # Use safe loading function that handles multi_class AttributeError
            model = _safe_load_model(model_path)

            # Additional patch for LogisticRegression models (defensive)
            model = _patch_logistic_regression_model(model)

            # Get model-specific threshold, fallback to default
            model_threshold = thresholds_dict.get(model_key, default_threshold)
            models[model_key] = {
                "model": model,
                "name": model_config["name"],
                "description": model_config["description"],
                "is_pipeline": model_config["is_pipeline"],
                "threshold": model_threshold,
            }
        except FileNotFoundError:
            st.sidebar.warning(f"⚠️ {model_config['name']} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {model_config['name']}: {e}")

    return models


def predict_with_model(model_obj, processed_input, is_pipeline=False):
    """Make prediction with a model or pipeline.

    Note: Both pipelines and regular models expect preprocessed input
    because training data (train.csv) is already preprocessed.
    """
    try:
        # Both pipelines and models use preprocessed input
        # Pipelines include SMOTE but are trained on preprocessed data

        # Handle LogisticRegression multi_class attribute error
        # This can occur due to scikit-learn version mismatches
        from sklearn.linear_model import LogisticRegression

        if isinstance(model_obj, LogisticRegression) and not hasattr(model_obj, "multi_class"):
            params = model_obj.get_params()
            model_obj.multi_class = params.get("multi_class", "auto")

        try:
            prob = model_obj.predict_proba(processed_input)[0][1]
        except AttributeError as ae:
            # Catch AttributeError specifically for multi_class and retry after patching
            if "multi_class" in str(ae) and isinstance(model_obj, LogisticRegression):
                params = model_obj.get_params()
                model_obj.multi_class = params.get("multi_class", "auto")
                prob = model_obj.predict_proba(processed_input)[0][1]
            else:
                raise
        return prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


@st.cache_resource
def load_background_data(n_samples=100):
    """Load background dataset for SHAP explainer (sampled from test set)."""
    try:
        test_path = DATA_PROCESSED_DIR / "test.csv"
        if not test_path.exists():
            return None

        test_df = pd.read_csv(test_path)

        # Check if target column exists
        if "target" not in test_df.columns:
            return None

        X_test = test_df.drop(columns=["target"])

        # Validate data
        if X_test.empty:
            return None

        # Sample background data (SHAP works better with smaller background sets)
        if len(X_test) > n_samples:
            background_data = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)
        else:
            background_data = X_test

        return background_data.values
    except FileNotFoundError:
        return None
    except Exception:
        # Silently return None - error will be handled in UI
        return None


@st.cache_resource
def load_shap_explainer(_model, background_data):
    """Create and cache SHAP explainer for the champion model.

    Note: _model is prefixed with underscore to exclude it from Streamlit's hashing,
    since XGBoost models are not hashable. The explainer is cached based on background_data.
    """
    # Check if model is a tree-based model (XGBoost, LightGBM, etc.)
    # Also check if it's a pipeline that contains a tree model
    from sklearn.pipeline import Pipeline

    model_to_use = _model
    model_type_str = str(type(_model)).lower()

    # If it's a pipeline, try to extract the tree model
    if isinstance(_model, Pipeline):
        # Look for XGBoost or tree model in pipeline steps
        for step_name, step_obj in _model.steps:
            step_type_str = str(type(step_obj)).lower()
            if any(
                xgb_type in step_type_str
                for xgb_type in [
                    "xgb",
                    "lightgbm",
                    "catboost",
                    "randomforest",
                    "decisiontree",
                    "gradientboosting",
                ]
            ):
                model_to_use = step_obj
                model_type_str = step_type_str
                break

    # More robust XGBoost detection - check for XGBClassifier class name or module
    try:
        import xgboost as xgb

        is_xgboost_model = isinstance(model_to_use, xgb.XGBClassifier) or isinstance(
            model_to_use, xgb.XGBRegressor
        )
    except ImportError:
        is_xgboost_model = False

    is_tree_model = is_xgboost_model or any(
        xgb_type in model_type_str
        for xgb_type in [
            "xgb",
            "lightgbm",
            "catboost",
            "randomforest",
            "decisiontree",
            "gradientboosting",
        ]
    )

    if not is_tree_model:
        return None

    try:
        # For XGBoost models, fix base_score issue and try multiple approaches
        model_for_explainer = model_to_use

        if is_xgboost_model:
            # Fix base_score attribute if it's stored as a string
            # (XGBoost version compatibility issue)
            # This fixes: ValueError: could not convert string to float: '[5E-1]'
            try:
                if hasattr(model_to_use, "get_booster"):
                    booster = model_to_use.get_booster()
                    # Get the base_score from booster config and fix it if needed
                    config = booster.save_config()
                    import json as json_lib

                    config_dict = json_lib.loads(config)

                    # Fix base_score if it's a string array like '[5E-1]'
                    if "learner" in config_dict and "learner_model_param" in config_dict["learner"]:
                        base_score_param = config_dict["learner"]["learner_model_param"].get(
                            "base_score", None
                        )
                        if (
                            base_score_param
                            and isinstance(base_score_param, str)
                            and base_score_param.startswith("[")
                        ):
                            # Parse string like '[5E-1]' to float
                            try:
                                base_score_str = base_score_param.strip("[]")
                                base_score_float = float(base_score_str)
                                # Update the config with proper float string representation
                                config_dict["learner"]["learner_model_param"]["base_score"] = str(
                                    base_score_float
                                )

                                # Load the fixed config into the booster
                                booster.load_config(json_lib.dumps(config_dict))
                            except (ValueError, KeyError, TypeError):
                                # If parsing fails, continue with booster as-is
                                pass

                    # Use the booster (fixed or original)
                    model_for_explainer = booster
            except Exception:
                # If fixing fails, try using booster directly or model itself
                try:
                    if hasattr(model_to_use, "get_booster"):
                        model_for_explainer = model_to_use.get_booster()
                    elif hasattr(model_to_use, "booster_") and model_to_use.booster_ is not None:
                        model_for_explainer = model_to_use.booster_
                    else:
                        model_for_explainer = model_to_use
                except Exception:
                    model_for_explainer = model_to_use

        # Try creating explainer with background data first, fallback to without if it fails
        explainer = None
        last_error = None

        if background_data is not None:
            try:
                # Ensure background_data is numpy array
                if isinstance(background_data, pd.DataFrame):
                    background_data = background_data.values
                # Ensure it's 2D
                if background_data.ndim == 1:
                    background_data = background_data.reshape(1, -1)
                explainer = shap.TreeExplainer(model_for_explainer, background_data)
            except Exception as e:
                last_error = str(e)
                # Fallback: try without background data
                try:
                    explainer = shap.TreeExplainer(model_for_explainer)
                    last_error = None  # Clear error on success
                except Exception as e2:
                    last_error = f"With background: {last_error}; Without background: {str(e2)}"
                    raise Exception(last_error)  # Re-raise with combined error info
        else:
            # No background data, use TreeExplainer without background
            explainer = shap.TreeExplainer(model_for_explainer)

        return explainer
    except AttributeError:
        # Model might not support TreeExplainer
        return None
    except Exception as e:
        # Store error for display
        error_msg = f"{type(e).__name__}: {str(e)}"
        if hasattr(e, "__traceback__"):
            import traceback as tb

            error_msg += f"\n\nTraceback:\n{tb.format_exc()}"
        load_shap_explainer._last_error = error_msg
        return None


def calculate_shap_values(explainer, processed_input):
    """Calculate SHAP values for a single prediction."""
    try:
        # Ensure processed_input is numpy array
        if isinstance(processed_input, pd.DataFrame):
            processed_input = processed_input.values
        elif not isinstance(processed_input, np.ndarray):
            processed_input = np.array(processed_input)

        # Ensure 2D array
        if processed_input.ndim == 1:
            processed_input = processed_input.reshape(1, -1)

        # For TreeExplainer, shap_values returns array of shape (n_samples, n_features)
        shap_values = explainer.shap_values(processed_input)

        # Handle binary classification - get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        expected_value = explainer.expected_value
        # Handle binary classification expected value
        if isinstance(expected_value, np.ndarray) and len(expected_value) > 1:
            expected_value = expected_value[1]  # Positive class

        return shap_values[0], expected_value  # Return first (and only) sample
    except Exception:
        # Return None on any error - will be handled in UI
        return None, None


def plot_shap_waterfall(explainer, shap_values, expected_value, feature_names, processed_input):
    """Create SHAP waterfall plot for local explanation."""
    try:
        # Ensure processed_input is numpy array
        if isinstance(processed_input, pd.DataFrame):
            data_values = processed_input.values[0]
        elif isinstance(processed_input, np.ndarray):
            data_values = processed_input[0] if processed_input.ndim > 1 else processed_input
        else:
            data_values = np.array(processed_input)[0]

        # Ensure shap_values is 1D array
        if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Create SHAP Explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_values.reshape(1, -1),
            base_values=np.array([expected_value]),
            data=data_values.reshape(1, -1),
            feature_names=list(feature_names),
        )

        # Use waterfall plot (best for single prediction)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_explanation[0], show=False)
        plt.tight_layout()
        return fig
    except Exception:
        # Fallback to bar plot if waterfall fails
        try:
            # Ensure processed_input is numpy array
            if isinstance(processed_input, pd.DataFrame):
                data_values = processed_input.values[0]
            elif isinstance(processed_input, np.ndarray):
                data_values = processed_input[0] if processed_input.ndim > 1 else processed_input
            else:
                data_values = np.array(processed_input)[0]

            # Ensure shap_values is 1D array
            if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
                shap_values = shap_values[0]

            shap_explanation = shap.Explanation(
                values=shap_values.reshape(1, -1),
                base_values=np.array([expected_value]),
                data=data_values.reshape(1, -1),
                feature_names=list(feature_names),
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.bar(shap_explanation[0], show=False)
            plt.tight_layout()
            return fig
        except Exception:
            return None


def plot_shap_summary(shap_values_all, feature_names):
    """Create SHAP summary bar plot for global feature importance."""
    try:
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.abs(shap_values_all).mean(axis=0)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        feature_importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": mean_shap})
            .sort_values("Importance", ascending=True)
            .tail(15)
        )  # Top 15 features

        ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"Could not create SHAP summary plot: {e}")
        return None


# Load assets
config = load_config()
preprocessor = load_preprocessor()
models_dict = load_models(config)

# Header
st.markdown(
    '<div class="main-header">🏦 Bank Marketing Campaign Predictor</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">AI-powered prediction system for term deposit '
    "subscription probability</div>",
    unsafe_allow_html=True,
)

# Check if models are loaded
if not models_dict:
    st.error("❌ No models found! Please train models first.")
    st.stop()

if preprocessor is None:
    st.stop()

# Display model loading status in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🤖 Model Status")
    if models_dict:
        st.success(f"✅ {len(models_dict)} model(s) loaded successfully")
        with st.expander("View loaded models"):
            for model_key, model_info in models_dict.items():
                st.write(f"• {model_info['name']}")
    else:
        st.error("❌ No models loaded")

# Sidebar - Customer Information
with st.sidebar:
    st.markdown("## 📋 Customer Information")
    st.markdown("---")

    # Numerical features
    st.markdown("### 📊 Numerical Features")
    age = st.number_input("Age", min_value=18, max_value=95, value=35, help="Customer age")
    balance = st.number_input(
        "Balance", min_value=-8000, max_value=100000, value=2000, help="Account balance"
    )
    day = st.slider(
        "Day of Last Contact",
        min_value=1,
        max_value=31,
        value=15,
        help="Day of the month when last contacted",
    )
    campaign = st.number_input(
        "Campaign Contacts",
        min_value=1,
        max_value=50,
        value=1,
        help="Number of contacts during this campaign",
    )
    pdays = st.number_input(
        "Days Since Last Contact",
        min_value=-1,
        max_value=999,
        value=-1,
        help="-1 means not previously contacted",
    )
    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        max_value=50,
        value=0,
        help="Number of contacts before this campaign",
    )

    st.markdown("---")

    # Categorical features
    st.markdown("### 📝 Categorical Features")
    job = st.selectbox(
        "Job",
        [
            "admin.",
            "unknown",
            "unemployed",
            "management",
            "housemaid",
            "entrepreneur",
            "student",
            "blue-collar",
            "self-employed",
            "retired",
            "technician",
            "services",
        ],
        help="Customer's job type",
    )
    marital = st.selectbox(
        "Marital Status", ["married", "divorced", "single"], help="Marital status"
    )
    education = st.selectbox(
        "Education", ["unknown", "secondary", "primary", "tertiary"], help="Education level"
    )
    default = st.selectbox("Has Default?", ["no", "yes"], help="Has credit in default?")
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"], help="Has housing loan?")
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"], help="Has personal loan?")
    contact = st.selectbox(
        "Contact Type", ["cellular", "telephone", "unknown"], help="Contact communication type"
    )
    month = st.selectbox(
        "Last Contact Month",
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        help="Last contact month",
    )
    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "other", "failure", "success"],
        help="Outcome of previous marketing campaign",
    )

# Prepare input data
user_data = pd.DataFrame(
    [
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        }
    ]
)

# Replace 'unknown' with NaN
user_data.replace("unknown", np.nan, inplace=True)

# Prediction button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button(
        "🚀 Predict with All Models", use_container_width=True, type="primary"
    )

if predict_button:
    try:
        # Process input
        processed_input = preprocessor.transform(user_data)

        # Get predictions from all models
        results = []
        for model_key, model_info in models_dict.items():
            prob = predict_with_model(
                model_info["model"], processed_input, model_info["is_pipeline"]
            )
            if prob is not None:
                results.append(
                    {
                        "Model": model_info["name"],
                        "Probability": prob,
                        "Description": model_info["description"],
                        "Threshold": model_info.get("threshold", 0.5),
                        "ModelKey": model_key,
                    }
                )

        if not results:
            st.error("❌ No predictions could be made. Please check model files.")
            st.stop()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="Probability", ascending=False)

        # Get champion model threshold (for display purposes)
        champion_row = results_df[results_df["Model"].str.contains("Champion", na=False)]
        champion_threshold = champion_row["Threshold"].values[0] if len(champion_row) > 0 else 0.5
        champion_prob = (
            champion_row["Probability"].values[0]
            if len(champion_row) > 0
            else results_df["Probability"].mean()
        )

        # Main results display
        st.markdown("## 📊 Prediction Results")

        # Metrics row
        avg_prob = results_df["Probability"].mean()
        max_prob = results_df["Probability"].max()
        min_prob = results_df["Probability"].min()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Champion Model",
                f"{champion_prob:.1%}",
                delta=(
                    f"{champion_prob - champion_threshold:.1%}"
                    if champion_prob >= champion_threshold
                    else None
                ),
            )
        with col2:
            st.metric("Average Probability", f"{avg_prob:.1%}")
        with col3:
            st.metric("Highest Probability", f"{max_prob:.1%}")
        with col4:
            st.metric("Champion Threshold", f"{champion_threshold:.1%}")

        st.markdown("---")

        # Results visualization
        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            st.markdown("### 🏆 Model Rankings")

            # Create styled dataframe
            display_df = results_df[["Model", "Probability"]].copy()
            display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x:.2%}")

            # Highlight champion model
            def highlight_champion(row):
                if "Champion" in row["Model"]:
                    return ["background-color: #ffd700; font-weight: bold"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_df.style.apply(highlight_champion, axis=1),
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            st.markdown("### 📈 Probability Comparison")

            # Create horizontal bar chart
            fig = px.bar(
                results_df,
                x="Probability",
                y="Model",
                orientation="h",
                color="Probability",
                color_continuous_scale="RdYlGn",
                text="Probability",
                labels={"Probability": "Subscription Probability", "Model": ""},
            )

            # Add champion threshold line (main reference)
            fig.add_vline(
                x=champion_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Champion Threshold ({champion_threshold:.1%})",
                annotation_position="top",
            )

            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_range=[0, 1],
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(fig, use_container_width=True)

        # Detailed model information
        st.markdown("---")
        st.markdown("### 📋 Detailed Model Information")

        for idx, row in results_df.iterrows():
            model_threshold = row["Threshold"]
            with st.expander(f"{row['Model']} - {row['Probability']:.2%}"):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Description:** {row['Description']}")
                    st.write(f"**Probability:** {row['Probability']:.4f}")
                    st.write(f"**Percentage:** {row['Probability']:.2%}")
                    st.write(f"**Model Threshold:** {model_threshold:.4f} ({model_threshold:.1%})")
                with col_info2:
                    # Decision based on model-specific threshold
                    if row["Probability"] >= model_threshold:
                        st.success("✅ **Recommendation:** Contact customer")
                        st.info(
                            f"Probability exceeds threshold by "
                            f"{(row['Probability'] - model_threshold):.2%}"
                        )
                    else:
                        st.warning("⚠️ **Recommendation:** Do not contact")
                        st.info(
                            f"Probability below threshold by "
                            f"{(model_threshold - row['Probability']):.2%}"
                        )

        # Final recommendation
        st.markdown("---")
        st.markdown("### 💡 Final Recommendation")

        if champion_prob >= champion_threshold:
            st.success(
                f"✅ **CONTACT RECOMMENDED**\n\n"
                f"The Champion Model predicts a {champion_prob:.1%} probability of subscription, "
                f"which exceeds the production threshold of {champion_threshold:.1%}. "
                f"It is recommended to contact this customer."
            )
        else:
            st.warning(
                f"⚠️ **DO NOT CONTACT**\n\n"
                f"The Champion Model predicts a {champion_prob:.1%} probability of subscription, "
                f"which is below the production threshold of {champion_threshold:.1%}. "
                f"It is not recommended to contact this customer at this time."
            )

        # Additional insights
        st.markdown("---")
        st.markdown("### 🔍 Additional Insights")

        col_insight1, col_insight2 = st.columns(2)

        with col_insight1:
            st.markdown("**Model Agreement:**")
            # Count models that recommend contact based on their own thresholds
            above_threshold = (results_df["Probability"] >= results_df["Threshold"]).sum()
            total_models = len(results_df)
            agreement = (above_threshold / total_models) * 100
            st.progress(agreement / 100)
            st.caption(
                f"{above_threshold}/{total_models} models recommend contact "
                f"({agreement:.0f}% agreement)"
            )

        with col_insight2:
            st.markdown("**Probability Range:**")
            st.info(
                f"Lowest: {min_prob:.2%} | Highest: {max_prob:.2%} | "
                f"Spread: {(max_prob - min_prob):.2%}"
            )

        # SHAP Explainability Section
        st.markdown("---")
        st.markdown("### 🔍 Model Explanation (SHAP)")
        st.info(
            "SHAP (SHapley Additive exPlanations) shows how each feature "
            "contributes to this prediction. "
            "Positive values push toward subscription, negative values push away."
        )

        # Check if champion model exists
        champion_model_key = None
        champion_model_obj = None
        for model_key, model_info in models_dict.items():
            if "Champion" in model_info["name"] or model_key == "xgboost_weighted_optimized":
                champion_model_key = model_key
                champion_model_obj = model_info["model"]
                break

        if champion_model_obj is not None:
            # Load background data
            background_data = load_background_data(n_samples=100)

            # Create SHAP explainer
            explainer = None
            shap_error = None
            try:
                explainer = load_shap_explainer(champion_model_obj, background_data)
                # Get error from function if it failed
                if explainer is None and hasattr(load_shap_explainer, "_last_error"):
                    shap_error = load_shap_explainer._last_error
            except Exception as e:
                shap_error = f"{type(e).__name__}: {str(e)}"

            if explainer is not None:
                # Calculate SHAP values
                shap_values, expected_value = calculate_shap_values(explainer, processed_input)

                if shap_values is not None and expected_value is not None:
                    # Get feature names from preprocessor
                    feature_names = preprocessor.get_feature_names_out()

                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(
                        ["📊 Local Explanation", "📈 Feature Contributions", "🌐 Global Importance"]
                    )

                    with tab1:
                        st.markdown("#### Waterfall Plot - Why This Prediction?")
                        st.caption(
                            "This shows how each feature moves the prediction from the base value "
                            f"({expected_value:.4f}) to the final prediction ({champion_prob:.4f})"
                        )

                        # Create waterfall plot
                        waterfall_fig = plot_shap_waterfall(
                            explainer, shap_values, expected_value, feature_names, processed_input
                        )

                        if waterfall_fig is not None:
                            st.pyplot(waterfall_fig)
                            plt.close()
                        else:
                            st.warning(
                                "Could not generate waterfall plot. "
                                "Showing feature contributions instead."
                            )
                            # Fallback: show feature contributions table
                            # Ensure processed_input is numpy array
                            if isinstance(processed_input, np.ndarray):
                                feature_values = (
                                    processed_input[0]
                                    if processed_input.ndim > 1
                                    else processed_input
                                )
                            else:
                                feature_values = np.array(processed_input)[0]

                            contrib_df = pd.DataFrame(
                                {
                                    "Feature": feature_names,
                                    "SHAP Value": shap_values,
                                    "Feature Value": feature_values,
                                }
                            )
                            contrib_df["Contribution"] = contrib_df["SHAP Value"].apply(
                                lambda x: "📈 Increases" if x > 0 else "📉 Decreases"
                            )
                            contrib_df = contrib_df.sort_values(
                                "SHAP Value", key=abs, ascending=False
                            ).head(15)
                            st.dataframe(
                                contrib_df[
                                    ["Feature", "SHAP Value", "Feature Value", "Contribution"]
                                ],
                                use_container_width=True,
                            )

                    with tab2:
                        st.markdown("#### Feature Contribution Table")
                        st.caption(
                            "Top features contributing to this prediction, "
                            "sorted by absolute impact"
                        )

                        # Create feature contribution dataframe
                        # Ensure processed_input is numpy array
                        if isinstance(processed_input, np.ndarray):
                            feature_values = (
                                processed_input[0] if processed_input.ndim > 1 else processed_input
                            )
                        else:
                            feature_values = np.array(processed_input)[0]

                        contrib_df = pd.DataFrame(
                            {
                                "Feature": feature_names,
                                "SHAP Value": shap_values,
                                "Feature Value": feature_values,
                                "|SHAP Value|": np.abs(shap_values),
                            }
                        )

                        # Sort by absolute SHAP value
                        contrib_df = contrib_df.sort_values("|SHAP Value|", ascending=False).head(
                            20
                        )

                        # Add color coding
                        def color_shap_value(val):
                            if val > 0:
                                # Green for positive
                                return "background-color: #d4edda; color: #155724"
                            else:
                                return (
                                    "background-color: #f8d7da; color: #721c24"  # Red for negative
                                )

                        styled_df = contrib_df[
                            ["Feature", "SHAP Value", "Feature Value"]
                        ].style.applymap(color_shap_value, subset=["SHAP Value"])

                        st.dataframe(styled_df, use_container_width=True, hide_index=True)

                        # Summary statistics
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        with col_sum1:
                            positive_contrib = (shap_values > 0).sum()
                            st.metric("Features Increasing Probability", positive_contrib)
                        with col_sum2:
                            negative_contrib = (shap_values < 0).sum()
                            st.metric("Features Decreasing Probability", negative_contrib)
                        with col_sum3:
                            total_impact = shap_values.sum()
                            st.metric("Net SHAP Impact", f"{total_impact:.4f}")

                    with tab3:
                        st.markdown("#### Global Feature Importance")
                        st.caption(
                            "Average feature importance across all predictions (from test set)"
                        )

                        # Load test data for global importance
                        try:
                            test_path = DATA_PROCESSED_DIR / "test.csv"
                            if test_path.exists():
                                test_df = pd.read_csv(test_path)
                                X_test = test_df.drop(columns=["target"]).values

                                # Sample for faster computation
                                if len(X_test) > 500:
                                    X_test_sample = X_test[:500]
                                else:
                                    X_test_sample = X_test

                                # Calculate SHAP values for sample
                                shap_values_all = explainer.shap_values(X_test_sample)
                                if isinstance(shap_values_all, list):
                                    shap_values_all = shap_values_all[1]  # Positive class

                                # Create summary plot
                                summary_fig = plot_shap_summary(shap_values_all, feature_names)

                                if summary_fig is not None:
                                    st.pyplot(summary_fig)
                                    plt.close()
                                else:
                                    st.info(
                                        "Global importance plot not available. "
                                        "Use local explanation instead."
                                    )
                            else:
                                st.info("Test data not found. Global importance requires test.csv")
                        except Exception as e:
                            st.warning(f"Could not generate global importance: {e}")
                            st.info(
                                "Use the Local Explanation tab for feature importance analysis."
                            )
                else:
                    st.warning("⚠️ Could not calculate SHAP values.")
                    st.info(
                        "This may occur if the model structure is incompatible "
                        "or input format is incorrect."
                    )
            else:
                st.warning("⚠️ Could not create SHAP explainer.")
                # Show more detailed error information
                model_type = str(type(champion_model_obj))
                error_msg = f"**Model Type:** {model_type}\n\n"

                if shap_error:
                    error_msg += f"**Error Details:** {shap_error}\n\n"
                    error_msg += "This error occurred while creating the SHAP TreeExplainer. "
                    error_msg += "Possible causes:\n"
                    error_msg += "- SHAP/XGBoost version incompatibility\n"
                    error_msg += "- Model was saved with a different XGBoost version\n"
                    error_msg += "- Model structure incompatibility\n\n"
                    error_msg += "**Troubleshooting:**\n"
                    error_msg += "1. Ensure XGBoost and SHAP versions are compatible\n"
                    error_msg += "2. Try retraining the model with the current XGBoost version\n"
                    error_msg += "3. Check if the model file is corrupted"
                else:
                    error_msg += (
                        "SHAP explanations require a tree-based model "
                        "(XGBoost, LightGBM, etc.). "
                    )
                    error_msg += (
                        "Ensure the champion model (xgboost_weighted_optimized.pkl) "
                        "is an XGBoost model."
                    )

                st.error(error_msg)
        else:
            st.warning("⚠️ Champion model not found.")
            st.info(
                "SHAP explanations require the champion model "
                "(xgboost_weighted_optimized.pkl). Please train the model first."
            )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.exception(e)
        st.info("Please ensure all models and preprocessor are properly trained and saved.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Bank Marketing Campaign Predictor | AI Final Project | Powered by XGBoost & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
