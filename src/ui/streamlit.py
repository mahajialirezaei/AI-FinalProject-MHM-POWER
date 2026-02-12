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
            # Use joblib.load() since models are saved with joblib.dump()
            model = joblib.load(model_path)
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
        prob = model_obj.predict_proba(processed_input)[0][1]
        return prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
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
        champion_threshold = (
            champion_row["Threshold"].values[0] if len(champion_row) > 0 else 0.5
        )
        champion_prob = (
            champion_row["Probability"].values[0] if len(champion_row) > 0 else results_df["Probability"].mean()
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
                delta=f"{champion_prob - champion_threshold:.1%}" if champion_prob >= champion_threshold else None,
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
