"""
Visualization functions for EDA
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml


def setup_plotting_style(config_path: str = "config/config.yaml"):
    """Setup plotting style from config."""
    config = load_config(config_path)
    plot_config = config["plotting"]

    sns.set_theme(style=plot_config["style"], palette=plot_config["palette"])
    plt.rcParams["figure.figsize"] = plot_config["figure_size"]
    plt.rcParams["font.size"] = plot_config["font_size"]


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / config_path

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_output_path(filename: str, config_path: str = "config/config.yaml") -> Path:
    """Get output path for saving figures."""
    config = load_config(config_path)
    output_dir = Path(__file__).parent.parent.parent / config["output"]["figures_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def plot_class_imbalance(
    df: pd.DataFrame,
    target_var: str = "y",
    config_path: str = "config/config.yaml",
    save: bool = True,
):
    """
    Create bar plot and pie chart for class imbalance analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_var : str
        Target variable name
    config_path : str
        Path to configuration file
    save : bool
        Whether to save the figure
    """
    setup_plotting_style(config_path)
    config = load_config(config_path)

    target_counts = df[target_var].value_counts()
    target_percentages = df[target_var].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot
    ax1 = axes[0]
    ax1.bar(
        target_counts.index,
        target_counts.values,
        color=["#3498db", "#e74c3c"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_xlabel("Target Variable (y)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax1.set_title("Class Distribution - Bar Plot", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (idx, val) in enumerate(target_counts.items()):
        ax1.text(
            i,
            val + target_counts.max() * 0.01,
            f"{val:,}\n({target_percentages[idx]:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Pie chart
    ax2 = axes[1]
    colors = ["#3498db", "#e74c3c"]
    wedges, texts, autotexts = ax2.pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.2f%%",
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05),
        shadow=True,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax2.set_title("Class Distribution - Pie Chart", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    if save:
        output_path = get_output_path("class_imbalance_analysis.png", config_path)
        plt.savefig(output_path, dpi=config["plotting"]["dpi"], bbox_inches="tight")
        print(f"[OK] Saved: {output_path}")

    plt.close()

    return target_counts, target_percentages


def plot_categorical_conversion(
    df: pd.DataFrame,
    categorical_vars: list = None,
    target_var: str = "y",
    config_path: str = "config/config.yaml",
    save: bool = True,
):
    """
    Create stacked bar plots for categorical conversion rates.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    categorical_vars : list
        List of categorical variables to plot
    target_var : str
        Target variable name
    config_path : str
        Path to configuration file
    save : bool
        Whether to save the figure
    """
    setup_plotting_style(config_path)
    config = load_config(config_path)

    if categorical_vars is None:
        categorical_vars = config["eda"]["categorical_vars"]

    fig, axes = plt.subplots(1, len(categorical_vars), figsize=(18, 6))

    for idx, var in enumerate(categorical_vars):
        ax = axes[idx] if len(categorical_vars) > 1 else axes

        # Create crosstab for stacked bar plot
        crosstab = pd.crosstab(df[var], df[target_var], normalize="index") * 100

        # Plot stacked bar
        crosstab.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=["#e74c3c", "#2ecc71"],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        ax.set_xlabel(var.capitalize(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Conversion Rate by {var.capitalize()}", fontsize=12, fontweight="bold", pad=10
        )
        ax.legend(title="Subscription", labels=["No", "Yes"], loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=45)

        # Add percentage annotations
        for container in ax.containers:
            ax.bar_label(
                container, fmt="%.1f%%", label_type="center", fontsize=8, fontweight="bold"
            )

    plt.tight_layout()

    if save:
        output_path = get_output_path("categorical_conversion_rate.png", config_path)
        plt.savefig(output_path, dpi=config["plotting"]["dpi"], bbox_inches="tight")
        print(f"[OK] Saved: {output_path}")

    plt.close()


def plot_numerical_analysis(
    df: pd.DataFrame, config_path: str = "config/config.yaml", save: bool = True
):
    """
    Create histogram for age and boxplot for balance.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    config_path : str
        Path to configuration file
    save : bool
        Whether to save the figure
    """
    setup_plotting_style(config_path)
    config = load_config(config_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram for age
    ax1 = axes[0]
    ax1.hist(df["age"], bins=30, color="#3498db", alpha=0.7, edgecolor="black", linewidth=1.2)
    ax1.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title("Age Distribution", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.axvline(
        df["age"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["age"].mean():.1f}',
    )
    ax1.axvline(
        df["age"].median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f'Median: {df["age"].median():.1f}',
    )
    ax1.legend()

    # Boxplot for balance
    ax2 = axes[1]
    ax2.boxplot(
        df["balance"],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#e74c3c", alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )
    ax2.set_ylabel("Balance (Euros)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Balance Distribution - Outlier Detection", fontsize=14, fontweight="bold", pad=15
    )
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Calculate and display outlier statistics
    Q1 = df["balance"].quantile(0.25)
    Q3 = df["balance"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df["balance"] < lower_bound) | (df["balance"] > upper_bound)]

    print("\nBalance Outlier Statistics:")
    print(f"  Q1: {Q1:.2f}")
    print(f"  Q3: {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    plt.tight_layout()

    if save:
        output_path = get_output_path("numerical_outliers_analysis.png", config_path)
        plt.savefig(output_path, dpi=config["plotting"]["dpi"], bbox_inches="tight")
        print(f"[OK] Saved: {output_path}")

    plt.close()

    return {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers),
        "outlier_percentage": len(outliers) / len(df) * 100,
    }


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numerical_vars: list = None,
    config_path: str = "config/config.yaml",
    save: bool = True,
):
    """
    Create correlation heatmap for numerical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    numerical_vars : list
        List of numerical variables
    config_path : str
        Path to configuration file
    save : bool
        Whether to save the figure
    """
    setup_plotting_style(config_path)
    config = load_config(config_path)

    if numerical_vars is None:
        numerical_vars = config["eda"]["numerical_vars"]

    numerical_df = df[numerical_vars]
    correlation_matrix = numerical_df.corr(method="pearson")

    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    plt.title(
        "Pearson Correlation Heatmap - Numerical Variables", fontsize=14, fontweight="bold", pad=20
    )
    plt.tight_layout()

    if save:
        output_path = get_output_path("correlation_heatmap.png", config_path)
        plt.savefig(output_path, dpi=config["plotting"]["dpi"], bbox_inches="tight")
        print(f"[OK] Saved: {output_path}")

    plt.close()

    return correlation_matrix


def plot_seasonality(
    df: pd.DataFrame,
    target_var: str = "y",
    config_path: str = "config/config.yaml",
    save: bool = True,
):
    """
    Plot success rate per month for seasonality analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_var : str
        Target variable name
    config_path : str
        Path to configuration file
    save : bool
        Whether to save the figure
    """
    setup_plotting_style(config_path)
    config = load_config(config_path)

    # Calculate success rate per month
    monthly_stats = (
        df.groupby("month")[target_var].agg(["count", lambda x: (x == "yes").sum()]).reset_index()
    )
    monthly_stats.columns = ["month", "total", "yes_count"]
    monthly_stats["success_rate"] = (monthly_stats["yes_count"] / monthly_stats["total"]) * 100

    # Order months chronologically
    month_order = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    monthly_stats["month"] = pd.Categorical(
        monthly_stats["month"], categories=month_order, ordered=True
    )
    monthly_stats = monthly_stats.sort_values("month")

    print("\nMonthly Success Rates:")
    print(monthly_stats[["month", "total", "yes_count", "success_rate"]].to_string(index=False))

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        monthly_stats["month"],
        monthly_stats["success_rate"],
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    plt.xlabel("Month", fontsize=12, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    plt.title(
        "Success Rate (Yes Counts) per Month - Seasonal Trends",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, rate in zip(bars, monthly_stats["success_rate"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        output_path = get_output_path("seasonality_analysis.png", config_path)
        plt.savefig(output_path, dpi=config["plotting"]["dpi"], bbox_inches="tight")
        print(f"[OK] Saved: {output_path}")

    plt.close()

    return monthly_stats


def plot_duration_analysis(
    df: pd.DataFrame,
    target_var: str = "y",
    config_path: str = "config/config.yaml",
    save: bool = True,
):
    """
    Create boxplot comparing duration for yes vs no classes.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_var : str
        Target variable name
    config_path : str
        Path to configuration file
    save : bool
        Whether to save the figure
    """
    setup_plotting_style(config_path)
    config = load_config(config_path)

    # Prepare data for boxplot
    duration_yes = df[df[target_var] == "yes"]["duration"]
    duration_no = df[df[target_var] == "no"]["duration"]

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [duration_no, duration_yes]
    box_plot = ax.boxplot(
        box_data,
        tick_labels=["No", "Yes"],
        patch_artist=True,
        boxprops=dict(facecolor="#e74c3c", alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )

    # Change color for 'Yes' box
    box_plot["boxes"][1].set_facecolor("#2ecc71")

    ax.set_xlabel("Subscription (y)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Duration (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Duration Distribution: Yes vs No", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add statistics text
    stats_text = f"Mean Duration - No: {duration_no.mean():.1f}s, Yes: {duration_yes.mean():.1f}s\n"
    stats_text += (
        f"Median Duration - No: {duration_no.median():.1f}s, Yes: {duration_yes.median():.1f}s"
    )
    ax.text(
        0.5,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save:
        output_path = get_output_path("duration_analysis.png", config_path)
        plt.savefig(output_path, dpi=config["plotting"]["dpi"], bbox_inches="tight")
        print(f"[OK] Saved: {output_path}")

    plt.close()

    # Print duration statistics
    print("\nDuration Statistics:")
    print(f"  No - Mean: {duration_no.mean():.2f}s, Median: {duration_no.median():.2f}s")
    print(f"  Yes - Mean: {duration_yes.mean():.2f}s, Median: {duration_yes.median():.2f}s")

    return {
        "duration_no": {"mean": duration_no.mean(), "median": duration_no.median()},
        "duration_yes": {"mean": duration_yes.mean(), "median": duration_yes.median()},
    }
