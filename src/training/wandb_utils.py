"""
WandB utility functions for consistent logging across all training scripts.
Supports online, offline, and disabled modes via configuration.
"""

import os
import wandb
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


def load_wandb_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load wandb configuration from config file."""
    project_root = Path(__file__).resolve().parent.parent.parent
    full_config_path = project_root / config_path
    
    if not full_config_path.exists():
        return {"enabled": False, "mode": "offline"}
    
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    wandb_config = config.get("wandb", {})
    return {
        "enabled": wandb_config.get("enabled", True),
        "mode": wandb_config.get("mode", "online"),  # online, offline, disabled
        "project": wandb_config.get("project", "ai-finalproject-mhm-power"),
        "entity": wandb_config.get("entity", None),
    }


def init_wandb(
    run_name: str,
    tags: Optional[list] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "config/config.yaml"
) -> Optional[wandb.run]:
    """
    Initialize WandB with configuration support.
    
    Parameters:
    -----------
    run_name : str
        Name for this run
    tags : list, optional
        Tags for the run
    config : dict, optional
        Additional config to log
    config_path : str
        Path to config file
        
    Returns:
    --------
    wandb.run or None
        Returns None if wandb is disabled
    """
    wandb_config = load_wandb_config(config_path)
    
    # Check if wandb is disabled
    if not wandb_config.get("enabled", True):
        print("[WandB] Disabled in config. Skipping initialization.")
        return None
    
    # Check environment variable (takes precedence)
    env_mode = os.getenv("WANDB_MODE", "").lower()
    if env_mode in ["offline", "disabled"]:
        wandb_config["mode"] = env_mode
    
    mode = wandb_config.get("mode", "online")
    
    if mode == "disabled":
        print("[WandB] Disabled. Skipping initialization.")
        return None
    
    # Prepare init parameters
    init_params = {
        "project": wandb_config.get("project", "ai-finalproject-mhm-power"),
        "name": run_name,
        "mode": mode,
        "tags": tags or [],
    }
    
    if wandb_config.get("entity"):
        init_params["entity"] = wandb_config["entity"]
    
    if config:
        init_params["config"] = config
    
    try:
        run = wandb.init(**init_params)
        print(f"[WandB] Initialized in {mode} mode: {run_name}")
        return run
    except Exception as e:
        print(f"[WandB] Warning: Failed to initialize: {e}")
        print("[WandB] Continuing without WandB logging...")
        return None


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log metrics to WandB if enabled."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_config(config: Dict[str, Any]):
    """Update WandB config if enabled."""
    if wandb.run is not None:
        wandb.config.update(config)


def log_artifact(file_path: str, artifact_name: str, artifact_type: str = "model"):
    """Log an artifact to WandB if enabled."""
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[WandB] Warning: Failed to log artifact: {e}")


def log_image(image_path: str, key: str = "image"):
    """Log an image to WandB if enabled."""
    if wandb.run is not None:
        try:
            wandb.log({key: wandb.Image(image_path)})
        except Exception as e:
            print(f"[WandB] Warning: Failed to log image: {e}")


def log_confusion_matrix(y_true, y_pred, class_names=None):
    """Log confusion matrix to WandB if enabled."""
    if wandb.run is not None:
        try:
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names or ["Class 0", "Class 1"]
                )
            })
        except Exception as e:
            print(f"[WandB] Warning: Failed to log confusion matrix: {e}")


def finish_wandb():
    """Finish WandB run if enabled."""
    if wandb.run is not None:
        try:
            wandb.finish()
            print("[WandB] Run finished successfully.")
        except Exception as e:
            print(f"[WandB] Warning: Failed to finish run: {e}")
