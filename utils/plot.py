import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()
logger = logging.getLogger(__name__)


def plot_learning_curve(history: Dict[str, List[int]], title: Optional[str]=None, fig_savepath: Optional[str]=None):
    df = pd.DataFrame(history, columns=["loss", "val_loss"])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    df.plot(ax=ax, xlabel="Epochs", ylabel="Loss")
    
    if title is None:
        title = "Learning curve"
    ax.set_title(title, fontsize=14)
    
    fig.tight_layout()
    
    if fig_savepath:
        Path(fig_savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_savepath)
        logger.info(f"Successfully saved learning curve at `{fig_savepath}`.")
    
    return


def plot_roc_auc_curve(fpr: np.ndarray,
                       tpr: np.ndarray,
                       auc: float,
                       title: Optional[str]=None,
                       figsize: Tuple[float, float]=(6, 6),
                       fig_savepath: Optional[str]=None):
    if title is None:
        title = "ROC AUC Curve"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="orange", linestyle="--", label="Random guess")
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    ax.set_title("ROC AUC Curve", fontsize=14)
    
    if fig_savepath:
        Path(fig_savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_savepath)
        logger.info(f"Successfully saved the ROC AUC rurve at `{fig_savepath}`.")
    
    return
