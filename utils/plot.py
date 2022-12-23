import logging
from pathlib import Path
from typing import Dict, List, Optional
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
