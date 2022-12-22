import logging
from pathlib import Path
from datetime import datetime

import tensorflow as tf

from dataloader.datasets import DataGenerator
from models.feature_model import load_and_compile_model
from utils.helper import load_config
from utils.plot import plot_learning_curve

logger = logging.getLogger(__name__)


DATA_DIRPATH = "tiny-imagenet-200/train/"
BATCH_SIZE = 256
IMAGE_SIZE = (64, 64)
VALIDATION_SPLIT = 0.2


def train(config_filepath: str, seed: int=0):    
    # ---- Load config ---
    config = load_config(config_filepath)
    
    
    # --- Load data ---
    assert Path(DATA_DIRPATH).exists()
    
    data_generator = DataGenerator(
        directory=DATA_DIRPATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=seed,
        validation_split=VALIDATION_SPLIT
    )

    ds_train, ds_val = data_generator.train, data_generator.val
    
    
    # --- Load the model ---
    model = load_and_compile_model(model="efficientnet")
    
    log_dir = Path(f"logs/{config['experiment_name']}/fit/")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config["early_stopping_patience"],
            monitor="val_loss",
            mode="min",
            restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5) # type: ignore
    ]
    
    
    # --- Train the model ---
    history = model.fit(ds_train,
                        epochs=config["epochs"],
                        validation_data=ds_val,
                        callbacks=callbacks)

    
    plot_learning_curve(history, fig_savepath=f"logs/{config['experiment_name']}/")
    

    # --- Save the model ---
    model_dirpath = Path(f"saved_models/{config['experiment_name']}")
    model_dirpath.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_dirpath)
    
    logger.info(f"Successfully saved model at `{model_dirpath}`.")
    
    return
