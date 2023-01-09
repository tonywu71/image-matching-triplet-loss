import logging
from pathlib import Path
from datetime import datetime

import tensorflow as tf

from dataloader.data_generator import DataGenerator
from models.feature_model import load_and_compile_model
from utils.plot import plot_learning_curve

logger = logging.getLogger(__name__)
HISTOGRAM_FREQ = 5


def train(config: dict, data_generator: DataGenerator) -> Path:    
    # --- Load datasets ---
    ds_train, ds_val = data_generator.train, data_generator.val
    
    # --- Load the model ---
    model = load_and_compile_model(model_name=config["feature_extractor"],
                                   embedding_dim=config["embedding_dim"],
                                   intermediate_ff_block_units=config["intermediate_ff_block_units"],
                                   dropout=config["dropout"],
                                   image_augmentation=config["image_augmentation"])
    
    # --- Callbacks ---
    log_dir = Path(f"logs/{config['experiment_name']}/fit/")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config["early_stopping_patience"],
            monitor="val_loss",
            mode="min",
            restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=HISTOGRAM_FREQ) # type: ignore
    ]
    
    # --- Train the model ---
    history = model.fit(ds_train,
                        epochs=config["epochs"],
                        validation_data=ds_val,
                        callbacks=callbacks)

    learning_curve_filepath = Path(f"logs/{config['experiment_name']}/learning_curve-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
    learning_curve_filepath.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(history.history, fig_savepath=str(learning_curve_filepath))
    
    # --- Save the model ---
    model_dirpath = Path(f"saved_models/{config['experiment_name']}")
    model_dirpath.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_dirpath)
    
    logger.info(f"Successfully saved model at `{model_dirpath}`.")
    
    return model_dirpath


def resume_training(model_dirpath: str, config: dict, data_generator: DataGenerator):
    # ---- Load previous model ---
    assert Path(model_dirpath).exists()
    model = tf.keras.models.load_model(model_dirpath)  # type: tf.keras.Model
    logger.info("Previous trained model was succesfully loaded.")
    
    # --- Load datasets ---
    ds_train, ds_val = data_generator.train, data_generator.val
    
    # --- Callbacks ---
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
    
    plot_learning_curve(history.history, fig_savepath=f"logs/{config['experiment_name']}/learning_curve.png")
    

    # --- Save the model ---
    model.save(model_dirpath)
    logger.info(f"Successfully saved model at `{model_dirpath}`.")
    
    return
