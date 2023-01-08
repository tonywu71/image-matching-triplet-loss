import logging
from pathlib import Path
from pprint import pprint

import tensorflow as tf
import optuna
from optuna.integration.tfkeras import TFKerasPruningCallback

from dataloader.data_generator import DataGenerator
from models.feature_model import load_and_compile_model
from models.hpt.hpt_utils import create_args_from_hpt_config
from train import DATA_DIRPATH, BATCH_SIZE, IMAGE_SIZE_DATASET, VALIDATION_SPLIT


logger = logging.getLogger(__name__)


STUDY_DIRPATH = Path("hpt_studies/")
STUDY_DIRPATH.mkdir(parents=True, exist_ok=True)


def hpt_train(model: tf.keras.Model,
              data_generator: DataGenerator,
              config: dict,
              trial: optuna.trial.Trial) -> tf.keras.callbacks.History:    
    # --- Load datasets ---
    ds_train, ds_val = data_generator.train, data_generator.val
    
    # --- Callbacks ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config["early_stopping_patience"],
            monitor="val_loss",
            mode="min",
            restore_best_weights=True)
    ]
    
    callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
    
    # --- Train the model ---
    history = model.fit(ds_train.take(2),
                        epochs=config["epochs"],
                        validation_data=ds_val.take(2),
                        callbacks=callbacks)
    
    return history


def get_objective(trial: optuna.trial.Trial, hpt_config: dict) -> float:
    """For a given set of hyperparameters, create a WindowGenerator and a ForecastModel. Then trains the model
    on the generated dataset and returns the objective value (here the validation loss).

    Returns:
        float
    """
    
    # --- Model-related hyperparameters ---
    embedding_dim = trial.suggest_categorical("embedding_dim", hpt_config["embedding_dim_grid"])
    intermediate_linear_units = trial.suggest_categorical("intermediate_linear_units", hpt_config["intermediate_linear_units_grid"])
    dropout = trial.suggest_categorical("dropout", hpt_config["dropout_grid"])
    
    logger.info("Successfully picked hyperparameters.")
    
    
    print("\n----------------------------------------------------------------------------------")
    logger.info(f"Start trial {trial.number}.")
    print(f"Start trial {trial.number} with parameters:")
    pprint(trial.params, indent=4)
    print("\n")
    
    
    # --- Get config for this trial's model ---
    model_config = create_args_from_hpt_config(
        hpt_config=hpt_config,
        embedding_dim=embedding_dim,
        intermediate_linear_units=intermediate_linear_units,
        dropout=dropout
    )
    
    
    # --- Print current config ---
    print("----------- Current configuration -----------")
    pprint(model_config)
    
    
    # --- Load data ---
    data_generator = DataGenerator(
        directory=DATA_DIRPATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE_DATASET,
        shuffle=True,
        seed=model_config["seed"],
        validation_split=VALIDATION_SPLIT
    )
    
    
    # --- Load the model ---
    model = load_and_compile_model(model_name=model_config["feature_extractor"],
                                   embedding_dim=model_config["embedding_dim"],
                                   intermediate_linear_units=model_config["intermediate_linear_units"],
                                   dropout=model_config["dropout"],
                                   image_augmentation=model_config["image_augmentation"])


    # --- Train and evaluate the model ---
    history = hpt_train(
        model=model,
        data_generator=data_generator,
        config=model_config,
        trial=trial)
    score = history.history["val_loss"][-1]
    
    # --- Free memory for the current model ---
    del model
    
    return score
