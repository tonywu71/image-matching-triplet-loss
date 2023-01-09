"""train.py is used to train a Feature Model
"""
import typer

from typing import Optional

import tensorflow as tf

from dataloader.data_generator import DataGenerator
from models.trainer import resume_training, train
from utils.helper import load_config


DATA_DIRPATH = "tiny-imagenet-200/train/"
BATCH_SIZE = 256
IMAGE_SIZE_DATASET = (64, 64)
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

 
def main(config_filepath: str=typer.Option(...),
         resume_filepath: Optional[str]=typer.Option(None)):
    """Train a model according to the input config.

    Args:
        config_filepath (str, optional): Filepath for the HPT config file..
        resume_filepath (Optional[str], optional): Filepath for the previously trained model. Configs must match.
    """
    
    print("\n\n----------------------------------------------------------------------------------\n\n")
    
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU(s) detected: {tf.config.list_physical_devices('GPU')}")
        
    print("\n\n----------------------------------------------------------------------------------\n\n")
    
    # ---- Load config ---
    config = load_config(config_filepath)
    
    # --- Load data ---
    data_generator = DataGenerator(
        directory=DATA_DIRPATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE_DATASET,
        shuffle=True,
        seed=config["seed"],
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )
    
    # --- Training ---
    if resume_filepath:
        resume_training(model_dirpath=resume_filepath, config=config, data_generator=data_generator)
        model_dirpath = resume_filepath  # rename arg for backward compatibility
    else:
        model_dirpath = train(config=config, data_generator=data_generator)
    
    print(f"Model has been successfully saved at `{model_dirpath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
