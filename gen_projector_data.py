import typer

from typing import Optional

import tensorflow as tf

from dataloader.data_generator import DataGenerator
from models.projector import save_embeddings_for_tf_projector
from utils.helper import load_config


DATA_DIRPATH = "tiny-imagenet-200/train/"
BATCH_SIZE = 256
IMAGE_SIZE = (64, 64)
VALIDATION_SPLIT = 0.2

 
def main(config_filepath: str=typer.Option(...), model_dirpath: str=typer.Option(...)):
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU(s) detected: {tf.config.list_physical_devices('GPU')}")
        
    print("----------------------------------------------------------------------------------\n\n")
    
    # ---- Load config ---
    config = load_config(config_filepath)
    
    # --- Load data ---
    data_generator = DataGenerator(
        directory=DATA_DIRPATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=config["seed"],
        validation_split=VALIDATION_SPLIT
    )
    
    data_generator.val = data_generator.val.shuffle().take(1000)
    
    # --- Visualization ---
    model = tf.keras.models.load_model(model_dirpath, compile=False)
    save_embeddings_for_tf_projector(model, ds_val=data_generator.val, # type: ignore
                                     experiment_name=config["experiment_name"])
    
    return


if __name__ == "__main__":
    typer.run(main)
