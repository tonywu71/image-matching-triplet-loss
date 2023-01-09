import typer

import tensorflow as tf

from dataloader.data_generator import DataGenerator, _preprocessing_function
from models.projector import save_embeddings_and_metadata, generate_sprite
from utils.helper import load_config


from train import DATA_DIRPATH, BATCH_SIZE, IMAGE_SIZE_DATASET, VAL_SPLIT

 
def main(config_filepath: str=typer.Option(...),
         model_dirpath: str=typer.Option(...),
         n_examples: int=typer.Option(5000)):
    print("\n")
    
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU(s) detected: {tf.config.list_physical_devices('GPU')}")
        print("----------------------------------------------------------------------------------\n\n")
    
    
    # ---- Load config ---
    config = load_config(config_filepath)
    
    
    # --- Load data ---
    data_generator = DataGenerator(
        directory=DATA_DIRPATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE_DATASET,
        shuffle=True,
        seed=config["seed"],
        val_split=VAL_SPLIT
    )
    
    ds_val_raw = data_generator.val_raw.shuffle(buffer_size=n_examples).take(n_examples)
    ds_val = ds_val_raw.map(_preprocessing_function).batch(BATCH_SIZE)
    
    
    # --- Load model ---
    model = tf.keras.models.load_model(model_dirpath, compile=False)
    
    
    # --- Generate embeddings and metadata ---
    vecs_filepath, meta_filepath = save_embeddings_and_metadata(model,  # type: ignore
                                                                ds_val=ds_val,
                                                                experiment_name=config["experiment_name"])
    print(f"Embeddings successfully saved at `{vecs_filepath}`.")
    print(f"Metadata successfully saved at `{meta_filepath}`.")
    
    
    # --- Generate sprite ---
    sprite_filepath = generate_sprite(ds_val_raw, experiment_name=config["experiment_name"])
    print(f"Sprite successfully saved at `{sprite_filepath}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
