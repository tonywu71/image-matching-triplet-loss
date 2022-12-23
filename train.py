import typer
import tensorflow as tf
from dataloader.datasets import DataGenerator
from models.projector import save_embeddings_for_tf_projector
from models.trainer import train


DATA_DIRPATH = "tiny-imagenet-200/train/"
BATCH_SIZE = 256
IMAGE_SIZE = (64, 64)
VALIDATION_SPLIT = 0.2

 
def main(config_filepath: str):
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU(s) detected: {tf.config.list_physical_devices('GPU')}")
    
    data_generator = DataGenerator(
        directory=DATA_DIRPATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=0,
        validation_split=VALIDATION_SPLIT
    )
    model_dirpath = train(config_filepath, data_generator=data_generator)
    
    model = tf.keras.models.load_model(model_dirpath)
    
    save_embeddings_for_tf_projector(model, ds_val=data_generator.val) # type: ignore
    
    return


if __name__ == "__main__":
    typer.run(main)
