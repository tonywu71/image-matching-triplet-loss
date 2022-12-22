import typer
import tensorflow as tf
from models.trainer import train
 
def main(config_filepath: str):
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU(s) detected: {tf.config.list_physical_devices('GPU')}")
        
    train(config_filepath=config_filepath)
    
    return


if __name__ == "__main__":
    typer.run(main)
