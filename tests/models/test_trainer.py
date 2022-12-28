from pathlib import Path
import shutil
from dataloader.datasets import DataGenerator
from models.trainer import train
from utils.helper import load_config


CONFIG_FILEPATH = "tests/models/config_test.yaml"

DATA_DIRPATH = "tiny-imagenet-200/train/"
BATCH_SIZE = 256
IMAGE_SIZE = (64, 64)
VALIDATION_SPLIT = 0.2


def test_trainer():
    config = load_config(CONFIG_FILEPATH)
    
    data_generator = DataGenerator(
        directory=DATA_DIRPATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=0,
        validation_split=VALIDATION_SPLIT
    )
    
    # Tackle long training by keeping only 1 batch of data
    data_generator.train = data_generator.train.take(1)
    data_generator.val = data_generator.val.take(1)
    
    try:
        model_dirpath = train(CONFIG_FILEPATH, data_generator=data_generator)
        assert model_dirpath.exists()
        
    except:
        raise
    
    finally:
        log_dir = Path(f"logs/{config['experiment_name']}/")
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)
        
        model_dirpath = Path(f"saved_models/{config['experiment_name']}/")
        if model_dirpath.exists():
            shutil.rmtree(model_dirpath, ignore_errors=True)
    