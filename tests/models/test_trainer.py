from pathlib import Path
import shutil
from models.trainer import train
from utils.helper import load_config

CONFIG_FILEPATH = "tests/models/config_test.yaml"


def test_trainer():
    config = load_config(CONFIG_FILEPATH)
    
    try:
        train(CONFIG_FILEPATH, seed=0)
        
    except:
        raise
    
    finally:
        log_dir = Path(f"logs/{config['experiment_name']}/")
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)
        
        model_dirpath = Path(f"saved_models/{config['experiment_name']}/")
        if model_dirpath.exists():
            shutil.rmtree(model_dirpath, ignore_errors=True)
    