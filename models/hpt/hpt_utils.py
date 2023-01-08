import json
import os
from pathlib import Path

import optuna
import yaml


STUDY_DIRPATH = Path("hpt_studies/")
STUDY_DIRPATH.mkdir(parents=True, exist_ok=True)


def ask_confirmation(filepath: str) -> bool:
    answer = input(f"⚠️  Running this script will overwrite the previous study `{filepath}`. " + \
        "Are you sure you want to continue? (y/n):  ") 
    if answer == "y": 
        return True
    elif answer == "n": 
        return False
    else: 
        print("Please enter y (yes) or n (no).")
        return False


def load_hpt_config(config_filepath: str) -> dict:
    with open(config_filepath, "r") as f:
        hpt_config = yaml.safe_load(f)
    
    return hpt_config
    

def create_args_from_hpt_config(hpt_config: dict, **trial_hparams) -> dict:
    args = {
        "experiment_name": hpt_config["study_name"],
        
        "seed": hpt_config["seed"],
        "image_augmentation": hpt_config["image_augmentation"],
        
        "feature_extractor": hpt_config["feature_extractor"],
        "embedding_dim": trial_hparams["embedding_dim"],
        "intermediate_linear_units": json.loads(trial_hparams["intermediate_linear_units"]),  # we had to use strings for Optuna compatibility
        "dropout": trial_hparams["dropout"],
        
        "epochs": hpt_config["epochs"],
        "early_stopping_patience": hpt_config["early_stopping_patience"]
    }
    return args


def create_new_study(filepath_db: str, hpt_config: dict):
    storage = f"sqlite:///{filepath_db}"
    study = optuna.create_study(direction="minimize", study_name=hpt_config["study_name"], storage=storage)
    print(f"Successfully created new study at '{storage}'.")
    return study


def load_previous_study(filepath_db: str, hpt_config: dict):
    # Load already existing study:
    if not os.path.isfile(filepath_db):
        raise FileNotFoundError
    
    study = optuna.create_study(study_name=hpt_config["study_name"], storage=f"sqlite:///{filepath_db}", load_if_exists=True)
    print(f"Previous study successfully loaded from '{filepath_db}'.")
    
    # Show best results so far:
    try:
        print("Best trial until now:")
        print(" Value: ", study.best_trial.value)
        print(" Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    except:  # Specific case where we want to use Distributed Optimization from scratch...
        print("No previous results found.")
    
    return study
