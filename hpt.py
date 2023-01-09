"""hpt.py is used to perform HyperParameter Tuning (HPT)
"""

from typing import Optional
import typer

import os
import logging
from functools import partial
from pathlib import Path


from models.hpt.hpt_trainer import get_objective
from models.hpt.hpt_utils import (
    load_hpt_config,
    ask_confirmation,
    create_new_study,
    load_previous_study
)


logger = logging.getLogger(__name__)
STUDY_DIRPATH = Path("hpt_studies/")


def main(hpt_config_filepath: str=typer.Option(...),
         resume: Optional[bool]=typer.Option(None)):
    """Run the Hyperparameter Tuning (HPT) using the bayesian optimization with Optuna 
    for the Feature Model.
    
    Args:
        hpt_config_filepath (str, optional): Filepath for the HPT config file..
        resume (Optional[str], optional): Set flag to resume the previous study.
    """
    
    print("\n\n----------------------------------------------------------------------------------\n\n")
    
    # --- Load HPT config ---
    hpt_config = load_hpt_config(hpt_config_filepath)
    
    # --- Create objective function to optimize ---
    objective = partial(get_objective, hpt_config=hpt_config)
    
    # --- Create a DB file to store HPT results ---
    STUDY_DIRPATH.mkdir(parents=True, exist_ok=True)
    filepath_db = f'{STUDY_DIRPATH}/{hpt_config["study_name"]}.db'
    
    # --- Handle how and where HPT results are saved ---
    if not resume:        
        if os.path.isfile(filepath_db):  # If study already exists...
            if ask_confirmation(filepath_db):
                os.remove(filepath_db)
            else:
                return  # abort the HPT script
        study = create_new_study(filepath_db=filepath_db, hpt_config=hpt_config)
    else:
        study = load_previous_study(filepath_db=filepath_db, hpt_config=hpt_config)
    
    logger.info("Successfully created/resumed Optuna study.")
    
    
    # --- Run the HPT bayesian optimization algorithm ---
    logger.info("Start optimization process.")
    study.optimize(objective, n_trials=hpt_config["n_trials"], gc_after_trial=True)
    logger.info("Finish optimization process.")
    
    
    # --- Print information about the best trial ---
    print("\n\n----------------------------------------------------------------------------------")
    print("Best trial:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("----------------------------------------------------------------------------------\n\n")
    
    return


if __name__ == "__main__":
    typer.run(main)
