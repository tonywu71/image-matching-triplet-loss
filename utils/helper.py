import yaml


def load_config(config_filepath: str) -> dict:
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)
    return config
