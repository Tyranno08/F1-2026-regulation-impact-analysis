# src/config_loader.py

import yaml
import os


def load_config(config_path=None):
    """
    Loads the project configuration from config/config.yaml.
    Returns the full config as a Python dictionary.
    """
    if config_path is None:
        # Resolve path relative to project root regardless of
        # where the script is called from
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully.")
    print(f"Project name: {config['project']['name']}")
    print(f"Seasons to analyze: {config['seasons']}")
    print(f"Number of circuits configured: {len(config['circuits'])}")