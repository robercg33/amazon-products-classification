
from pydantic import DirectoryPath, ValidationError
from pydantic_settings import BaseSettings

import json

class ModelSettings(BaseSettings):


    model_path: DirectoryPath
    model_name: str

    @classmethod
    def from_json(cls, file_path: str) -> 'ModelSettings':
        """
        Load ModelSettings from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            ModelSettings: An instance of the ModelSettings class.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading settings from {file_path}: {e}") from e
        except ValidationError as ve:
            raise ValueError(f"Validation error while loading settings: {ve}") from ve
        

# Load the settings from the config file
config_path = 'src/config/res/model_config.json'
model_settings = ModelSettings.from_json(config_path)