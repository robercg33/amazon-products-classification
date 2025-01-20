import json
from typing import List

from pydantic import DirectoryPath, ValidationError
from pydantic_settings import BaseSettings

class PreprocessingSettings(BaseSettings):

    # Encoder filepath
    encoders_path: DirectoryPath
    encoders_filename: str

    # Text features to get embeddings from
    text_to_embed: List[str]

    # Actual features to use
    numeric: List[str]
    numeric_log: List[str]
    categ: List[str]
    embed: List[str]

    # Features mapping
    features_mapping: dict

    # Embed config
    embed_config: dict


    @classmethod
    def from_json(cls, file_path: str) -> 'PreprocessingSettings':
        """
        Load PreprocessingSettings from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            PreprocessingSettings: An instance of the PreprocessingSettings class.
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
config_path = 'src/config/res/preprocess_config.json'
preprocess_settings = PreprocessingSettings.from_json(config_path)
