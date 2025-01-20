from pydantic import ValidationError
from pydantic_settings import BaseSettings

import json


class PathsSettings(BaseSettings):

    model_output_dir : str
    data_route : str
    embeddings_route : str

    @classmethod
    def from_json(cls, file_path: str) -> 'PathsSettings':
        """
        Load PathsSettings from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            PathsSettings: An instance of the PathsSettings class.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading settings from {file_path}: {e}") from e
        except ValidationError as ve:
            raise ValueError(f"Validation error while loading settings: {ve}") from ve

config_path = 'config/res/paths.json'
paths_settings = PathsSettings.from_json(config_path)