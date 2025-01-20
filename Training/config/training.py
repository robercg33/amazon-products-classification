from pydantic import field_validator, ValidationError, model_validator
from pydantic_settings import BaseSettings
from typing import List
import json
import random


class TrainingSettings(BaseSettings):

    data_seed: int = random.randint(0, 10000) # Random seed por splitting the data
    numeric: List[str] # List of numeric columns
    numeric_log: List[str] # List of numeric columns to apply log transformation
    categ: List[str] # List of categorical columns
    embed: List[str] # List of embeddings columns
    target: str # Target column
    test_size: float = 0.3 # Test size for splitting the data

    # Test size validation
    @field_validator('test_size')
    def check_test_size(cls, v):
        assert 0 < v < 1, f'test_size must be between 0 and 1, got {v}'
        return v
    
    # Validate every column in numeric_log is in numeric
    @model_validator(mode='before')
    def check_log_in_numeric(cls, data):
        assert all([col in data['numeric'] for col in data['numeric_log']]), 'numeric_log columns must be in numeric columns'
        return data
    
    # Load the settings from a JSON file
    @classmethod
    def from_json(cls, file_path: str) -> 'TrainingSettings':
        """
        Load TrainingSettings from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            TrainingSettings: An instance of the TrainingSettings class.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading settings from {file_path}: {e}") from e
        except ValidationError as ve:
            raise ValueError(f"Validation error while loading settings: {ve}") from ve

#Build the settings
config_path = 'config/res/training_config.json'
training_settings = TrainingSettings.from_json(config_path)