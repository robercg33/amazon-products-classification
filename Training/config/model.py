from pydantic import field_validator, ValidationError
from pydantic_settings import BaseSettings

import json
import random


class ModelSettings(BaseSettings):
    """
    ML model configuration settings.

    Attributes:
        penalty (str): Regularization type ('l1', 'l2', or 'elasticnet').
        seed (int): Random seed for reproducibility.
        solver (str): Optimization algorithm.
        max_iter (int): Maximum number of iterations.
        class_weight (str): Class weight balancing.
        model_name (str): Name of the model.
        multi_class (str): Multi-class strategy.
    """
    penalty: str = 'l2'
    seed: int = random.randint(0, 10000)
    solver: str = 'liblinear'
    max_iter: int = 1000
    #class_weight: str = None
    class_weight: str | None = None
    model_name: str = 'logistic_regression'
    multi_class: str = 'ovr'

    @field_validator('penalty')
    def check_penalty(cls, v):
        assert v in ['l1', 'l2', 'elasticnet'], 'penalty must be one of l1, l2, or elasticnet'
        return v

    @field_validator('solver')
    def check_solver(cls, v):
        assert v in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], \
            'solver must be one of newton-cg, lbfgs, liblinear, sag, saga'
        return v
    
    @field_validator('max_iter')
    def check_max_iter(cls, v):
        assert v > 0, f'max_iter must be greater than 0, got {v}'
        return v
    
    @field_validator('class_weight', mode='after')
    def check_class_weight(cls, v):
        assert v in [None, 'balanced'], \
            'class_weight must be one of None, balanced'
        return v
    
    @field_validator('multi_class')
    def check_multi_class(cls, v):
        assert v in ['ovr', 'multinomial', 'auto'], \
            'multi_class must be one of ovr, multinomial, auto'
        return v

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


config_path = 'config/res/model_config.json'

model_settings = ModelSettings.from_json(config_path)