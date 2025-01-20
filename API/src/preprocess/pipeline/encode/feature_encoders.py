from loguru import logger
from pydantic import BaseModel

import pickle
from pathlib import Path

from config import preprocess_settings


class FeatureEncoder(BaseModel):
    '''
    A class to load and store the encoders for the features.
    Attributes:
        encoders (dict): A dictionary containing the encoders for the features:
            - 'numeric_scaler': A dictionary containing the scalers for the numeric features.
            - 'categorical_encoder': A dictionary containing the encoder for the categorical features.
            - 'target_encoder': A dictionary containing the encoder for the target variable.
    '''

    encoders : dict


    @classmethod
    def load_from_pickle(cls, encoders_path: str):
        """
        Load the encoders from a pickle file.

        Args:
            encoders_path (str): Path to the pickle file.

        Returns:
            FeatureEncoder: An instance of the FeatureEncoder class.
        """
        encoders_path = Path(encoders_path)

        if not encoders_path.exists():
            raise FileNotFoundError(f'Encoders not found at {encoders_path}.')

        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        return cls(encoders=encoders)
    
# Load the feature encoders
logger.info('Loading feature encoders.')
encoders_path = f'{preprocess_settings.encoders_path}/{preprocess_settings.encoders_filename}'
feature_encoder = FeatureEncoder.load_from_pickle(encoders_path)