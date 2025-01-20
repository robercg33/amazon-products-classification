


import pickle
from pathlib import Path

from loguru import logger

from config import model_settings
from schema.amazon_product import AmazonProduct
from preprocess import data_preprocess


class ModelInferenceService:


    def __init__(self):
        
        self.model = None
        self.model_path = model_settings.model_path
        self.model_name = model_settings.model_name

    def load_model(self):
        
        logger.info(f'Checking model existence at {self.model_path}/{self.model_name}.')

        model_path = Path(f'{self.model_path}/{self.model_name}')

        if not model_path.exists():
            logger.error(f'Model not found at {model_path}.')
            raise FileNotFoundError(f'Model not found at {model_path}.')

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f'Model loaded from {model_path}.')

    def predict(self, amazon_product):

        # Preprocess the data
        features_preprocessed = data_preprocess.preprocess_data(amazon_product)

        # Make prediction
        prediction = self.model.predict(features_preprocessed)

        # Return the label for the prediction
        return data_preprocess.get_prediction_label(prediction)
