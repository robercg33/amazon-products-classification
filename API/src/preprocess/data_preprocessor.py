from loguru import logger

from .pipeline.clean import clean_input
from .pipeline.embed.embeddings import get_embeddings_from_texts
from .pipeline.encode.encode import encode_features, get_predicted_label
from .pipeline.transform import get_transformed_input


class DataPreprocess:

    """
    Conver data from input format to processed and encoded features.
    """


    def preprocess_data(self, amazon_product):
        """
        Preprocesses the given Amazon product data by transforming, cleaning, 
        generating embeddings, and encoding features.
        Args:
            amazon_product (AmazonProduct): An object representing an Amazon product.
        Returns:
            dict: The preprocessed and encoded features of the Amazon product.
        """

        # Transform the data
        transformed_data = get_transformed_input(amazon_product)

        # Clean the data
        cleaned_data = clean_input(transformed_data)

        # Get the embeddings
        embeddings = get_embeddings_from_texts(cleaned_data)

        # Encode the features
        encoded_features = encode_features(cleaned_data, embeddings)

        return encoded_features
    
    def get_prediction_label(self, prediction):
        """
        Get the label for a given prediction.
        Args:
            prediction (any): The prediction result for which the label needs to be determined.
        Returns:
            str: The label corresponding to the given prediction.
        """

        return get_predicted_label(prediction)


# Instantiate the DataPreprocess class
logger.info('Instantiating DataPreprocess class.')
data_preprocess = DataPreprocess()
