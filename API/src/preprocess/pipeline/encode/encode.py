import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from loguru import logger

from config import preprocess_settings
from . import feature_encoder


def encode_features(features_clean, features_emb):
    """
    Encodes and preprocesses features for performing inference.
    Parameters:
        features_clean (dict): A dictionary containing the cleaned features.
        features_emb (dict): A dictionary containing the embeddings for the text fields.
    Returns:
        numpy.ndarray: A numpy array containing the concatenated and encoded features.

    The function performs the following steps:
    1. Applies log transformation to specified numeric features.
    2. Scales numeric features using a pre-fitted numeric scaler.
    3. Encodes categorical features using a pre-fitted categorical encoder.
    4. Sums the embeddings of specified columns.
    5. Concatenates the scaled numeric features, encoded categorical features, and summed embeddings.
    """

    logger.info('Encoding features.')
    try:
        ### Numeric features
        # Log transform numeric features
        for log_feature in preprocess_settings.numeric_log:
            features_clean[log_feature] = np.log1p(features_clean[log_feature])
        
        # Add numeric features to the list
        numeric_raw = np.array([features_clean[feature] for feature in preprocess_settings.numeric])

        # Scale numeric features using numeric scaler
        numeric_scaled = feature_encoder.encoders['numeric_scaler'].transform(numeric_raw.reshape(1, -1))

        ### Categorical features

        # Get the categorical features
        categ_raw = np.array([features_clean[feature] for feature in preprocess_settings.categ])

        # Encode categorical features using categorical encoder
        categ_encoded = feature_encoder.encoders['categorical_encoder'].transform(categ_raw.reshape(1, -1))

        ### Embeddings
        # Sum the embeddings of the three columns
        embeddings = np.sum([features_emb[col] for col in preprocess_settings.embed], axis=0)

    except Exception as e:
        logger.error(f'Error encoding features: {e}')
        raise

    ### Concatenate the features and return them
    return np.hstack([numeric_scaled, categ_encoded, embeddings])


def get_predicted_label(prediction):
    """
    Get the predicted label from the encoded prediction.
    Parameters:
        prediction (array-like): The encoded prediction values.
    Returns:
        array-like: The decoded predicted labels.
    """

    # Get the predicted label
    predicted_label = np.squeeze(
        feature_encoder.encoders['target_encoder'].inverse_transform(prediction)
    ).item()


    return predicted_label
