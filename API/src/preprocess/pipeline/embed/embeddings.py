import torch
from loguru import logger

from . import bert_embedder as embedder
from config import preprocess_settings


def get_embeddings_from_texts(
        features_clean
    ):
    """
    Generate embeddings for specified text columns in a dictionary using a BERT model.
    Parameters:
        features_clean (dict): A dictionary containing the cleaned features.
    Returns:
        dict: A dictionary containing the embeddings for the text fields.
    """

    logger.info('Generating embeddings.')
    try:

        embeddings_dict = {} 

        # Iterate through the text columns and embed the text
        for col in preprocess_settings.text_to_embed:

            # Get the embeddings for the text
            with torch.no_grad():
                embeddings = embedder.embed(features_clean[col]).cpu().numpy()

            # Add them to the dictionary
            embeddings_dict[f'{col}_emb'] = embeddings
    
    except Exception as e:
        logger.error(f'Error generating embeddings: {e}')
        raise

    # Return the Dictionary with the embeddings
    return embeddings_dict
    