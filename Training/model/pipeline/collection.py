'''
This module loads an already cleaned DF from a parquet file
and returns it loaded int a pandas DF
'''

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from config import training_settings
from config import paths_settings


data_path = paths_settings.data_route
embeddings_path = paths_settings.embeddings_route

def load_data(fraction = 1.0):
        
    """
    Loads data from a parquet file and adds embeddings to it.
    This function reads data from a specified parquet file and loads it into a pandas DataFrame.
    It also reads embeddings from a JSON file and adds them to the DataFrame. If a fraction less
    than 1.0 is specified, the data is sampled accordingly, stratifying by the target column.
    Args:
        fraction (float, optional): The fraction of data to load. Defaults to 1.0.
    Returns:
        pd.DataFrame: The DataFrame containing the loaded data with embeddings.
    """

    logger.info(f'Loading data from {data_path}...')

    # Data
    data = pd.read_parquet(data_path)

    # Embeddings
    embeddings_dict = np.load(embeddings_path)

    # Adds embeddings to data
    for embed_feature in embeddings_dict.keys():
        data[embed_feature] = embeddings_dict[embed_feature].tolist()

    # Sample the data if needed (using the data seed)
    # and stratifying by the target column
    if fraction < 1.0:
        data, _ = train_test_split(
            data, 
            test_size=(1-fraction), 
            random_state=training_settings.data_seed,
            stratify=data[training_settings.target]
        )
    
    logger.info(f'Data loaded. Number of records: {data.shape[0]}')

    # Return the data
    return data