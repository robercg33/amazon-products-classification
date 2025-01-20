from torch.utils.data import DataLoader
from embed.bert import BertEmbedder, BertDataset

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from loguru import logger


def get_embeddings_from_texts(
        df, 
        text_cols, 
        batch_size=32, 
        max_length=128, 
        padding=True, 
        truncation=True
    ):
    """
    Generate embeddings for specified text columns in a DataFrame using a BERT model.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing text data.
        text_cols (list of str): List of column names in the DataFrame that contain text data to be embedded.
        batch_size (int, optional): The batch size for the DataLoader. Default is 32.
        max_length (int, optional): The maximum length of the text sequences for the BERT model. Default is 128.
        padding (bool, optional): Whether to pad the text sequences. Default is True.
        truncation (bool, optional): Whether to truncate the text sequences. Default is True.
    Returns:
        dict: A dictionary containing the embeddings for each text column.
        dict: A dictionary containing the configuration used to generate the embeddings
    """
    # Fill the NaN values with empty strings
    for col in text_cols:
        df[col] = df[col].fillna('')

    try:
        # Instantiate the BertEmbedder
        embedder = BertEmbedder(
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
            max_length=max_length, 
            padding=padding, 
            truncation=truncation
        )

        embeddings_dict = {} 

        # Iterate through the text columns and embed the text
        for col in text_cols:

            # Create the dataset
            text_dataset = BertDataset(df[col].values)

            # Create the dataloader (important to shuffle=False, keeping rows order)
            text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)

            # Iterate through the dataloader and get the embeddings
            with torch.no_grad():
                embeddings = []
                for batch in tqdm(text_dataloader, desc=f'Embedding {col}'):

                    # Get the embeddings
                    batch_embeddings = embedder.embed(batch['text']).cpu().numpy()

                    # Append the embeddings to the list
                    embeddings.append(batch_embeddings)
                
                # Create a new column in the DataFrame with the embeddings converted into a single
                # numpy array
                embeddings_dict[f'{col}_emb'] = np.vstack(embeddings)
    
    except Exception as e:
        logger.error(f'Error generating embeddings: {e}')
        raise

    #Create the embed config
    embed_config = {
        'max_length': max_length,
        'padding': padding,
        'truncation': truncation
    }

    # Return the DataFrame with the embeddings
    return embeddings_dict, embed_config
    