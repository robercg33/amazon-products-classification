from model.pipeline.collection import load_data
from config import training_settings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


def prepare_data(data_percentage=1.0):
    """
    Prepares the data for training and testing by performing the following steps:
    1. Loads the data.
    2. Creates the feature matrix (X) and target vector (y).
    3. Encodes the target variable.
    4. Splits the data into training and testing sets.
    5. Preprocesses numeric and categorical features.
    6. Concatenates numeric, categorical, and embedding features.
    7. Creates a dictionary with the encoders used for preprocessing.
    Parameters:
        data_percentage (float, optional): The fraction of the data to use. Defaults to 1.0.
    Returns:
        x_train (np.ndarray): Preprocessed training features.
        x_test (np.ndarray): Preprocessed testing features.
        y_train (np.ndarray): Training target variable.
        y_test (np.ndarray): Testing target variable.
        encoders (dict): Dictionary containing the encoders for numeric, categorical, and target variables.
    """

    # Load the data
    logger.info('Loading data...')
    try:
        data = load_data(data_percentage)

        # Create the X and y varaibles (features and target)
        x = data[
            training_settings.numeric +
            training_settings.categ +
            training_settings.embed
        ]
        y = data[training_settings.target]

        # Encode the target variable and get the encoder
        logger.info('Encoding target variable.')
        y_encoded, target_encoder = _enconde_target(y)

        # Split into train and test
        logger.info('Splitting data into training and testing sets.')
        x_train, x_test, y_train, y_test = _split_train_test(
            x, 
            y_encoded, 
            training_settings.test_size, 
            training_settings.data_seed
        )

        ### Preprocess data
        logger.info('Preprocessing data.')
        # Numeric
        logger.info('Preprocessing - Numeric')
        x_train_numeric, x_test_numeric, scaler = _preprocess_numeric(
            train_df=x_train,
            test_df=x_test,
            numeric_vars=training_settings.numeric,
            log_numeric=training_settings.numeric_log
        )

        # Categorical
        logger.info('Preprocessing - Categorical')
        x_train_categorical, x_test_categorical, encoder = _preprocess_categorical(
            train_df=x_train,
            test_df=x_test,
            categ_vars=training_settings.categ
        )

        # Embeddings
        # The sum of the embeddings of the three columns will be used as the embedding features
        x_train_embed = np.sum([np.vstack(x_train[col].values) for col in training_settings.embed], axis=0)
        x_test_embed = np.sum([np.vstack(x_test[col].values) for col in training_settings.embed], axis=0)

        ## Join the data
        # Order -> numeric, categorical, embeddings
        x_train = np.hstack([x_train_numeric, x_train_categorical, x_train_embed])
        x_test = np.hstack([x_test_numeric, x_test_categorical, x_test_embed])

    except Exception as e:
        logger.error(f'Error preprocessing data: {e}')
        raise

    # Create a dictionary with the encoders
    encoders = {
        'numeric_scaler': scaler,
        'categorical_encoder': encoder,
        'target_encoder': target_encoder
    }

    logger.info('Data preprocessing completed.')

    # Return data and encoders
    return x_train, x_test, y_train, y_test, encoders


def _enconde_target(target):
    '''
    Parameters:
        target (array-like): The target variable to be encoded.
    Returns:
        tuple: A tuple containing the encoded target variable (array) and the fitted LabelEncoder instance.
    '''

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(target)

    return y_encoded, target_encoder

def _split_train_test(x, y, test_size, seed):
    """
    Splits the data into training and testing sets.
    Parameters:
        x (array-like): Features dataset.
        y (array-like): Target dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing four elements:
            - x_train (array-like): Training features.
            - x_test (array-like): Testing features.
            - y_train (array-like): Training targets.
            - y_test (array-like): Testing targets.
    """

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=test_size, 
        random_state=seed, 
        stratify=y
    )

    return x_train, x_test, y_train, y_test

def _preprocess_numeric(train_df, test_df, numeric_vars, log_numeric=[]):
    """
    Preprocess numeric variables in the training and test datasets.
    Parameters:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset.
        numeric_vars (list): List of numeric variable names to be scaled.
        log_numeric (list, optional): List of numeric variable names to apply log1p transformation. Defaults to an empty list.
    Returns:
        tuple: A tuple containing:
            - train_numeric_df (pd.DataFrame): The transformed training dataset with scaled numeric variables.
            - test_numeric_df (pd.DataFrame): The transformed test dataset with scaled numeric variables.
            - scaler (StandardScaler): The fitted StandardScaler object.
    """

    # Apply log1p to numeric values that requires it
    for col in log_numeric:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])

    # Standard Scale Numeric Variables
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_vars]) # Fit on the training set only

    # Transform both train and test data
    train_numeric = scaler.transform(train_df[numeric_vars])
    test_numeric = scaler.transform(test_df[numeric_vars])

    # Create DataFrames
    train_numeric_df = pd.DataFrame(
        train_numeric,
        columns=numeric_vars,
        index=train_df.index
    )
    test_numeric_df = pd.DataFrame(
        test_numeric,
        columns=numeric_vars,
        index=test_df.index
    )

    # Return the transformed data and the 
    return train_numeric_df, test_numeric_df, scaler

def _preprocess_categorical(train_df, test_df, categ_vars):
    """
    Preprocesses categorical variables in the given training and testing dataframes using one-hot encoding.
    Parameters:
        train_df (pd.DataFrame): The training dataframe containing the categorical variables.
        test_df (pd.DataFrame): The testing dataframe containing the categorical variables.
        categ_vars (list of str): List of column names in the dataframes that are categorical variables.
    Returns:
        tuple: A tuple containing:
            - train_categorical_df (pd.DataFrame): The transformed training dataframe with one-hot encoded categorical variables.
            - test_categorical_df (pd.DataFrame): The transformed testing dataframe with one-hot encoded categorical variables.
            - ohe (OneHotEncoder): The fitted OneHotEncoder instance.
    """

    # One-Hot Encode Categorical Variables
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    ohe.fit(train_df[categ_vars])

    train_categorical = ohe.transform(train_df[categ_vars])
    test_categorical = ohe.transform(test_df[categ_vars])

    train_categorical_df = pd.DataFrame(
        train_categorical,
        columns=ohe.get_feature_names_out(categ_vars),
        index=train_df.index
    )
    test_categorical_df = pd.DataFrame(
        test_categorical,
        columns=ohe.get_feature_names_out(categ_vars),
        index=test_df.index
    )

    return train_categorical_df, test_categorical_df, ohe
