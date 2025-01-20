import re

from loguru import logger

def clean_text(text):
    """
    Cleans text by removing HTML tags, non-printable characters, 
    excessive whitespace (including tabs/newlines), and special symbols 
    (but keeps punctuation like . , ! ? ; : ' ").
    """

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-printable characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove excessive whitespace (including tabs/newlines)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove special symbols (keeping letters, digits, whitespace, punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"]+', '', text)

    return text


def create_features(df, features_list, category_threshold=0.01):
    """
    Create categorical features for the specified columns in the DataFrame based on a threshold.
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        features_list (list of str): List of column names to create categorical features for.
        category_threshold (float, optional): The threshold for category frequency. 
                                            Categories with a frequency lower than this threshold 
                                            will be grouped into 'Other'. Default is 0.01.
    Returns:
        dict: A dictionary containing the feature mappings for each column.
    """
    feature_mappings = {}
    for col in features_list:

        # Get bins filtering by threshold
        coul_count = df.num_also_buy.value_counts(normalize=True)
        col_bins = coul_count[coul_count > category_threshold].index.tolist()

        # Categorize the column
        df[f'{col}_cat'] = df[col].apply(lambda x: str(x) if x in col_bins else 'Other')

        # Add the feature mapping
        feature_mappings[col] = {i: str(i) for i in col_bins}
        
    return feature_mappings



def clean_df(df, text_cols, features_list):
    """
    Cleans the DataFrame by applying the clean_text function to specified text columns 
    and creating additional features.
    Parameters:
        df (pandas.DataFrame): The DataFrame to be cleaned.
        text_cols (list of str): List of column names in the DataFrame that contain text data to be cleaned.
        features_list (list): List of features to be created in the DataFrame.
    Returns:
        pandas.DataFrame: The cleaned DataFrame with additional features.
        dict: A dictionary containing the feature mappings for each column.
    """

    try:

        # Apply cleaning function to text columns
        logger.info('Cleaning text columns...')
        for col in text_cols:
            df[col] = df[col].apply(clean_text)
            df[f'len_{col}'] = df[col].apply(len)

        # Create features
        logger.info('Creating features...')
        features_mapping = create_features(df, features_list)

    except Exception as e:
        logger.error(f'Error cleaning DataFrame: {e}')
        raise

    return df, features_mapping