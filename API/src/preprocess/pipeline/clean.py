import re

from loguru import logger

from config import preprocess_settings

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


def create_features(transformed_input):
    """
    Categorizes the input features based on the bins defined in the preprocess_settings.
    Parameters:
        transformed_input (dict): The input data to be categorized.
    Returns:
        dict: A dictionary containing the features for each column categorized.
    """
    feature_mappings = {}
    for feature in preprocess_settings.features_mapping.keys():

        # Categorize the column
        # Get the bins available for the feature
        feature_dict = preprocess_settings.features_mapping[feature]
        col_bins = list(feature_dict.keys())

        # Get the input value, converting to a string to match json keys string format
        input = str(transformed_input[feature])

        # Categorize it. Add its value if it is on the beans, otherwise its category is 'Other'
        feature_mappings[f'{feature}_cat'] = feature_dict[input] if input in col_bins else 'Other'
        
    return feature_mappings

def clean_input(transformed_input):
    """
    Receives a DataFrame and:
    - Cleans the text columns using the clean_text function.
    - Adds the length of the text columns.
    - Creates additional features based on the preprocess_settings.
    Parameters:
        transformed_input (pandas.DataFrame): The input dict to be cleaned.
    Returns:
        dict: A dictionary containing the cleaned features.
    """

    # Dictionary with the clean features
    features_clean = {}

    try:

        # Apply cleaning function to text columns
        logger.info('Cleaning text columns.')
        for col in preprocess_settings.text_to_embed:
            # Clean
            features_clean[col] = clean_text(transformed_input[col])
            # Add the length of the text (except for the features column. That is already given)
            if col != 'features':
                features_clean[f'len_{col}'] = len(transformed_input[col])
            else:
                features_clean['num_features'] = transformed_input['num_features']

        # Create features
        logger.info('Creating features.')
        features_mapping = create_features(transformed_input)

        # Add the features mapping to the features_clean dictionary
        features_clean.update(features_mapping)

    except Exception as e:
        logger.error(f'Error cleaning data: {e}')
        raise

    return features_clean