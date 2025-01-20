from config import model_settings, paths_settings, training_settings
from model.pipeline.preprocess import prepare_data
from model.pipeline.metrics import evaluate_logistic_regression

from loguru import logger
import pickle
from sklearn.linear_model import LogisticRegression

from datetime import datetime
import json


def build_model(model_version, data_percentage = 1.0):
    """
    Build, train, evaluate, and save a machine learning model.
    Parameters:
        model_version (str): The version identifier for the model.
        data_percentage (float, optional): The percentage of data to use for training. Defaults to 1.0.
    Returns:
        None
    This function performs the following steps:
        1. Prepares the data for training and testing.
        2. Trains the model using the training data.
        3. Evaluates the model using the test data.
        4. Creates metadata for the training process.
        5. Saves the trained model, encoders, and training metadata.
    """
    
    # Get the prepared data
    x_train, x_test, y_train, y_test, encoders = prepare_data(data_percentage)

    # Train the model
    model, train_time = train_model(x_train, y_train)

    # Evaluate the model
    logger.info('Evaluating the model.')
    metrics = evaluate_logistic_regression(
        model, 
        x_test, 
        y_test, 
        encoders['target_encoder'].classes_,
        model_version
    )

    # Create the training metadata
    training_metadata = {
        'model_name': model_settings.model_name,
        'model_version': model_version,
        'train_time': train_time,
        'data_percentage': data_percentage,
        'seed' : model_settings.seed,
        'penalty' : model_settings.penalty,
        'solver' : model_settings.solver,
        'max_iter' : model_settings.max_iter,
        'class_weight' : model_settings.class_weight,
        'numeric' : training_settings.numeric,
        'numeric_log' : training_settings.numeric_log,
        'categ' : training_settings.categ,
        'embed' : training_settings.embed
    }

    # Save the model
    logger.info('Saving the model.')
    save_output(
        model, 
        model_version, 
        encoders, 
        training_metadata,
        metrics
    )


def train_model(x_train, y_train):

    # Create the Logistic Regression model with the specified parameters
    model = LogisticRegression(
        penalty=model_settings.penalty,
        random_state=model_settings.seed,
        max_iter=model_settings.max_iter,
        solver=model_settings.solver,
        class_weight=model_settings.class_weight,
        multi_class=model_settings.multi_class
    )

    # Fit the model
    try:
        logger.info('Training the model.')

        start_time = datetime.now()
        model.fit(x_train, y_train)

        train_time = (datetime.now() - start_time).total_seconds()

        logger.info(f'Training completed in {train_time} seconds.')
    except Exception as e:
        logger.error(f'Error training the model: {e}')
        raise
    
    # Return the trained model
    return model, train_time

def save_output(model, model_version, encoders, training_metadata, metrics):

    model_path = f'{paths_settings.model_output_dir}/{model_settings.model_name}_{model_version}.pkl'
    encoders_path = f'{paths_settings.model_output_dir}/encoders_{model_version}.pkl'
    metadata_path = f'{paths_settings.model_output_dir}/train_metadata_{model_version}.json'
    metrics_path = f'{paths_settings.model_output_dir}/metrics/metrics_{model_version}.json'
    
    # Save the model to a file
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    logger.info(f'Model saved to {model_path}')
    logger.info(f'Encoders saved to {encoders_path}')
    logger.info(f'Training metadata saved to {metadata_path}')
    logger.info(f'Metrics saved to {metrics_path}')
