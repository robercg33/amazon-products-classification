from model.pipeline.model import build_model

from loguru import logger

import argparse


def main(model_version, data_percentage = 1.0):
    """
    Main function to train the model.
    Parameters:
        model_version (str): The version of the model to be trained.
        data_percentage (float, optional): The percentage of the data to be used for training. Defaults to 1.0.
    Returns:
        None
    """

    logger.info('Training the model...')
    
    build_model(
        model_version=model_version,
        data_percentage=data_percentage
    )

    logger.info('Process finished.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--data_percentage', type=float, default=1.0, help='The percentage of data to use for training')
    parser.add_argument('--model_version', type=str, required=True, help='The version identifier for the model')

    args = parser.parse_args()

    # Execute the main function
    try:
        main(
            model_version=args.model_version,
            data_percentage=args.data_percentage
        )  
    except Exception as e:
        logger.error(f'Error during execution: {e}')
        exit(1)