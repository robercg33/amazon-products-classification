
from embed.embeddings import get_embeddings_from_texts
from utils.clean import clean_df
from utils.load import load_data, save_data_to_s3

import os
import argparse

from loguru import logger
from dotenv import load_dotenv
import pandas as pd

#Load the environment variables
load_dotenv()

# The text columns to embed
text_cols = ['title', 'description', 'features']

# Numeric features to categorize
features_list = ['num_also_buy', 'num_also_view', 'num_images']

path = 'data/amz_products_small.jsonl.gz'

def main(output_route, batch_size, max_length, sample_percentage=1.0):

    # Load the data from the input file
    logger.info(f'Parsing data from {path}...')
    data = load_data(
        path=path,
        sample_percentage=sample_percentage
    )

    # Clean the DataFrame
    logger.info('Starting data cleaning.')
    cleaned_data, features_mapping = clean_df(
        data, 
        text_cols, 
        features_list
    )

    logger.info(f'Clean DataFrame shape: {cleaned_data.shape}')

    # Get the embeddings
    logger.info('Starting embeddings generation.')
    embeddings_dict, embed_config = get_embeddings_from_texts(
        cleaned_data,
        text_cols,
        batch_size=batch_size,
        max_length=max_length
    )

    logger.info(f'Clean DataFrame shape: {cleaned_data.shape}')

    # Save the JSON to a json config
    cleaning_config = {
        'features_mapping': features_mapping,
        'embed_config': embed_config
    }

    # Save the DataFrame to a parquet in S3
    logger.info(f'Saving DataFrame to s3://{os.getenv('S3_BUCKET_NAME')}/{output_route}...')
    save_data_to_s3(
        df=cleaned_data,
        cleaning_config=cleaning_config,
        embeddings_dict=embeddings_dict,
        aws_region=os.getenv('AWS_REGION'),
        aws_role_arn=os.getenv('AWS_ROLE_ARN'),
        bucket_name=os.getenv('S3_BUCKET_NAME'),
        key=output_route
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--output_route', type=str, required=True, help='The output route for the data')
    parser.add_argument('--batch_size', type=int, required=True, help='The batch size for processing')
    parser.add_argument('--max_length', type=int, required=True, help='The maximum length for text processing')
    parser.add_argument('--data_percentage', type=float, default=1.0, help='The percentage of data to process')

    args = parser.parse_args()

    # Execute the main function
    try:
        main(
            output_route=args.output_route,
            batch_size=args.batch_size,
            max_length=args.max_length,
            sample_percentage=args.data_percentage
        )
    except Exception as e: 
        logger.error(f'Error during execution: {e}')
        exit(1)

    logger.info('Execution completed successfully.')

