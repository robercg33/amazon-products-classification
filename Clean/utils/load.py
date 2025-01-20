import boto3
import gzip
import json
import pandas as pd
from io import BytesIO

from loguru import logger
import numpy as np

def load_data(path, sample_percentage=1.0):
    """
    This method retrieves a gzipped JSONL file from the specified path,
    processes each JSON object to extract relevant fields, and returns a pandas DataFrame.
    Parameters:
    - path (str): The path to the gzipped JSONL file.
    - sample_percentage (float, optional): The fraction of records to load. Default is 1.0.
    Returns:
    - pd.DataFrame: A pandas DataFrame containing the records with the following columns:
        - title (str): The title of the product.
        - main_cat (str): The main category of the product.
        - num_also_buy (int): The number of 'also_buy' items.
        - num_also_view (int): The number of 'also_view' items.
        - num_images (int): The number of images.
        - num_description (int): The number of descriptions.
        - num_features (int): The number of features.
        - description (str): The concatenated description text.
        - len_description (int): The length of the concatenated description.
        - features (str): The concatenated features text.
        - len_features (int): The length of the concatenated features.
    """

    try:
        # Load the gzipped JSONL file
        records = []

        # Apply preprocessing to each JSON object
        logger.info('Processing JSON objects...')
        with gzip.open(path, 'r') as f:
            for line in f:
                product = json.loads(line)

                row = {}

                # Target
                row['main_cat'] = product.get('main_cat', '')

                # Text fields
                row['title'] = product.get('title', '')
                row['description'] = ('\n'.join(product.get('description', []))).strip()
                row['features'] = ('\n'.join(product.get('feature', []))).strip()

                # Data fields
                row['num_also_buy'] = len(product.get('also_buy', []))
                row['num_also_view'] = len(product.get('also_view', []))
                row['num_images'] = len(product.get('image', []))
                row['num_description'] = len(product.get('description', []))
                row['num_features'] = len(product.get('feature', []))

                records.append(row)

        # Return the DataFrame
        df = pd.DataFrame(records)

        # Sample the data if needed, stratifying by the target column
        if sample_percentage < 1.0:
            grouped = df.groupby('main_cat', group_keys=False)
            sample = grouped.apply(lambda x: x.sample(frac=sample_percentage))
            df = sample.reset_index(drop=True)

        logger.info(f'Loaded {len(df)} records')

    except Exception as e:
        logger.error(f'Error parsing data: {e}')
        raise
    
    return df

def save_data_to_s3(df, cleaning_config, embeddings_dict, aws_role_arn, aws_region, bucket_name, key):
    """
    Save a pandas DataFrame and embeddings dictionary to an S3 bucket in Parquet format.
    Parameters:
        df (pandas.DataFrame): The DataFrame to save.
        cleaning_config (dict): A dictionary containing the cleaning configuration.
        embeddings_dict (dict): A dictionary containing the embeddings for each text column.
        aws_role_arn (str): The Amazon Resource Name (ARN) of the AWS role to assume.
        aws_region (str): The AWS region where the S3 bucket is located.
        bucket_name (str): The name of the S3 bucket.
        key (str): The key (path) within the S3 bucket where the DataFrame will be saved.
    Raises:
        Exception: If there is an error assuming the AWS role or uploading the DataFrame to S3.
    """

    # Create the sts client to get the role for reading/writing to S3
    sts_client = boto3.client('sts')

    # Assume the role
    try:
        response = sts_client.assume_role(
            RoleArn=aws_role_arn,
            RoleSessionName='s3-loader-session'
        )
    except Exception as e:
        logger.error(f'Error assuming role: {e}')
        raise

    # Create the S3 client
    s3_client = boto3.client(
        's3', 
        region_name=aws_region,
        aws_access_key_id=response['Credentials']['AccessKeyId'],
        aws_secret_access_key=response['Credentials']['SecretAccessKey'],
        aws_session_token=response['Credentials']['SessionToken']
    )

    # Save the DataFrame to S3
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    logger.info(f'parquet file size: {buffer.getbuffer().nbytes / (1024 * 1024):.2f} MB')
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=buffer.getvalue()
    )

    # Save the cleaning configuration to S3
    cleaning_config_key = key.replace('.parquet', '_config.json')
    s3_client.put_object(
        Bucket=bucket_name,
        Key=cleaning_config_key,
        Body=json.dumps(cleaning_config)
    )

    # Save the embeddings to S3
    embeddings_key = key.replace('.parquet', '_embeddings.npz')

    # Compression is used to reduce the size of the embeddings file
    buffer = BytesIO()
    np.savez_compressed(buffer, **embeddings_dict)
    buffer.seek(0)

    logger.info(f'embeddings file size: {buffer.getbuffer().nbytes / (1024 * 1024):.2f} MB')

    # Upload the compressed embeddings file to S3
    s3_client.put_object(
        Bucket=bucket_name,
        Key=embeddings_key,
        Body=buffer.getvalue()
    )

    logger.info(f'Saved DataFrame to s3://{bucket_name}/{key}')
    logger.info(f'Saved cleaning configuration to s3://{bucket_name}/{cleaning_config_key}')
    logger.info(f'Saved embeddings to s3://{bucket_name}/{embeddings_key}')