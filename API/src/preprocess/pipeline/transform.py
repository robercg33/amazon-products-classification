from loguru import logger

from schema.amazon_product import AmazonProduct

def get_transformed_input(amazon_product):
    """
    Transforms the input Amazon product data into a dictionary with specific features.
    Args:
        amazon_product (object): An object representing an Amazon product with the following attributes:
            - title (str): The title of the product.
            - description (list of str): A list of strings representing the product description.
            - feature (list of str): A list of strings representing the product features.
            - also_buy (list): A list of products that are frequently bought together.
            - also_view (list): A list of products that are frequently viewed together.
            - image (list): A list of images associated with the product.
    Returns:
        dict: A dictionary containing the transformed input with the following keys:
            - 'title' (str): The title of the product.
            - 'description' (str): The concatenated and stripped product description.
            - 'features' (str): The concatenated and stripped product features.
            - 'num_also_buy' (int): The number of products frequently bought together.
            - 'num_also_view' (int): The number of products frequently viewed together.
            - 'num_images' (int): The number of images associated with the product.
            - 'num_features' (int): The number of features associated with the product.
    """

    # Create a dictionary with the transformed input
    transformed_input = {}

    try:
        # Add the text features
        transformed_input['title'] = amazon_product.title
        transformed_input['description'] = ('\n'.join(amazon_product.description)).strip()
        transformed_input['features'] = ('\n'.join(amazon_product.feature)).strip()

        # Add the numerical features
        transformed_input['num_also_buy'] = len(amazon_product.also_buy)
        transformed_input['num_also_view'] = len(amazon_product.also_view)
        transformed_input['num_images'] = len(amazon_product.image)
        transformed_input['num_features'] = len(amazon_product.feature)

    except Exception as e:
        logger.error(f'Error transforming input: {e}')
        raise

    # Return the transformed input
    return transformed_input
