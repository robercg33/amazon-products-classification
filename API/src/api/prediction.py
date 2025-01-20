from flask import Blueprint, request, jsonify, abort
from loguru import logger
from pydantic import ValidationError

from schema.amazon_product import AmazonProduct
from services import model_inference_service


# Create a Blueprint for the prediction API
prediction_bp = Blueprint('prediction', __name__)


## Endpoint - POST
@prediction_bp.route('/predict', methods=['POST'])
def get_prediction():
    """
    Get a prediction for an Amazon product.
    Returns:
        str: The predicted label for the Amazon product.
    """
    try:
        # Get the Amazon product data from the request
        amazon_product = AmazonProduct(**request.json)

        # Get the prediction
        prediction = model_inference_service.predict(amazon_product)
        logger.info(f'Prediction: {prediction}')

    except ValidationError as ve:
        abort(400, f'Invalid input: {ve}')
    except Exception as e:
        abort(500, f'An error occurred: {e}')

    return jsonify({'prediction': prediction}), 200