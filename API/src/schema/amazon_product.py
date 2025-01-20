'''
Schema for the input raw data.
'''
from typing import List

from pydantic import BaseModel


class AmazonProduct(BaseModel):
    '''
    Amazon Product schema.

    also_buy: List of products that are also bought.
    also_view: List of products that are also viewed.
    description: List of strings containing the description of the product.
    feature: List of strings containing the features of the product.
    image: List of strings containing the image URLs of the product.
    title: Title of the product.
    '''

    also_buy: List[str]
    also_view: List[str]
    description: List[str]
    feature: List[str]
    image: List[str]
    title : str