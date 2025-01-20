# Product Classification

A comprehensive solution for product classification using machine learning and deep learning techniques. The project explores:

- Data preprocessing and feature engineering for both text and non-text data.
- Implementation and exploration of models including logistic regression, XGBoost, and custom neural networks using PyTorch.
- Usage of DistilBERT for text embedding and fine-tuning for multi-class classification.
- Deployment-ready training and inference pipelines with clear documentation.

The aim of this project is to explore the dataset and build a ML model capable of performing a multi-class classification problem, building:
- Data cleaning module
- Training module
- API module

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [EDA & Cleaning](#eda-and-cleaning)
3. [Feature Engineering](#feature-engineering)
4. [Feature preprocessing](#feature-preprocessing)
5. [Model Selection](#model-selection)
6. [Model Training](#model-training)
7. [Model Metrics](#model-metrics)
8. [API](#api)
9. [Improvements & Next Steps](#pimprovements-&-next-steps)

## Dataset Description

The dataset is a simplified version of [Amazon 2018](https://jmcauley.ucsd.edu/data/amazon/), only containing products and their descriptions.

The dataset consists of a jsonl file where each is a json string describing a product.

Example of a product in the dataset:
```json
{
 "also_buy": ["B071WSK6R8", "B006K8N5WQ", "B01ASDJLX0", "B00658TPYI"],
 "also_view": [],
 "asin": "B00N31IGPO",
 "brand": "Speed Dealer Customs",
 "category": ["Automotive", "Replacement Parts", "Shocks, Struts & Suspension", "Tie Rod Ends & Parts", "Tie Rod Ends"],
 "description": ["Universal heim joint tie rod weld in tube adapter bung. Made in the USA by Speed Dealer Customs. Tube adapter measurements are as in the title, please contact us about any questions you 
may have."],
 "feature": ["Completely CNC machined 1045 Steel", "Single RH Tube Adapter", "Thread: 3/4-16", "O.D.: 1-1/4", "Fits 1-1/4\" tube with .120\" wall thickness"],
 "image": [],
 "price": "",
 "title": "3/4-16 RH Weld In Threaded Heim Joint Tube Adapter Bung for 1-1/4&quot; Dia by .120 Wall Tube",
 "main_cat": "Automotive"
}
```

### Field description
- also_buy/also_view: IDs of related products
- asin: ID of the product
- brand: brand of the product
- category: list of categories the product belong to, usually in hierarchical order. **WE WON'T BE USING THIS VARIABLE AS FEATURE**
- description: description of the product
- feature: bullet point format features of the product
- image: url of product images (migth be empty)
- price: price in US dollars (might be empty)
- title: name of the product
- main_cat: main category of the product

`main_cat` can have one of the following values:
```json
["All Electronics",
 "Amazon Fashion",
 "Amazon Home",
 "Arts, Crafts & Sewing",
 "Automotive",
 "Books",
 "Buy a Kindle",
 "Camera & Photo",
 "Cell Phones & Accessories",
 "Computers",
 "Digital Music",
 "Grocery",
 "Health & Personal Care",
 "Home Audio & Theater",
 "Industrial & Scientific",
 "Movies & TV",
 "Musical Instruments",
 "Office Products",
 "Pet Supplies",
 "Sports & Outdoors",
 "Tools & Home Improvement",
 "Toys & Games",
 "Video Games"]
```

[Download dataset](https://drive.google.com/file/d/1xFQY8Z5FyYnFPrGTg7oUkXw3CCryInov/view?usp=drive_link)

Data can be read directly from the gzip file as:
```python
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
```

## EDA and Cleaning
In this part, I briefly explain the EDA performed and the decided cleaning steps and features selection for the model.

### Non-text features

The following "non-text" fields were provided:

- `also_buy` (**USED**): Not directly useful, converted into a variable indicating the length of the list.
- `also_view` (**USED**): Same as the previous one, converted to a new feature indicating the length of the list.
- `asin` (**DISCARDED**): Unique product identifier, not useful. *Discarded*
- `brand` (**DISCARDED**): Brand of the product. At first, I wanted to use it as another categorical variable taking the most relevant marks and grouping the rest into a 'fallback' category, but it turned out that the fallback category encompassed nearly all the data.
- `price` (**DISCARDED**): Cleaned and parsed into number if they match a price format. After that, realised that around half of the records didn't match the price format or were empty, so I ended up not using it. *Discarded*.
- `image` (**USED**): List of URLs containing images. The decision has been to count the number of images on the list, as done with previous lists, creating a new feature.

### Text features

The following text fields were given: 

- `description` (**USED**): List of strings containing the description of the product. **Joined into a single string**.
- `features` (**USED**): List of strings containing features of the product. **Joined into a single string**.
- `title` (**USED**): String representing the product title.

After analyzing, I create a **cleaning text function**, that does the following:

1. Remove HTML tags. This is because I saw a lot of text containing them, and are not useful for the purpose of the exercise.
2. Remove non-printable characters.
3. Remove excessive whitespace.
4. Remove special symbol, keeping only letters, digits and regular punctuation symbols.

### Target

The target variable `main_cat`, has every record as a non-null value, and, the one and most important thing to say is that its distribution is not balanced, as it can be seen in the following table:

| Main Category               | Count  |
|-----------------------------|--------|
| Tools & Home Improvement    | 74358  |
| Automotive                  | 73434  |
| Arts, Crafts & Sewing       | 72556  |
| Toys & Games                | 72034  |
| Office Products             | 71681  |
| Amazon Home                 | 71362  |
| Grocery                     | 70184  |
| Sports & Outdoors           | 70177  |
| Books                       | 69685  |
| Computers                   | 67156  |
| Movies & TV                 | 60608  |
| Amazon Fashion              | 59747  |
| Cell Phones & Accessories   | 59432  |
| Pet Supplies                | 57490  |
| Industrial & Scientific     | 55503  |
| All Electronics             | 52878  |
| Digital Music               | 48319  |
| Camera & Photo              | 36687  |
| Musical Instruments         | 35873  |
| Home Audio & Theater        | 33401  |
| Video Games                 | 21736  |
| Health & Personal Care      | 14116  |

Either way, every class is well represented, but this should later have to be taken into account.


## Feature Engineering

The actual features that will be used for the model, derived from the original cleaned features.

### Created features

- `num_also_buy_cat`: length of "also_buy" list, *categorized into representative bins*.
- `num_also_view_cat`: length of "also_view" list, *categorized into representative bins*.
- `num_images_cat`: length of "image" list, *categorized into representative bins*.
- `len_title`: length of the *cleaned* `title`.
- `len_description`: length of the *cleaned and joined* `description`.
- `num_features`: length of the LIST indicated in the input variable `feature`. Instead of using the length of the features joined string, I used this as each element on the list is supposed to indicate a feature of the product. As I already had this new feature created, I didn't create `len_features` as both variables would be highly correlated.

### Embeddings features

For adding the texts fields as features, we will create embeddings using a pre-trained [distil-BERT](https://huggingface.co/distilbert/distilbert-base-uncased) model, using the following directives:

1. Truncate to *`max_length`* parameter, the rest of the text would be not used for embeddings.
2. Padding to *`max_length`* parameter, for consistency when getting the embeddings.

Each of the text features [`title`, `description`, `features`] will have its own BERT embedding vector generated:

- `title_emb`: embedding vector for the title cleaned string.
- `description_emb`: embedding vector for the description cleaned string.
- `features_emb`: embedding vector for the features cleaned string.

**IMPORTANT** - The embeddings generated are then summed up to generate a **single embeddings feature** with the combined information of the three fields:

- By doing that, we are avoiding records with a default embedding vector because there are no rows in the whole dataset in which the three texts columns are empty, so by summing them we are ensuring to always have a meaningfull embedding vector for each row.

- This also, reduces the number of embeddings features to just one 768 dimension (BERT embed dim) vector, instead of three, speeding up inference. 


## Feature Preprocessing

Next step is to create a cleaned and ready-to-use dataset from the input data following the directories specified on [1](#eda-and-cleaning) and [2](#feature-engineering).

The ideal scenario would be to implement the data collecting and preprocessing steps in the same module as the training step, but as I need to generate embeddings from the texts, it would take too long to do that in a machine without a GPU, so I decided to separate the data preparation in another module to run in a cloud machine with GPU, just with the purpose of speeding up the embedding generation.

The code is located in the [Clean](/Clean/) module, and it performs the following:

1. Reads the data from the gzipped file and parses it. A data percentage parameter is added to process just part of the data, as the whole dataset would generate an incredible heavy embeddings file. Important to note, this sampling is done stratifying by the target variable `main_cat`, ensuring the distribution of the sampled data mimics the distribution of the target on the original data.
2. Cleans the text columns and preprocess the rest of features as is indicated previously.
3. Generate embeddings using a distil-BERT model for each text feature.
4. Stores the cleaned and generated features in a dataframe in parquet format and the embeddings in a `.npz` file (numpy compressed file). The module writes the outputs to an AWS S3 bucket. **This is done to facilitate data retrieval from the cloud machine**

The cloud machine used was hosted by [vastai](https://vast.ai/), as I am already familiarized with this environment. They offer super cheap GPU cloud machines.

## Model Selection

For selecting the model, I tried different models with a very tiny fraction of the provided dataset. The tests and training are sketched in this [notebook](/Notebooks/metrics.ipynb). The models I tried were:

- Logistic Regression
- XGBoost
- Neural Network

The three of them seemed to perform similar, so I opted for the **Logistic Regression** for the following reasons:

1. Resources: A logistic regression takes very little resources to train and deploy, and do not need a GPU for faster training/inference, like a NN would.
2. Simplicity: Is a very simple model, if needed, we could interpret the coefficients of each variable to understand how they relate to the target (this would be more difficult with the embeddings, but can be done with the rest of variables).
3. Handle imbalances: As we saw, the dataset is imbalanced, which depending on the case can be a problem or not. If we decide to act on it, a logistic regression offers a simple way to manage it with the *`class_weight`* parameter.

## Model Training

The model training module, located on [Training](/Training/), just takes the clean data and the embeddings generated with the [Clean](/Clean/) module, and trains a logistic regression, with the parameters and configuration specified in the config files ([model](/Training/config/res/model_config.json) & [train](/Training/config/res/training_config.json)).

The data is already included in the [/data](/Training/data/) path, containig a parquet file with 124842 records cleaned from the original provided dataset and its embeddings in a `.npz` file.

- The model outputs (at [/model/models](/Training/model/models/)) several things:
    - *modelname_modelversion.pkl*: Pickle file named after the model name and version specified on the config file, containing the trained model.
    - *encoders_modelversion.pkl*: Pickle file containing the encoders used for features and target in the model training.
    - *train_metadata.json*: JSON file containing metadata of the training process (training time, random states used, parameters...)
    - *metrics/*: A directory with two plots (roc curves and confusion matrix) and a JSON file containing the metrics performed to the model after training.

For running, just building the docker image and running it would be sufficient:

`docker build -t <image-name> <module_path>`

And for running the image, I a simple docker run command should be sufficient, mounting the output folder from the container into a local folder for storing the output:

`docker run -v "<local_path>:/app/model/models" <image-name>`


## Model Metrics

I trained two Logistic Regression models on the dataset:

- One targeting the class imbalance
- Another without taking that into account

The results and explanations can be found on the [metrics](/Notebooks/metrics.ipynb) notebook.

The selected model to use was the Non-Balanced Logistic Regression.

## API

For the API architecture, the following schema was used: 

- src
    - api: API routes (in this case, only predict)
    - config: Configuration files and environment setup
    - models: Pre-trained models and encoder files
    - peprocess: Preprocess logic for raw input data
    - schema: Pydantic model for input parsing
    - services: Services that manage the API logic (inference of the LR model in this case)
- test (*not implemented*): Here, the unit and integration tests for the API would be placed.

The dockerfile exposes the API on the port 5000 of the container. For deploying the API in a localhost, it is only necessary to build the image:

`docker build -t <image-name> <module_path>`

And running in a container, mapping the 5000 port of the container with a port on localhost

`docker run -p host_port:5000 <image-name>`

For calling the API, a POST request should be then sent to

**localhost:host_port/predict**

With a JSON on the body containing **ONLY**:

- The input variables used from the original data:
    - also_buy
    - also_view
    - title
    - feature
    - description
    - image

As the following example:

```json
{
    "also_buy" : [],
    "also_view" : ["B00WTHIZT0", "B010LVBVKA", "B0054IH8C6", "B0082C62BO"],
    "title" : "Travel Smart by Conair Neck Security Pouch",
    "feature" : [],
    "description" : ["Always keep your belonging close and secure with the Travel Smart by Conair Neck Security Pouch. This security pouch features a large pocket with Velcro closure and a zippered compartment to keep your belongings secure in one place. It has a super soft binding for extra comfort and is made of lightweight, washable cotton fabric. Easily adjust the cord to wear around neck or over shoulder."],
    "image": ["https://images-na.ssl-images-amazon.com/images/I/31j9QJALRhL._SS40_.jpg"]
}
```

## Improvements & Next Steps

This project has been built in very little time, so there is a lot of room for improvement. Some next steps could be:

- **Try other models** and other approaches, like **finetuning a language model directly for the classification problem**.
- **Try more hyperparameter tunning**
- **Make the trainig script more parametrizable** instead of fixing the parameters on the config file.
- **Cross validate the results**, in the same dataset, with built in functions from *scikit-learn*
- **Add unit and integration tests for the api**, using *pytest* or any other testing library.
- **Add more error handling** for each of the modules, as the one implemented is a very simple one.
