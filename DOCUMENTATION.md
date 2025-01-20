# Project documentation

## Overview

This document explain the processes, descisions and workflows used on this project

## Table of Contents
1. [EDA & Cleaning](#eda-and-cleaning)
2. [Feature Engineering](#feature-engineering)
3. [Feature preprocessing](#feature-preprocessing)
4. [Model Selection](#model-selection)
5. [Model Training](#model-training)
6. [Model Metrics](#model-metrics)
7. [API](#api)
8. [Possible Improvements](#possible-improvements)
9. [Questions Asked](#questions-asked)


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

## Possible Improvements

This project has been built in very little time, so there is a lot of room for improvement. Some next steps could be:

- **Try other models** and other approaches, like finetuning a language model directly for the classification problem, or other approaches.
- **Try more hyperparameter tunning**
- **Make the trainig script more parametrizable** instead of fixing the parameters on the config file.
- **Cross validate the results**, in the same dataset, with built in functions from *scikit-learn*
- **Add unit and integration tests for the api**, using *pytest* or any other testing library.
- **Add more error handling** for each of the modules, as the one implemented is a very simple one.

## Question Asked

- What would you change in your solution if you needed to predict all the categories?
    - In this case, we would be talking about a multi-label problem instead of a single-label one. Then, a Logistic Regression model would not be valid for this purpose. A more suitable approach then would be a Deep Learning model, like a neural network, where the output instead of using a softmax and taking the argmax (like you would do in a single-label problem), we would need to get the **separate probability for each class**, using a sigmoid activation function for the output layer.
    - Also, the target variable should be converted to a ONE-HOT vector of length *N_CLASSES* where each element represents if the class is present.
    - The loss function used should also change, for example using the *Binary Cross Entropy* loss

- How would you deploy this API on the cloud?

    - The deployment for the API on the cloud would depend on the type of model and the type of task that this model is going to serve. The steps to proceed:
        1. Upload the image to the repository used on the cloud service (AWS ECR, GCP Artifact Registry)
        2. Choose the service. Depending on the service, the API might need to be always running, then an AWS EC2 instance or a GCP compute engine, or just serve requests spaced out in time, then a serverless architecture would be prefered, like AWS Fargate, lambda or GCP Cloud Run.
        3. Choose the machine type. Depending on the model and service requirements, the selected machine would need more or less memory, vCPUs or even GPUs
        4. Expose the API through a managed API gateway like AWS API Gateway or GCP API Gateway.

- If this model was deployed to categorize products without any supervision which metrics would you check to detect data drifting? When would you need to retrain?

    - I would pay attention mainly to two metrics:

        - Input data distribution: If the input data distribution changes it could potentially change the output labels
        - Output distributions: Same as the input, a change in the distribution of what the model usually predicts/classify could be an indicator that the model needs to be retrained.

    - If any of input/output distribution changes, then I would consider retrain the model.
    - If new features in the data arrives, then the model would also need to be retrained.
