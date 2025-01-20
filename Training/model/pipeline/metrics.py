from config import paths_settings, model_settings

from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import os



def evaluate_logistic_regression(model, x_test, y_test, class_names, model_version):
    """
    Evaluate a logistic regression model on test data and compute various metrics.
    Parameters:
        model (sklearn.linear_model.LogisticRegression): The logistic regression model to evaluate.
        x_test (numpy.ndarray or pandas.DataFrame): The test features.
        y_test (numpy.ndarray or pandas.Series): The true labels for the test data.
        class_names (list of str): The names of the classes.
        model_version (str): The version identifier for the model.
    Returns:
        dict: A dictionary containing the following metrics:
            - 'accuracy': Accuracy of the model.
            - 'classification_report': Detailed classification report.
            - 'confusion_matrix': Confusion matrix.
            - 'roc_auc': ROC AUC score.
    This function performs the following steps:
        1. Makes predictions using the provided model on the test data.
        2. Computes various metrics such as confusion matrix and ROC curve.
        3. Saves the confusion matrix and ROC curve as images.
    """

    # Make predictions
    try:
        logger.info('Making predictions...')
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)

        # Get metrics
        logger.info('Computing metrics...')
        metrics = compute_metrics(
            y_test, 
            y_pred, 
            y_prob, 
            class_names
        )

        # Save Confussion Matrix and AUC curves to images
        logger.info('Saving metrics to images...')
        confussion_matrix_to_img(
            metrics['confusion_matrix'], 
            class_names,
            model_version
        )
        # Convert the confusion matrix to a list for serializing it to json later
        metrics['confusion_matrix'] = metrics['confusion_matrix'].tolist()

        roc_curve_to_img(
            y_test, 
            y_prob, 
            class_names,
            model_version
        )
    except Exception as e:
        logger.error(f'Error computing metrics: {e}')
        raise

    logger.info('Metrics completed.')
    return metrics


def compute_metrics(y_true, y_pred, y_prob, class_names):
    """
    Compute various evaluation metrics for a classification model.
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like): Predicted probabilities.
        class_names (list): List of class names.
    Returns:
        dict: A dictionary containing the following metrics:
            - 'accuracy': Accuracy of the model.
            - 'classification_report': Detailed classification report.
            - 'confusion_matrix': Confusion matrix.
            - 'roc_auc': ROC AUC score.
    """

    #Dictionary to store the metrics
    metrics = {}

    # Compute the accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    # Log the accuracy
    logger.info(f'Accuracy: {metrics["accuracy"]:.2f}')

    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True
    )

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # ROC AUC (calculate only if ovr)
    if model_settings.multi_class == 'ovr':
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')

    # Return metrics
    return metrics
    

def roc_curve_to_img(y_true, y_prob, class_names, model_version):
    """
    Generates and saves a ROC curve plot as an image.
    Parameters:
        y_true (array-like): True binary labels or binary label indicators.
        y_prob (array-like): Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
        class_names (list of str): List of class names corresponding to the columns in y_prob.
        model_version (str): The version identifier for the model.
    Returns:
        None
    """

    # Ensure the directory exists
    output_dir = f'{paths_settings.model_output_dir}/metrics'
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC: {roc_auc_score(y_true == i, y_prob[:, i]):.2f})')

    # Add labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Save the plot to an image
    image_path = os.path.join(output_dir, f'roc_curve_{model_version}.png')
    plt.savefig(image_path)
    plt.close()


def confussion_matrix_to_img(confusion_matrix, class_names, model_version):
    """
    Generates a heatmap from a confusion matrix and saves it as an image.
    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix to be visualized.
        class_names (list of str): The names of the classes to be displayed on the axes.
        model_version (str): The version identifier for the model
    Returns:
        None
    """

    # Ensure the directory exists
    output_dir = f'{paths_settings.model_output_dir}/metrics'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    # Add labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot to an image
    image_path = os.path.join(output_dir, f'confusion_matrix_{model_version}.png')
    plt.savefig(image_path)
    plt.close()