import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification


    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    TP = 0
    FP = 0
    FN = 0
    
    
    
    TP = np.sum(np.logical_and(prediction == 1, ground_truth == 1))
    
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(prediction == 1, ground_truth == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(prediction == 0, ground_truth == 1))

    
    accuracy = TP / prediction.size
    recall = TP /(TP +FN)
    precision = TP /(TP +FP)
    f1 = 2*(precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return np.sum(prediction == ground_truth)/prediction.size
