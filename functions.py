'''
Functions - Cleaned version containing only actively used functions
'''

import pandas as pd 
import numpy as np 
import torch

def calc_coverage(predictions, groundtruth, intervals, per_task=True):
    '''
    predictions: list with predictions 
    groundtruth: list with true values 
    intervals: if list has two elements, then it is the upper and lower from GP-like models 
                if has more than two, then it is the same length as the predictions and groundtruth and comes from 
                the conformal algorithm  
    
    '''
    predictions_tensor = torch.Tensor(predictions)
    groundtruth_tensor = torch.Tensor(groundtruth)

    mean_coverage, mean_intervals = 0,0 

    if len(intervals) == 2: 
        # upper and lower 
        lower = intervals[0]
        upper = intervals[1]

        assert len(upper) == len(lower)
        groundtruth_tensor = torch.Tensor(groundtruth)
        upper_tensor = torch.Tensor(upper) 
        lower_tensor = torch.Tensor(lower)
        intervals = torch.abs(upper_tensor-lower_tensor)

        coverage =  torch.logical_and(upper_tensor >= groundtruth_tensor, lower_tensor <= groundtruth_tensor)

    else: 
        coverage = [] 
        upper, lower = [], [] 

        for i in range(len(intervals)):
            upper.append(predictions[i] + intervals[i])
            lower.append(predictions[i] - intervals[i])

        upper_tensor = torch.Tensor(upper) 
        lower_tensor = torch.Tensor(lower)

        intervals = torch.abs(lower_tensor-upper_tensor) 
        coverage =  torch.logical_and(upper_tensor >= groundtruth_tensor, lower_tensor <= groundtruth_tensor)

    mean_coverage = torch.count_nonzero(coverage)/coverage.shape[0]
    mean_intervals = torch.mean(intervals)
    return coverage, intervals, mean_coverage, mean_intervals


def process_temporal_singletask_data(train_x, train_y, test_x, test_y): 
    
    assert train_x.shape[0] == train_y.shape[0]
    train_x_data = [] 
    assert len(train_x) > 0

    for i, t in enumerate(train_x): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]    
        train_x_data.append(np.expand_dims(np.array(b), 0))

    test_x_data = []
    for i, t in enumerate(test_x): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        test_x_data.append(np.expand_dims(np.array(b), 0))

    test_y_data = [] 
    for i, t in enumerate(test_y): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        test_y_data.append(np.expand_dims(np.array(b), 0))
    
    train_y_data = [] 
    for i, t in enumerate(train_y): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        train_y_data.append(np.expand_dims(np.array(b), 0))

    train_x_data = np.concatenate(train_x_data, axis=0)
    test_x_data = np.concatenate(test_x_data, axis=0)
    train_y_data = np.concatenate(train_y_data, axis=0)
    test_y_data = np.concatenate(test_y_data, axis=0)
    
    train_y, test_y = np.array(train_y), np.array(test_y)

    data_train_x = torch.Tensor(train_x_data)
    data_train_y = torch.Tensor(train_y_data)
    data_test_x = torch.Tensor(test_x_data)
    data_test_y = torch.Tensor(test_y_data)  

    return data_train_x, data_train_y, data_test_x, data_test_y


def mae(y, y_hat):
    """
    PARAMETERS
    y: array of ground truth values
    y_hat: array of predicted values

    RETURN
    mae_result: float of mean absolute error 
    abs_diff: absolute error
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    diff = np.subtract(y, y_hat)
    abs_diff = np.fabs(diff)
    mae_result = np.sum(abs_diff, axis=0)/len(y_hat)

    return mae_result, abs_diff


def mse(y, y_hat): 
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    diff = np.subtract(y, y_hat)
    diff_squared = np.power(diff, 2) 

    mse_result = np.sum(diff_squared, axis=0)/len(y_hat)

    rmse_result = np.sqrt(mse_result)

    return mse_result, rmse_result, diff_squared


def R2(y, y_hat): 
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    ybar = np.sum(y, axis=0)/len(y)
    ssreg = np.sum((y_hat-ybar)**2, axis=0)
    sstot = np.sum((y - ybar)**2, axis=0)

    r_sq = ssreg/sstot

    return r_sq


def save_model(model, optimizer, likelihood, filename="model_state.pth"):
    """
    Save the model state dictionary and the optimizer state and the data
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'likelihood_state_dict': likelihood.state_dict()
    }, filename)
