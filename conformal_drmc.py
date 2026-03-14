'''
- Deep Regression Monte Carlo Dropout for Randomly-Timed Biomarker Trajectories
- Conformal Prediction for Randomly-Timed Biomarker Trajectories with Deep Regression Monte Carlo Dropout
'''

import numpy as np
import torch
import torch.nn as nn
import argparse
import time 
import pandas as pd
import pickle
import json
from functions import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepRegressionMC(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        """
        Initializes the deep regression model with Monte Carlo Dropout.
        Args:
            input_dim: Int, number of input features.
            dropout_rate: Float, probability of dropping a neuron.
        """
        super(DeepRegressionMC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single regression value
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass with MC Dropout.
        Args:
            x: Tensor, input features.
        Returns:
            Tensor: Regression predictions.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply MC Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply MC Dropout
        x = self.fc3(x)  # Single output for regression
        return x

def train_deep_regression(
    model, train_x, train_y, epochs, learning_rate, gpuid=-1
):
    """
    Trains the deep regression model with MSE loss.
    Args:
        model: DeepRegressionMC, the regression model.
        train_x: Tensor, input training data.
        train_y: Tensor, target training data.
        epochs: Int, number of training epochs.
        learning_rate: Float, learning rate for the optimizer.
        gpuid: Int, GPU device ID (-1 for CPU).
    """
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(train_x).squeeze()
        loss = criterion(preds, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

def predict_and_update_results_per_subject(
    model, test_x, test_y, test_ids, test_time, num_samples, kfold,
    population_results,z, gpuid=-1
):
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)

    model.train()  # Keep dropout active during testing

    test_ids = np.array(test_ids)
    unique_ids = np.unique(test_ids)

    ae_list = []
    uncertainty_list = []
    all_mean_preds = []  # Collect mean predictions for all subjects

    for subject_id in unique_ids:
        subject_mask = (test_ids == subject_id)
        subject_x = test_x[torch.tensor(subject_mask, device=device)]
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        # Perform MC Dropout for the current subject
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(model(subject_x).cpu().numpy())
        preds = np.array(preds)  # Shape: (num_samples, num_subject_points)

        # Mean and variance for MC Dropout
        mean_preds = preds.mean(axis=0)
        uncertainty = preds.var(axis=0)

        # Collect mean predictions for the subject
        all_mean_preds.append(mean_preds)

        # Confidence intervals for the subject
        lower_bounds = mean_preds - z * np.sqrt(uncertainty)
        upper_bounds = mean_preds + z * np.sqrt(uncertainty)

        # Check coverage
        subject_y_np = subject_y.cpu().numpy()
        in_interval = (subject_y_np >= lower_bounds) & (subject_y_np <= upper_bounds)
        coverage = int(in_interval.all())

        # MAE and interval size
        mae = mean_absolute_error(subject_y_np, mean_preds)
        interval_size = np.mean(upper_bounds - lower_bounds)

        # Winkler score
        winkler_scores = []
        winkler_score = 0 
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = lower_bounds[i]
            u = upper_bounds[i]

            if l <= y_true <= u:
                winkler_score = u - l
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)

            winkler_scores.append(winkler_score[0])

        # Update per-sample results
        for i in range(len(subject_y)):
            population_results['id'].append(subject_id)
            population_results['kfold'].append(kfold)
            population_results['score'].append(mean_preds[i][0])
            population_results['lower'].append(lower_bounds[i][0])
            population_results['upper'].append(upper_bounds[i][0])
            population_results['variance'].append(uncertainty[i][0])
            population_results['y'].append(subject_y[i].item())
            population_results['time'].append(subject_time[i].item())
            population_results['winkler'].append(winkler_scores[i])
            ae = abs(subject_y[i].item() - mean_preds[i])
            population_results['ae'].append(ae[0])
            ae_list.append(ae)
            uncertainty_list.append(uncertainty[i])

    # Concatenate mean predictions for all subjects
    mean_preds_all = np.concatenate(all_mean_preds)
    assert mean_preds_all.shape[0] == test_y.shape[0], "Mismatch between test_y and predictions"

def conformalized_inference(model, test_x, test_y, test_ids,test_time, num_samples, kfold, results, qhat, study, gpuid=-1):

    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)

    model.train()  # Keep dropout active during testing

    test_ids = np.array(test_ids)
    unique_ids = np.unique(test_ids)

    all_mean_preds = []  # Collect mean predictions for all subjects
    # print('Within the inference code Qhat', qhat)
    # print('Unique IDs', len(unique_ids))
    for subject_id in unique_ids:
        subject_mask = (test_ids == subject_id)
        subject_x = test_x[torch.tensor(subject_mask, device=device)]
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        # Perform MC Dropout for the current subject
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(model(subject_x).cpu().numpy())
        preds = np.array(preds)  # Shape: (num_samples, num_subject_points)

        # Mean and variance for MC Dropout
        mean_preds = preds.mean(axis=0)
        uncertainty = preds.var(axis=0)

        # Collect mean predictions for the subject
        all_mean_preds.append(mean_preds)

        # Confidence intervals for the subject
        lower_bounds = mean_preds - qhat * np.sqrt(uncertainty)
        upper_bounds = mean_preds + qhat * np.sqrt(uncertainty)

        # Winkler Scores 
        # Winkler score 
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = lower_bounds[i]
            u = upper_bounds[i]

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score[0]) 

        # Update per-sample results
        for i in range(len(subject_y)):
            results['id'].append(subject_id)
            results['kfold'].append(kfold)
            results['score'].append(mean_preds[i][0])
            results['lower'].append(lower_bounds[i][0])
            results['upper'].append(upper_bounds[i][0])
            results['variance'].append(uncertainty[i][0])
            results['y'].append(subject_y[i].item())
            results['time'].append(subject_time[i].item())
            ae = abs(subject_y[i].item() - mean_preds[i])
            results['ae'].append(ae[0])
            results['study'].append(study)
            results['winkler'].append(winkler_scores[i])

    return results

def conformalized_inference_with_train_sigma(model, test_x, test_y, test_ids,test_time, num_samples, kfold, results, qhat, sigma, study, gpuid=-1):

    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)

    model.train()  # Keep dropout active during testing

    test_ids = np.array(test_ids)
    unique_ids = np.unique(test_ids)

    all_mean_preds = []  # Collect mean predictions for all subjects
    # print('Within the inference code Qhat', qhat)
    print('Unique IDs', len(unique_ids))
    for subject_id in unique_ids:
        subject_mask = (test_ids == subject_id)
        subject_x = test_x[torch.tensor(subject_mask, device=device)]
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        # Perform MC Dropout for the current subject
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(model(subject_x).cpu().numpy())
        preds = np.array(preds)  # Shape: (num_samples, num_subject_points)

        # Mean and variance for MC Dropout
        mean_preds = preds.mean(axis=0)
        uncertainty = preds.var(axis=0)

        # Collect mean predictions for the subject
        all_mean_preds.append(mean_preds)

        # Confidence intervals for the subject
        lower_bounds = mean_preds - qhat * sigma
        upper_bounds = mean_preds + qhat * sigma 

        # Winkler Scores 
        # Winkler score 
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = lower_bounds[i]
            u = upper_bounds[i]

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score[0]) 

        # Update per-sample results
        for i in range(len(subject_y)):
            results['id'].append(subject_id)
            results['kfold'].append(kfold)
            results['score'].append(mean_preds[i][0])
            results['lower'].append(lower_bounds[i][0])
            results['upper'].append(upper_bounds[i][0])
            results['variance'].append(uncertainty[i][0])
            results['y'].append(subject_y[i].item())
            results['time'].append(subject_time[i].item())
            ae = abs(subject_y[i].item() - mean_preds[i])
            results['ae'].append(ae[0])
            results['study'].append(study)
            results['winkler'].append(winkler_scores[i])

    return results

def inference(
    model, test_x, test_y, test_ids, test_time, num_samples, kfold,
    results, z, study,  gpuid=-1):

    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)

    model.train()  # Keep dropout active during testing

    test_ids = np.array(test_ids)
    unique_ids = np.unique(test_ids)

    all_mean_preds = []  # Collect mean predictions for all subjects

    for subject_id in unique_ids:
        subject_mask = (test_ids == subject_id)
        subject_x = test_x[torch.tensor(subject_mask, device=device)]
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        # Perform MC Dropout for the current subject
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(model(subject_x).cpu().numpy())
        preds = np.array(preds)  # Shape: (num_samples, num_subject_points)

        # Mean and variance for MC Dropout
        mean_preds = preds.mean(axis=0)
        uncertainty = preds.var(axis=0)

        # Collect mean predictions for the subject
        all_mean_preds.append(mean_preds)

        # Confidence intervals for the subject
        lower_bounds = mean_preds - z * np.sqrt(uncertainty)
        upper_bounds = mean_preds + z * np.sqrt(uncertainty)

        # Winkler score 
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = lower_bounds[i]
            u = upper_bounds[i]

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score) 

        # Update per-sample results
        for i in range(len(subject_y)):
            results['id'].append(subject_id)
            results['kfold'].append(kfold)
            results['score'].append(mean_preds[i][0])
            results['lower'].append(lower_bounds[i][0])
            results['upper'].append(upper_bounds[i][0])
            results['variance'].append(uncertainty[i][0])
            results['y'].append(subject_y[i].item())
            results['time'].append(subject_time[i].item())
            ae = abs(subject_y[i].item() - mean_preds[i])
            results['ae'].append(ae[0])
            results['winkler'].append(winkler_scores[i])
            results['study'].append(study)

    return results


parser = argparse.ArgumentParser(description='Deep Regression with Monte Carlo Dropout')
## Data Parameters 
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--file", help="Identifier for the data", default="./data/data.csv")
parser.add_argument("--biomarker_idx", type=int, default=14)
## Conformal Prediction Parameters 
parser.add_argument("--alpha", help='Significance Level', type=float, default=0.05)
parser.add_argument("--calibrationset", help='Size of the Calibration Set', type=float, default=0.2)

population_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [] }
conformal_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [], 'study':[] }

qhat_dict = {'qhat': [], 'calibration_set_size': [], 'fold': []}

t0= time.time()
args = parser.parse_args()
gpuid = int(args.gpuid)
file = args.file

biomarker_idx = args.biomarker_idx
alpha = float(args.alpha)
calibrationset = args.calibrationset

if alpha == 0.1: 
    z = 1.645
elif alpha == 0.05: 
    z = 1.96
elif alpha == 0.01: 
    z = 2.576
    
datasamples = pd.read_csv(file)
list_index = biomarker_idx

# print(biomarker_idx)
for fold in range(10): 
    # print('FOLD::', fold)
    train_ids, test_ids = [], []     

    with (open("./data/folds/fold_" + str(fold) +  "_train.pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
      
    with (open("./data/folds/fold_" + str(fold) + "_test.pkl", "rb")) as openfile:
        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break
    
    train_ids = train_ids[0]
    test_ids = test_ids[0]

    # print('Train IDs', len(train_ids))
    # print('Test IDs', len(test_ids))

    for t in test_ids: 
        if t in train_ids: 
            raise ValueError('Test Samples belong to the train!')

    ### SET UP THE TRAIN/TEST DATA FOR THE MULTITASK GP### 
    train_x = datasamples[datasamples['anon_id'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['anon_id'].isin(train_ids)]['Y']    
    test_x = datasamples[datasamples['anon_id'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['anon_id'].isin(test_ids)]['Y']

    corresponding_test_ids = datasamples[datasamples['anon_id'].isin(test_ids)]['anon_id'].to_list()
    corresponding_train_ids = datasamples[datasamples['anon_id'].isin(train_ids)]['anon_id'].to_list() 
    assert len(corresponding_test_ids) == test_x.shape[0]
    assert len(corresponding_train_ids) == train_x.shape[0]

    train_x, train_y, test_x, test_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpuid) 
        train_y = train_y.cuda(gpuid)
        test_x = test_x.cuda(gpuid) 
        test_y = test_y.cuda(gpuid)

    ### Check if there is negative time
    time_ = train_x[:, -1].cpu().detach().numpy().tolist()

    # print('Train Time', len(time_))
    # print('Train Subjectes', len(corresponding_train_ids))

    test_y = test_y[:, list_index]
    train_y = train_y[:, list_index]

    train_y = train_y.squeeze() 
    test_y = test_y.squeeze()

    #### Define the Quantile Regressor ####
    # Constants
    input_dim = train_x.shape[1]  # Number of input features
    epochs = 700
    learning_rate = 0.02
    gpuid = 0  # Use -1 for CPU

    # Initialize model
    model = DeepRegressionMC(input_dim, dropout_rate=0.2)

    # Train the model
    print("Starting training...")
    train_deep_regression(
        model=model,
        train_x=train_x,
        train_y=train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        gpuid=gpuid
    )
    # print("Training completed.")


    test_time = np.array(test_x[:, -1].cpu().detach().numpy())  # Convert directly to NumPy array
    test_ids = corresponding_test_ids
    # convert test_ids to str
    test_ids = [str(i) for i in test_ids]

    # Perform inference and update results
    predict_and_update_results_per_subject(
        model=model,
        test_x=test_x,
        test_y=test_y,
        test_ids=test_ids,
        test_time=test_time,
        num_samples=100,
        kfold=fold,
        population_results=population_results,
        z=z, 
        gpuid=gpuid
    )

    # print('Split the data into train and calibration set')
    ### Split the train data into train/calibration set ###
    # convert to float
    conformal_split_percentage = float(calibrationset)

    # print('Random Selection of Calibration Set')
    calibration_ids = np.random.choice(train_ids, int(conformal_split_percentage*len(train_ids)), replace=False)
    train_ids = [x for x in train_ids if x not in calibration_ids]

    # print('Train IDs', len(train_ids))
    # print('Calibration IDs', len(calibration_ids))

    for t in calibration_ids:
        if t in train_ids: 
            raise ValueError('Calibration Samples belong to the train!')
        
    ### SET UP THE TRAIN/TEST DATA FOR THE MULTITASK GP###
    train_x = datasamples[datasamples['anon_id'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['anon_id'].isin(train_ids)]['Y']
    calibration_x = datasamples[datasamples['anon_id'].isin(calibration_ids)]['X']
    calibration_y = datasamples[datasamples['anon_id'].isin(calibration_ids)]['Y']

    corresponding_train_ids = datasamples[datasamples['anon_id'].isin(train_ids)]['anon_id'].to_list()
    corresponding_calibration_ids = datasamples[datasamples['anon_id'].isin(calibration_ids)]['anon_id'].to_list()

    assert len(corresponding_train_ids) == train_x.shape[0]
    assert len(corresponding_calibration_ids) == calibration_x.shape[0]

    train_x, train_y, calibration_x, calibration_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=calibration_x, test_y=calibration_y)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpuid) 
        train_y = train_y.cuda(gpuid)
        calibration_x = calibration_x.cuda(gpuid)
        calibration_y = calibration_y.cuda(gpuid)


    calibration_y = calibration_y[:, list_index]
    train_y = train_y[:, list_index]

    train_y = train_y.squeeze()
    calibration_y = calibration_y.squeeze()

    # print('Train the Deep Regressor')
    # Initialize model
    conformal_model = DeepRegressionMC(input_dim, dropout_rate=0.2)

    # Train the model
    # print("Starting training...")

    train_deep_regression(
        model=conformal_model,
        train_x=train_x,
        train_y=train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        gpuid=gpuid
    )
    # print("Training completed.")

    # Calculat the Max Train Residual 
    train_time = np.array(train_x[:, -1].cpu().detach().numpy())  # Convert directly to NumPy array
    train_ids = corresponding_train_ids
    # convert test_ids to str
    train_ids = [str(i) for i in train_ids]

    # Perform inference on calibration dataset 
    calibration_time = np.array(calibration_x[:, -1].cpu().detach().numpy())  # Convert directly to NumPy array
    calibration_ids = corresponding_calibration_ids
    # convert test_ids to str
    calibration_ids = [str(i) for i in calibration_ids]
    calibration_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [], 'study':[] }

    # Perform inference and update results
    calibration_results = inference(
        model=conformal_model,
        test_x=calibration_x,
        test_y=calibration_y,
        kfold=fold,
        test_ids=calibration_ids,
        test_time=calibration_time,
        results=calibration_results,
        num_samples=100,
        gpuid=gpuid, 
        z=z, 
        study='sample_study_calibration'
    )

    calibration_results_df = pd.DataFrame(calibration_results)
    # print('Calibration Results')

    calibration_results_df = pd.DataFrame(data=calibration_results)
    
    # print('Calculate the Non-Conformity Scores')
    conformity_scores_per_subject = {'id': [], 'conformal_scores': []}
    for subject in calibration_results_df['id'].unique():
        subject_df = calibration_results_df[calibration_results_df['id'] == subject]

        std = np.sqrt(subject_df['variance'])

        conformal_scores = np.abs(subject_df['score'] - subject_df['y'])/std #
        nonnorm = np.abs(subject_df['score'] - subject_df['y'])
        
        conformity_scores_per_subject['id'].append(subject)
        conformity_scores_per_subject['conformal_scores'].append(np.max(conformal_scores))

    conformity_scores_per_subject_df = pd.DataFrame(data=conformity_scores_per_subject)

    # gather all the conformity scores
    conformity_scores = np.array(conformity_scores_per_subject['conformal_scores'])
    sorted_conformity_scores = np.sort(conformity_scores)
    # n is the number of the validation subjects 
    n = conformity_scores.shape[0]
    conformal_alpha = alpha
    # alpha is the user-chosen error rate. In w ords, the probability that the prediction set contains the correct label is almost exactly 1 􀀀 ; we call
    # this property marginal coverage, since the probability is marginal (averaged) over the randomness in the
    # calibration and test points

    k = int(np.ceil((n + 1) * (1 - conformal_alpha)))
    # Ensure k does not exceed n
    k = min(k, n)
    # Get the (n - k + 1)-th smallest value since we want the k-th largest value
    qhat = sorted_conformity_scores[k-1]
    # print('Qhat', qhat)
    # print('Calibration Set Size',len(calibration_ids))

    qhat_dict['qhat'].append(qhat)
    qhat_dict['calibration_set_size'].append(len(calibration_results_df['id'].unique()))
    qhat_dict['fold'].append(fold)

    # print('Test the Conformalized Deep Regressor with Monte Carlo Dropout')
    test_time = np.array(test_x[:, -1].cpu().detach().numpy())  # Convert directly to NumPy array
    test_ids = corresponding_test_ids
    # convert test_ids to str
    test_ids = [str(i) for i in test_ids]

    # Perform inference and update results
    conformal_results = conformalized_inference(
        model=conformal_model,
        test_x=test_x,
        test_y=test_y,
        test_ids=test_ids,
        test_time=test_time,
        num_samples=100,
        kfold=fold,
        results=conformal_results,
        qhat=qhat,
        gpuid=gpuid, 
        study='sample_study'
    )


# Convert dictionaries to DataFrames
population_results_df = pd.DataFrame(population_results)
conformal_results_df = pd.DataFrame(conformal_results)

# Save results to CSV
qhat_df = pd.DataFrame(data=qhat_dict)
qhat_df.to_csv("./results/qhat_drmc_"+str(biomarker_idx)+"_results_calibrationset_" + str(calibrationset) + "_alpha_" + str(alpha) + ".csv", index=False)
population_results_df.to_csv("./results/drmc_"+str(biomarker_idx)+"_results_calibrationset_" + str(calibrationset) + "_alpha_" + str(alpha) + ".csv", index=False)
conformal_results_df.to_csv("./results/conformalized_drmc_"+str(biomarker_idx)+"_results_calibrationset_" + str(calibrationset) + "_alpha_" + str(alpha) + ".csv", index=False)
