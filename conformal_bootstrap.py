'''
Conformal Prediction for Randomly-Timed Biomarker Trajectories with Bootstrap predictor
'''
import numpy as np 
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import argparse
import json
import time 
import pickle
from functions import * 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import random 

##### SET RANDOM SEED ####
# Set random seeds
seed = 42  
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
##################################

def generate_bootstrap_datasets(data, labels, num_bootstraps):
    n = len(data)
    bootstrap_datasets = []
    for _ in range(num_bootstraps):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        bootstrap_datasets.append((data[indices], labels[indices]))
    return bootstrap_datasets

class DeepRegression(nn.Module):
    def __init__(self, input_dim):
        super(DeepRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single regression output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_bootstrap_ensemble(train_x, train_y, num_models, input_dim, epochs, learning_rate, gpuid):
    bootstrap_datasets = generate_bootstrap_datasets(train_x.cpu().numpy(), train_y.cpu().numpy(), num_models)
    models = []
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")

    for i, (boot_x, boot_y) in enumerate(bootstrap_datasets):
        model = DeepRegression(input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        boot_x = torch.tensor(boot_x, dtype=torch.float32).to(device)
        boot_y = torch.tensor(boot_y, dtype=torch.float32).to(device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(boot_x).squeeze()
            loss = criterion(preds, boot_y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Model {i+1}/{num_models}, Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        models.append(model)

    return models

def predict_and_update_results_per_subject(
    models,  # List of models from bootstrap training
    test_x, test_y, test_ids, test_time, kfold,
    population_results, alpha, z, gpuid=-1
):
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)

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

        # Perform predictions with all models in the ensemble
        preds = []
        with torch.no_grad():
            for model in models:
                model.eval()
                preds.append(model(subject_x).cpu().numpy())
        preds = np.array(preds)  # Shape: (num_models, num_subject_points)

        # Mean and variance for bootstrap ensemble
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

        # Calculate the Winkler score
        winkler_scores = []
        winkler_score = 0
        for i in range(len(subject_y)):

            y_true = subject_y[i].item()
            l = lower_bounds[i][0]
            u = upper_bounds[i][0]

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score)

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

def bootstrap_predictions(models, test_x, gpuid):
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    test_x = test_x.to(device)

    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(test_x).cpu().numpy()
            predictions.append(preds)
    predictions = np.array(predictions)  # Shape: (num_models, num_samples)
    return predictions

def compute_uncertainty(predictions):
    mean_preds = predictions.mean(axis=0)
    epistemic_uncertainty = predictions.var(axis=0)
    return mean_preds, epistemic_uncertainty

parser = argparse.ArgumentParser(description='Bootstrap Deep Regression')
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--file", help="Identifier for the data", default="./data/data.csv")
parser.add_argument("--biomarker_idx", type=int, default=14)
## Conformal Prediction Parameters 
parser.add_argument("--alpha", help='Significance Level', type=float, default=0.05)
parser.add_argument("--calibrationset", help='Size of the Calibration Set', type=float, default=0.2)

population_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [] }
conformal_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [], 'study':[] }

qhat_dict = {'qhat': [], 'calibration_set_size': [], 'fold': [] }

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

for fold in range(10): 
    # print('FOLD::', fold)
    train_ids, test_ids = [], []     

    with (open("./data/folds/fold_" + str(fold) +  "_train.pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
      
    with (open("./data/folds/fold_" + str(fold) +  "_test.pkl", "rb")) as openfile:
        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break
    
    train_ids = train_ids[0]
    test_ids = test_ids[0]

    # print('Train the Deep Kernel Regressor')
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

    #### Define the Bootstrap Regressor ####
    input_dim = train_x.shape[1]
    num_models = 10  # Number of bootstrap models
    epochs = 50
    learning_rate = 0.01

    models = train_bootstrap_ensemble(
    train_x=train_x,
    train_y=train_y,
    num_models=num_models,
    input_dim=input_dim,
    epochs=epochs,
    learning_rate=learning_rate,
    gpuid=gpuid
    )
    
    # print("Bootstrap Training completed.")
    # Prepare test data
    test_time = np.array(test_x[:, -1].cpu().detach().numpy())  # Extract test time
    z = 1.96  # For 95% confidence interval

    # Run predictions and update results
    # print("Running predictions and updating results...")
    predict_and_update_results_per_subject(
        models=models,
        test_x=test_x,
        test_y=test_y,
        test_ids=corresponding_test_ids,  # List of test subject IDs
        test_time=test_time,
        kfold=0,  # Assuming this is fold 0
        population_results=population_results,
        z=z,
        alpha=alpha,
        gpuid=gpuid
    )
    # print("Results updated.")
    
    # print('Split the data into train and calibration set')
    # Split the data into train and calibration set
    conformal_split_percentage = float(calibrationset)  # Fraction for calibration set

    # Randomly select calibration IDs from the training IDs
    # print('Random Selection of Calibration Set')
    calibration_ids = np.random.choice(train_ids, int(conformal_split_percentage * len(train_ids)), replace=False)
    train_ids = [x for x in train_ids if x not in calibration_ids]

    # print('Train IDs:', len(train_ids))
    # print('Calibration IDs:', len(calibration_ids))

    # Ensure there is no overlap between training and calibration IDs
    for t in calibration_ids:
        if t in train_ids:
            raise ValueError('Calibration samples belong to the train!')

    # Extract training and calibration data
    train_x = datasamples[datasamples['anon_id'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['anon_id'].isin(train_ids)]['Y']
    calibration_x = datasamples[datasamples['anon_id'].isin(calibration_ids)]['X']
    calibration_y = datasamples[datasamples['anon_id'].isin(calibration_ids)]['Y']

    # Convert IDs to lists for processing
    corresponding_train_ids = datasamples[datasamples['anon_id'].isin(train_ids)]['anon_id'].to_list()
    corresponding_calibration_ids = datasamples[datasamples['anon_id'].isin(calibration_ids)]['anon_id'].to_list()

    # Check consistency of data shapes
    assert len(corresponding_train_ids) == train_x.shape[0], "Mismatch in train IDs and train data"
    assert len(corresponding_calibration_ids) == calibration_x.shape[0], "Mismatch in calibration IDs and calibration data"

    # Process temporal data for the task
    train_x, train_y, calibration_x, calibration_y = process_temporal_singletask_data(
        train_x=train_x, train_y=train_y, test_x=calibration_x, test_y=calibration_y
    )

    # Move data to GPU if available
    if torch.cuda.is_available():
        train_x, train_y = train_x.cuda(gpuid), train_y.cuda(gpuid)
        calibration_x, calibration_y = calibration_x.cuda(gpuid), calibration_y.cuda(gpuid)

    # Select specific ROI for training
    calibration_y = calibration_y[:, list_index]
    train_y = train_y[:, list_index]

    # Squeeze dimensions for regression
    train_y = train_y.squeeze()
    calibration_y = calibration_y.squeeze()

    # print('Train the Conformalized Deep Regression Bootstrap Models')
    # Train a bootstrap ensemble
    input_dim = train_x.shape[1]  # Number of input features
    num_models = 10  # Number of bootstrap models
    epochs = 50
    learning_rate = 0.01

    # print("Training bootstrap ensemble...")
    bootstrap_conf_models = train_bootstrap_ensemble(
        train_x=train_x,
        train_y=train_y,
        num_models=num_models,
        input_dim=input_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        gpuid=gpuid
    )
    # print("Bootstrap training completed.")


    # Perform inference on calibration dataset
    calibration_time = np.array(calibration_x[:, -1].cpu().detach().numpy())  # Extract time
    calibration_results = {
        'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [],
        'variance': [], 'y': [], 'time': [], 'ae': []} 

    # print("Performing inference on calibration set...")
    # Use the ensemble of bootstrap models to make predictions
    predictions = bootstrap_predictions(models=bootstrap_conf_models, test_x=calibration_x, gpuid=gpuid)
    mean_preds, epistemic_uncertainty = compute_uncertainty(predictions)

    # Compute confidence intervals
    lower_bounds = mean_preds - z * np.sqrt(epistemic_uncertainty)
    upper_bounds = mean_preds + z * np.sqrt(epistemic_uncertainty)

    # Update calibration results
    for i, subject_id in enumerate(corresponding_calibration_ids):
        calibration_results['id'].append(subject_id)
        calibration_results['kfold'].append(fold)
        calibration_results['score'].append(mean_preds[i][0])
        calibration_results['lower'].append(lower_bounds[i][0])
        calibration_results['upper'].append(upper_bounds[i][0])
        calibration_results['variance'].append(epistemic_uncertainty[i][0])
        calibration_results['y'].append(calibration_y[i].item())
        calibration_results['time'].append(calibration_time[i])
        calibration_results['ae'].append(abs(calibration_y[i].item() - mean_preds[i][0]))

    for key, value in calibration_results.items():
        # print(key, len(value))

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

    # print('Test the Conformalized Bootstrap Deep Regressor')
    # Prepare test data
    test_time = np.array(test_x[:, -1].cpu().detach().numpy())  # Extract time
    test_ids = corresponding_test_ids
    test_ids = [str(i) for i in test_ids]  # Convert IDs to string for consistency

    # Perform inference using bootstrap ensemble and conformal intervals
    # print("Performing conformalized inference on test set...")
    predictions = bootstrap_predictions(models=bootstrap_conf_models, test_x=test_x, gpuid=gpuid)
    mean_preds, epistemic_uncertainty = compute_uncertainty(predictions)

    # Compute conformal prediction intervals using qhat
    lower_bounds = mean_preds - qhat * np.sqrt(epistemic_uncertainty)
    upper_bounds = mean_preds + qhat * np.sqrt(epistemic_uncertainty)

    # Update results for each test subject
    for i, subject_id in enumerate(test_ids):
        conformal_results['id'].append(subject_id)
        conformal_results['kfold'].append(fold)
        conformal_results['score'].append(mean_preds[i][0])
        conformal_results['lower'].append(lower_bounds[i][0])
        conformal_results['upper'].append(upper_bounds[i][0])
        conformal_results['variance'].append(epistemic_uncertainty[i][0])
        conformal_results['y'].append(test_y[i].item())
        conformal_results['time'].append(test_time[i])
        ae = abs(test_y[i].item() - mean_preds[i][0])
        conformal_results['ae'].append(ae)

        # Compute Winkler scores for conformal intervals
        y_true = test_y[i].item()
        l = lower_bounds[i][0]
        u = upper_bounds[i][0]
        if l <= y_true <= u:
            winkler_score = u - l  # Width of interval
        elif y_true < l:
            winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
        else:  # y_true > u
            winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction
        conformal_results['winkler'].append(winkler_score)

    conformal_results['study'].extend(['adniblsa'] * len(test_ids))  # Add study name


# Convert dictionaries to DataFrames
population_results_df = pd.DataFrame(population_results)
conformal_results_df = pd.DataFrame(conformal_results)

# Save results to CSV
qhat_df = pd.DataFrame(data=qhat_dict)
qhat_df.to_csv("./results/qhat_bootstrap_"+str(biomarker_idx)+"_results_calibrationset_" + str(calibrationset) + "_alpha_" + str(alpha) + ".csv", index=False)
population_results_df.to_csv("./results/deep_regression_bootstrap_"+str(biomarker_idx)+"_results_calibrationset_" + str(calibrationset) + "_alpha_" + str(alpha) + ".csv", index=False)
conformal_results_df.to_csv("./results/conformalized_deep_regression_bootstrap_"+str(biomarker_idx)+"_results_calibrationset_" + str(calibrationset) + "_alpha_" + str(alpha) + ".csv", index=False)

def calculate_subject_level_metrics(df):
    df['interval_width'] = df['upper'] - df['lower']
    grouped = df.groupby(['id', 'kfold']).apply(lambda group: pd.Series({
        'id': group['id'].iloc[0],
        'kfold': group['kfold'].iloc[0],
        'coverage': int((group['lower'] <= group['y']).all() & (group['y'] <= group['upper']).all()),
        'interval_width': group['interval_width'].mean()
    })).reset_index(drop=True)
    return grouped

# Calculate mean and confidence intervals for each fold
def calculate_fold_metrics(results, method_name):
    grouped = results.groupby('kfold').agg(
        mean_coverage=('coverage', 'mean'),
        mean_interval_width=('interval_width', 'mean'),
        coverage_ci=('coverage', lambda x: 1.96 * np.std(x) / np.sqrt(len(x))),
        width_ci=('interval_width', lambda x: 1.96 * np.std(x) / np.sqrt(len(x))),
    ).reset_index()
    grouped['method'] = method_name
    return grouped

dqr_results = calculate_subject_level_metrics(population_results_df)
dqr_conformalized_results = calculate_subject_level_metrics(conformal_results_df)
