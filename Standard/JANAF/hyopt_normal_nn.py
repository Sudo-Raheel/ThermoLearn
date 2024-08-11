import os
import time
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import ParameterGrid, KFold
from thermo_net.torch_neural_new import ANN_torch
from joblib import Parallel, delayed

# setting common seed
seed_value = 1337
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# loading the data
data = pd.read_csv('../data/total_nist.csv')
data = data.dropna()
text_file = open("../data/feature_rank1_energy_nist.txt", "r")
feature_sorted = text_file.read()
feat_columns = feature_sorted.split("\n")[0:-1]
free_eng = data['F_kJ/mol']
X = data[feat_columns]
energy = data['e_kJ/mol'].values
entropy = data['S _J/K/mol'].values
def main():
    start_time = time.time()

    param_grid = {
        'input_dim': [20,25,30,35,40],# Adjust the values based on your preference
        'lr': [0.005,0.01,0.05,0.1],
        'activation': ['leaky_relu','selu'],
        'epochs': [200,250,300],
        'batch_size': [32,64],
        'loss': ['mse'],
     }

#0  leaky_relu          32     150         40  mse  0.05  1.1   4.0  100          0.988368

    # K-fold cross-validation parameters
    n_splits = 4
    kf = KFold(n_splits=n_splits)

    # Use joblib to parallelize the outer loop
    results = Parallel(n_jobs=-1)(delayed(run_parameter_set)(params, kf) for params in ParameterGrid(param_grid))
    average_scores = {}
    for params in ParameterGrid(param_grid):
        param_scores = [result[0] for result in results if result[1] == params]
        average_score = np.mean(param_scores, axis=0).T
        average_scores[str(params)] = average_score

    # Splitting string representation to extract keys and values
    hyperparameters = [eval(param_str) for param_str in average_scores.keys()]
    keys = list(hyperparameters[0].keys())

    # Creating a dictionary to hold the data
    data = {key: [] for key in keys}
    data['Free_eng_score'] = []
    
    for params in hyperparameters:
        for key in keys:
            data[key].append(params[key])

        # Access the only value directly, no need for indexing
        data['Free_eng_score'].append(average_scores[str(params)][0])
    avg_r2_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    avg_r2_df.to_csv('normal_average.csv', index=False)

    print("total_time_taken", time.time() - start_time)

def run_fold(params, fold, train_index, test_index):
    return train_evaluate(params, fold, train_index, test_index)

def run_parameter_set(params, kf):
    seed_value=1337
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    param_scores = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        result = run_fold(params, fold, train_index, test_index)
        param_scores.extend(result[0])
    average_score = np.mean(param_scores)
    return ([average_score], params)
def train_evaluate(params, fold, train_index, test_index):
    seed_value=1337
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    model = ANN_torch(**params)
    m = params['input_dim']
    X_train1 = X.iloc[train_index][feat_columns[0:m]].values
    X_test1 = X.iloc[test_index][feat_columns[0:m]].values
    free_eng_train1, free_eng_test1 = free_eng[train_index], free_eng[test_index]
    feng_pred = model.train_predict(X_train1,free_eng_train1,X_test1)
    score = r2_score(free_eng_test1, feng_pred)
    score_mae=mean_squared_error(free_eng_test1, feng_pred)
    return [score], params

if __name__ == '__main__':
    main()
