import os
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import ParameterGrid, KFold
from thermo_net.torch_neural_new import ANN_thermo1,ANN_thermo2
from joblib import Parallel, delayed
import concurrent.futures
import torch
# setting common seed
seed_value = 1337
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
# loading the data
data = pd.read_csv('../data/total_nist.csv')
data = data.dropna()
print(len(data))
text_file = open("../data/feature_rank1_energy_nist.txt", "r")
feature_sorted = text_file.read()
feat_columns = feature_sorted.split("\n")[0:-1]
#print(feat_columns)
free_eng = data['F_kJ/mol']
X = data[feat_columns]
energy1 = data['e_kJ/mol'].values
entropy = data['S _J/K/mol'].values
energy=np.array(energy1)*6.8
Tempt=1200
def main():
    start_time = time.time()
    param_grid = {
        'input_dim': [20,25,30],# Adjust the values based on your preference
        'wht':[0.9],
        's':[4],
        'lr': [0.001,0.1],
        'activation': ['leaky_relu'],
        'epochs': [300],
        'batch_size': [32],
        'loss': ['mse'],
        'w1': [2,3,4],
        'w2': [2,3,4],
        'w3': [1,10,50,100],
        'Temperature': [1200]  # not a hyperparameter but need this to instantiate
     }

    #43914         1200  leaky_relu          32     300         25  mse  0.1   4   4   3   4  50  0.9        0.961766   0.981047   0.974402  515.846667
        # K-fold cross-validation parameters
    n_splits = 4  # number of folds
    kf = KFold(n_splits=n_splits)
    results = Parallel(n_jobs=-1)(delayed(run_fold)(params, fold, train_index, test_index)
                                   for params in ParameterGrid(param_grid)
                                   for fold, (train_index, test_index) in enumerate(kf.split(X)))
    average_scores = {}
    for params in ParameterGrid(param_grid):
        param_scores = [result[0] for result in results if result[1] == params]
        average_score = np.mean(param_scores, axis=0)
        average_scores[str(params)] = average_score

    # Splitting string representation to extract keys and values
    hyperparameters = [eval(param_str) for param_str in average_scores.keys()]
    keys = list(hyperparameters[0].keys())

    # Creating a dictionary to hold the data
    data = {key: [] for key in keys}
    data['Free_eng_score'] = []
    data['eng_score'] = []
    data['ent_score'] = []
    data['mae_feng']=[]

    # Populating data dictionary with values
    for params in hyperparameters:
        for key in keys:
            data[key].append(params[key])
        data['Free_eng_score'].append(average_scores[str(params)][0])
        data['eng_score'].append(average_scores[str(params)][1])
        data['ent_score'].append(average_scores[str(params)][2])
        data['mae_feng'].append(average_scores[str(params)][3])

    # Creating DataFrame
    avg_r2_df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    avg_r2_df.to_csv('tln_scores.csv', index=False)
    print("total_time_taken", time.time() - start_time)
def run_fold(params, fold, train_index, test_index):
    return train_evaluate(params, fold, train_index, test_index)
def train_evaluate(params, fold, train_index, test_index):
  #  print("Hello")
    seed_value=1337
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    model = ANN_thermo1(**params)
    m = params['input_dim']
    X_train1 = X.iloc[train_index][feat_columns[0:m]].values
    X_test1 = X.iloc[test_index][feat_columns[0:m]].values
    entropy_train1, entropy_test1 = entropy[train_index], entropy[test_index]
    energy_train1, energy_test1 = energy[train_index], energy[test_index]
    free_eng_train1, free_eng_test1 = free_eng[train_index], free_eng[test_index]
    entropy_pred, energy_pred, _ = model.train_predict(X_train1, entropy_train1, energy_train1, X_test1, free_eng_train1)
    free_eng_loss = [k[0]/6.8 - (Tempt/1000)* k[1] for k in zip(energy_pred, entropy_pred)]
    score = r2_score(free_eng_test1, free_eng_loss)
    score2=r2_score(energy_test1,energy_pred)
    score3=r2_score(entropy_test1,entropy_pred)
    maee=mean_squared_error(free_eng_test1, free_eng_loss)
    print(params,"Free Energy" ,score)
    print("Energy R2:",score2)
    print("ENTROPY R2:",score3)
    return [score, score2, score3,maee], params
if __name__ == '__main__':
    main()

