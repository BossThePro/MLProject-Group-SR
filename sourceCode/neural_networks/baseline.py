import numpy as np
import pandas as pd
# from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import tqdm
import copy
import os
import time
# from torchmetrics.functional import r2_score
from sklearn.metrics import r2_score
import sklearn.metrics


df_training = pd.read_csv("data/claims_train_scaled.csv")

# max_size = df_training['ClaimNb'].value_counts().max()
# lst = [df_training]
# for class_index, group in df_training.groupby('ClaimNb'):
#     lst.append(group.sample(max_size-len(group), replace=True))
# df_training = pd.concat(lst)


# > Exclude Categorical Variables
# X = df_training.loc[:, ~df_training.columns.isin(['ClaimNb','Region','VehGas','VehBrand','Area'])] # excludes categorical variables

# > NOTE: technically turns data exposure_train_df and exposure_val_df into sample weights (which exposure is), but too lazy to change var name
# no_claims_exposure = np.sum(df_training[df_training['ClaimNb']==0].loc[:,'Exposure'], axis=0)
# positive_claims_exposure = np.sum(df_training[df_training['ClaimNb']>0].loc[:,'Exposure'], axis=0)
# ratio_exposure = no_claims_exposure/positive_claims_exposure

# df_no_claims = df_training[df_training['ClaimNb']==0].copy()
# df_positive_claims = df_training[df_training['ClaimNb']>0].copy()


# df_positive_claims.loc[:,'Exposure'] = df_positive_claims.loc[:,'Exposure'] * ratio_exposure
# df_training[df_training['ClaimNb']>0] = df_positive_claims.copy()
# > NOTE END

# > Include Categorical Variables
X = df_training.loc[:, df_training.columns != 'ClaimNb']
X = X.loc[:, X.columns != 'Area']
X = pd.get_dummies(X, columns=['Region','VehGas','VehBrand'], drop_first=True, dtype=float)


Y = df_training.loc[:,"ClaimNb"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# > Seperate Exposure 
exposure_train_df = X_train.loc[:, X_train.columns == 'Exposure'] 
exposure_val_df = X_val.loc[:, X_val.columns == 'Exposure']

X_train = X_train.loc[:, X_train.columns != 'Exposure']
X_val = X_val.loc[:, X_val.columns != 'Exposure']

X_train = np.array(X_train,dtype="float32")
X_val = np.array(X_val,dtype="float32")
y_train = np.array(y_train,dtype="float32")
y_val = np.array(y_val,dtype="float32")

temp_mean = np.mean(y_train)


# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

y_predicted = np.ones((y_val.shape[0],)) * temp_mean
# y_predicted = torch.tensor(y_predicted, dtype=torch.float32)

# y_predicted = torch.exp(y_predicted)
# y_predicted = soft_plus(y_predicted)
print("Variance: ", torch.var(torch.tensor(y_predicted, dtype=torch.float32)))
print("Max: ", torch.max(torch.tensor(y_predicted, dtype=torch.float32)))
print("Min: ", torch.min(torch.tensor(y_predicted, dtype=torch.float32)))
print("Mean: ", torch.mean(torch.tensor(y_predicted, dtype=torch.float32)))
print("R^2: ", float(r2_score(torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1), torch.tensor(y_predicted, dtype=torch.float32))))
print("D^2: ", float(sklearn.metrics.d2_tweedie_score(y_true=y_val, y_pred=y_predicted,power=1)))




# loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='mean', full=True)  # poisson Negative Likelihood Loss

loss_val = sklearn.metrics.mean_poisson_deviance(y_val,y_predicted)

# loss_val = loss_fn(y_predicted, y_val)
# loss_val = float(loss_val)

print("Validation Loss: ", loss_val)