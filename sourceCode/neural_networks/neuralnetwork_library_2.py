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
from scipy.special import factorial as factorial

soft_plus = nn.Softplus()

last_epoch_time = time.time()
average_epoch_time = 0

def progress_ui(epoch_num, epoch_max, loss_val, loss_train, r_sq):
    global last_epoch_time
    global average_epoch_time
    global best_loss

    full_count = int((epoch_num+1)/epoch_max*10)
    if epoch_num == 0 or full_count == 10:
        half_count = 0
    else:
        half_count = round(((epoch_num+1)%(epoch_max/10))/(epoch_max/10))
    empty_count = 10 - (full_count+half_count)
    print(f"")

    average_epoch_time *= (epoch_num)/(epoch_num+1)
    average_epoch_time += (time.time()-last_epoch_time)/(epoch_num+1)
    total_estimated_time_left = (epoch_max-(epoch_num+1))*average_epoch_time

    size = os.get_terminal_size()
    for i in range(size[1]-5):
        print('\n')
    print(f"Best Val. Loss: {best_loss} / R-Squared: {r_sq}")
    print(f"Val. Loss: {loss_val} / Trn. Loss: {loss_train} / Average time per epoch: {round(average_epoch_time,2)}s / Est. Remaining Time: {round(total_estimated_time_left,2)}s")
    print(f"\033[91mEpoch:\033[0m {epoch_num+1} / {epoch_max} | \033[92m[{bytes((219,)).decode('cp437')*full_count}{"░"*half_count}{" "*empty_count}] ({round((epoch_num+1)/epoch_max*100, 2)}%)\033[0m")
    last_epoch_time = time.time()

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

exposure_train = np.array(exposure_train_df, dtype="float32")
exposure_val = np.array(exposure_val_df, dtype="float32")

X_train = X_train.loc[:, X_train.columns != 'Exposure']
X_val = X_val.loc[:, X_val.columns != 'Exposure']
# /> End Seperate Exposure

X_train = np.array(X_train,dtype="float32")
X_val = np.array(X_val,dtype="float32")
y_train = np.array(y_train,dtype="float32")
y_val = np.array(y_val,dtype="float32")

# >> Make y frequency instead
# y_train = y_train / exposure_train.reshape(-1,)
# y_val = y_val / exposure_val.reshape(-1,)
# />> End make y frequency instead

a = nn.Linear(37, 192)
torch.nn.init.xavier_uniform_(a.weight)
b = nn.Linear(192, 192)
torch.nn.init.xavier_uniform_(b.weight)
c = nn.Linear(192, 192)
torch.nn.init.xavier_uniform_(c.weight)
d = nn.Linear(192, 1)
torch.nn.init.xavier_uniform_(d.weight)

model = nn.Sequential(
    a,
    nn.ELU(),
    b,
    nn.ELU(),
    c,
    nn.ELU(),
    d
    # Expo()
)

# loss_fn = nn.MSELoss()  # mean square error
# loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='mean', full=True)  # poisson Negative Likelihood Loss
from torch.autograd import Function

class WeightedPoissonNLLLoss(Function):
    @staticmethod
    def forward(ctx, y_pred, y, exposure):    
        ctx.save_for_backward(y_pred, y, exposure)
        # print(y_pred)
        # print(y)
        # print(torch.log(y_pred))
        # print(factorial(y))
        # print(torch.mean(y_pred - y * torch.log(y_pred) + factorial(y)))
        # exit()
        # if torch.mean(y_pred * exposure - y * torch.log(y_pred * exposure) + factorial(y)) > 300:
        #     print(y_pred)
        #     print(y)
        #     print(exposure)
        #     exit()
        # print(torch.mean(y_pred))
        # print(torch.mean(exposure))
        # print(torch.mean(y))
        # print(torch.mean(y_pred * exposure - y * torch.log(y_pred * exposure)))
        return torch.mean(y_pred * exposure - y * torch.log(y_pred * exposure))
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y, exposure = ctx.saved_tensors
        # temp = torch.clamp(y_pred, min=1e-12)
        grad_input = exposure - y/y_pred
        # print(grad_input)
        # exit()
        # print(grad_input)
        return grad_input, None, None

loss_fn = WeightedPoissonNLLLoss.apply

optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)


# training parameters
n_epochs = 30  # number of epochs to run
batch_size = 1000  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
 
# Hold the best model
best_loss = np.inf   # init to infinity
best_weights = None
history_test = []
history_train = []

# training loop

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
        # start = np.random.choice([x for x in bar])
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            exposure_batch = exposure_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            y_pred = torch.exp(y_pred)
            # y_pred = soft_plus(y_pred)

            loss = loss_fn(y_pred, y_batch, torch.tensor(exposure_batch, requires_grad=False))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            # bar.set_postfix(mse=float(loss))

    # evaluate accuracy at end of each epoch
    with torch.no_grad():
        model.eval()
        y_pred = model(X_val)
        y_train_pred = model(X_train)
        y_pred = torch.exp(y_pred) 
        y_train_pred = torch.exp(y_train_pred) 
        # y_pred = soft_plus(y_pred)
        # y_train_pred = soft_plus(y_train_pred)

        loss_train = loss_fn(y_train_pred, y_train, torch.tensor(exposure_train, requires_grad=False))
        loss_train = float(loss_train)
        loss_val = loss_fn(y_pred, y_val, torch.tensor(exposure_val, requires_grad=False))
        loss_val = float(loss_val)

        r_sq = float(r2_score(y_val.numpy(), y_pred.numpy()))

        history_train.append(loss_train)
        history_test.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = copy.deepcopy(model.state_dict())
    progress_ui(epoch_num=epoch, epoch_max=n_epochs, loss_val=loss_val, loss_train=loss_train, r_sq=r_sq)
 
# restore model and return best accuracy
model.load_state_dict(best_weights)


# PATH = './sourceCode/neural_networks/models'
# torch.save(model.state_dict(), PATH)

# model = Net()
# model.load_state_dict(torch.load(PATH, weights_only=True))

print("Best Validation Loss: %.10f" % best_loss)

y_predicted = model(X_val)
y_predicted = torch.exp(y_predicted)
# y_predicted = soft_plus(y_predicted)
print("Variance: ", torch.var(y_predicted))
print("Max: ", torch.max(y_predicted))
print("Min: ", torch.min(y_predicted))
print("R^2: ", float(r2_score(y_val.numpy(), y_predicted.detach().numpy())))

y_val_pred = y_predicted.detach().numpy()

y_results = pd.concat([pd.DataFrame(y_val,columns=['ClaimsNb']), pd.DataFrame(y_val_pred,columns=['Predicted'])], axis=1)
y_results.to_csv("results.csv", index=False)

# print("RMSE: %.10f" % np.sqrt(best_mse))
plt.plot(history_test, label='Testing Loss')
plt.plot(history_train, label='Training Loss')
plt.show()

plt.clf()

# temp_samp = np.random.randint(X_val.shape[0], size=20000)
# X_sample = X_val[temp_samp, :]
# Y_sample = y_val[temp_samp, :]

# history_samp = []

# for i in range(1,11):
#     a = i * 0.1
#     X_sample_temp = np.copy(X_sample)
#     X_sample_temp[:, 0] = i
#     y_pred_sample = model(torch.tensor(X_sample_temp, dtype=torch.float32))
#     y_pred_sample = torch.exp(y_pred_sample)
#     history_samp.append(torch.mean(y_pred_sample).detach().numpy())

# plt.plot(np.arange(1,11)*0.1, history_samp)
# plt.show()