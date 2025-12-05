import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tqdm
import copy
import os
import time

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

last_epoch_time = time.time()
average_epoch_time = 0

def progress_ui(epoch_num, epoch_max, loss):
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
    total_estimated_time_left = (epoch_max-epoch_num)*average_epoch_time

    size = os.get_terminal_size()
    for i in range(size[1]-5):
        print('\n')
    print(f"Best Val. Loss: {best_loss}")
    print(f"Val. Loss: {loss} / Average time per epoch: {round(average_epoch_time,2)}s / Est. Remaining Time: {round(total_estimated_time_left,2)}s")
    print(f"Epoch {epoch_num+1} / {epoch_max} | [{bytes((219,)).decode('cp437')*full_count}{"░"*half_count}{" "*empty_count}] ({round((epoch_num+1)/epoch_max*100, 2)}%)")
    last_epoch_time = time.time()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df_training = pd.read_csv("data/claims_train_scaled.csv")
# > Exclude Categorical Variables
# X = df_training.loc[:, ~df_training.columns.isin(['ClaimNb','Region','VehGas','VehBrand','Area'])] # excludes categorical variables

# > NOTE: technically turns data exposure_train_df and exposure_val_df into sample weights (which exposure is), but too lazy to change var name
# no_claims_exposure = np.sum(df_training[df_training['ClaimNb']==0].loc[:,'Exposure'], axis=0)
# positive_claims_exposure = np.sum(df_training[df_training['ClaimNb']>0].loc[:,'Exposure'], axis=0)
# ratio_exposure = no_claims_exposure/positive_claims_exposure
# print(ratio_exposure)
# df_no_claims = df_training[df_training['ClaimNb']==0].copy()
# df_positive_claims = df_training[df_training['ClaimNb']>0].copy()


# df_positive_claims.loc[:,'Exposure'] = df_positive_claims.loc[:,'Exposure'] * ratio_exposure
# df_training[df_training['ClaimNb']>0] = df_positive_claims.copy()
# # df_training[df_training['ClaimNb']>0] = df_positive_claims.copy()
# > NOTE END

# > Include Categorical Variables
X = df_training.loc[:, df_training.columns != 'ClaimNb']
X = pd.get_dummies(X, columns=['Region','VehGas','VehBrand','Area'], drop_first=True, dtype=float)


Y = df_training.loc[:,"ClaimNb"]

# scaler = StandardScaler()
# X = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# X_train = torch.from_numpy(np.array(X_train,dtype="float32"))
# X_val = torch.from_numpy(np.array(X_val,dtype="float32"))
# y_train = torch.from_numpy(np.array(y_train,dtype="float32"))
# y_val = torch.from_numpy(np.array(y_val,dtype="float32"))

X_train = np.array(X_train,dtype="float32")
X_val = np.array(X_val,dtype="float32")
y_train = np.array(y_train,dtype="float32")
y_val = np.array(y_val,dtype="float32")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def mpdv(output, target):
    loss = 2 * (target * torch.log(target/output) + output - target)
    return torch.mean(loss)

sfpl = nn.Softplus()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(43, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, 6)
        self.fc4 = nn.Linear(6, 1)

    def forward(self, x):
        # Hidden Layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = sfpl(self.fc4(x))

        # Output Layer
        # x = F.softplus(x)
        return x

model = Net()

# model = nn.Sequential(
#     nn.Linear(43, 24),
#     nn.ReLU(),
#     nn.Linear(24, 12),
#     nn.ReLU(),
#     nn.Linear(12, 6),
#     nn.ReLU(),
#     nn.Linear(6, 1)
#     # Expo()
# )

loss_fn = nn.PoissonNLLLoss()
# loss_fn = mpdv

optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# training parameters
n_epochs = 25
batch_size = 10
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_loss = np.inf   # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]


            y_pred = model(X_batch)
            # y_pred = torch.exp(y_pred)
            # loss = loss_fn(y_pred, y_batch)
            loss = mpdv(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    with torch.no_grad():
        model.eval()
        y_pred = model(X_val)
        # y_pred = torch.exp(y_pred)
        # loss = loss_fn(y_pred, y_val)
        loss = mpdv(y_pred, y_val)
        loss = float(loss)
        history.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_weights = copy.deepcopy(model.state_dict())
        progress_ui(epoch_num=epoch, epoch_max=n_epochs, loss=loss)

print('Finished Training')

# PATH = './sourceCode/neural_networks/models'
# torch.save(net.state_dict(), PATH)

# model = Net()
# model.load_state_dict(torch.load(PATH, weights_only=True))

y_pred_val = model(X_val)

_, predicted = outputs

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))