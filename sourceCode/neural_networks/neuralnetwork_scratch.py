import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
from scipy.special import factorial as factorial

def initialize_weights_normal(mean, std, shape):
    return np.random.normal(mean, std, shape)

def relu(z):
    # print(z)
    # res = z.copy()
    # res = np.where(res<0, 0)
    # return res
    return np.array(np.maximum(0, z))

def relu_derivative(z):
    # res = z.copy()
    # res = np.where(res>=0, 1, res)
    # res = np.where(res<0, 0, res)
    # return res
    return np.array((z > 0).astype(float))

def elu(x, alpha=1): 
    res = x.copy()
    res = np.where(res<0, alpha*np.exp(res)-1, res)
    return res

def elu_derivative(x,alpha=1):
    res = x.copy()
    res = np.where(res>=0, 1, res)
    res = np.where(res<0, alpha*np.exp(res), res)
    return res

# def b_side_mpdv(x,b=1):
#     res = None
#     res = np.where(res>=y_true, 2*b*(1-(y_true/y_pred_clipped)), res)
#     res = np.where(res<y_true, 2*(1-(y_true/y_pred_clipped)), res)
#     return res

def b_side_mpdv_derivative(y_true,y_pred,b=1):
    temp_1 = y_true.copy()
    temp_2 = y_pred.copy()
    res = temp_1/temp_2
    res = np.where(y_pred>=y_true, 2*b*(1-res), res)
    res = np.where(y_pred<y_true, 2*(1-res), res)
    return res

def c_side_mpdv_derivative(y_true,y_pred,b=1,c=1):
    temp_1 = y_true.copy()
    temp_2 = y_pred.copy()
    res = temp_1/temp_2
    res = np.where(y_pred>=y_true, 2*b*(1-res), res)
    res = np.where(y_true==0, 2*c*res, res)
    res = np.where(y_pred<y_true, 2*c*(1-res), res)
    return res

def softplus(z, beta=1):
    return (1/beta)*np.log(1+np.exp(beta*z))

def softplus_derivative(z, beta=1):
    return np.exp(beta*z)/(np.exp(beta*z)+1)

# Regression Cost Functions
def rss(y_true: np.array, y_pred: np.array):
    return sum(np.pow(y_true-y_pred,2))

def mse(y_true,y_pred):
    return (1/len(y_true))*rss(y_true,y_pred)

class NN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        """
        Initialize network parameters
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden layer neurons
            output_dim: Number of output classes
            learning_rate: Learning rate for gradient descent
        """
        self.lr = learning_rate
        
        # Initialize weights and biases with small random values
        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        """
        Forward pass through the network
        Args:
            X: Input data of shape (batch_size, input_dim)
        Returns:
            y_pred: Predicted probabilities of shape (batch_size, output_dim)
        """
        # >>> Hidden layer
        self.z1 = (X @ self.W1) + self.b1 # dim: batch_size x hidden_dim

        self.a1 = elu(self.z1) # Exponential Linear Activation
        # self.a1 = relu(self.z1) # Rectified Linear Unit Activation
        
        # >>> Output layer
        self.z2 = (self.a1 @ self.W2) + self.b2
        # self.y_pred = np.array(self.z2).reshape((-1,)) # Linear Activation
        # self.y_pred = np.exp(np.array(self.z2)).reshape((-1,)) # Exponential Activation
        self.y_pred = softplus(np.array(self.z2)).reshape((-1,)) # Softplus Activation
        # self.y_pred = relu(np.array(self.z2)).reshape((-1,)) # ReLU Activation

        

        return self.y_pred

    def backward(self, X, y_true, exposure_df):
        """
        Backward pass - compute gradients
        
        Args:
            X: Input data of shape (batch_size, input_dim)
            y_true: True labels of shape (batch_size, output_dim)
        """
        batch_size = X.shape[0]
        
        # >>> Output layer gradients
        # >> RSS Cost Function Derivative
        # temp_1 = self.y_pred - y_true # dim: output_dim x batch_size

        # >> MPDv Cost Function Derivative
        y_pred_clipped = np.clip(self.y_pred, 1e-12, None, dtype=np.float64)
        y_pred_clipped = self.y_pred # when output layer activation function = exponential, y_pred never reaches 0, so already clipped, but problems with weight initialization
        temp_1 = c_side_mpdv_derivative(y_true, y_pred_clipped, b=1, c=1)

        temp_2 = np.array(exposure_df).reshape(-1,)
        if temp_1.shape != temp_2.shape:
            raise Exception(f"{temp_1.shape} != {temp_2.shape}")
        dLdyhat = temp_1 * temp_2

        # > Linear Activation
        # dLdz2 = dLdyhat # because dYhatdz2 is an identity matrix, derivative of linear activation function
        # > Exponential Activation
        # dLdz2 = np.array(dLdyhat).reshape(-1,1) * np.exp(self.z2).reshape(-1,1)
        # > Softplus Activation
        dLdz2 = np.array(dLdyhat).reshape(-1,1) * softplus_derivative(self.z2).reshape(-1,1)
        # > ReLU Activation
        # dLdz2 = np.array(dLdyhat).reshape(-1,1) * relu_derivative(self.z2).reshape(-1,1)

        self.dLdW2 = (self.a1.T @ dLdz2) / batch_size  # dim: batch_size x hidden_dim
        self.db2 = np.sum(dLdz2, axis=0) / batch_size  # Shape: (1, output_dim) # untouched

        # >>> Hidden layer gradients
        dLda1 = np.array(dLdz2).reshape(-1,1) @ self.W2.T
        dLdz1 = dLda1 * elu_derivative(self.z1)
        # dLdz1 = dLda1 * relu_derivative(self.z1)

        self.dLdW1 = (X.T @ dLdz1) / batch_size
        self.db1 = np.sum(dLdz1, axis=0) / batch_size  # Shape: (1, hidden_dim) # untouched 

    def update_weights(self):
        """Update weights using gradient descent"""
        self.W1 -= self.lr * self.dLdW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dLdW2.reshape(-1,1)
        self.b2 -= self.lr * self.db2

    def compute_loss(self, y_true, y_pred, exposure_df):
        y_pred_clipped = np.array(np.clip(y_pred, 0, None, dtype=np.float64))

        # > Idon'tknowwhich loss function
        # loss = -np.sum(np.array(y_true, dtype=np.float64) * np.log(y_pred_clipped)) / y_true.shape[0]

        # > Mean Poisson Deviance Loss
        # y_true_in = np.array(y_true).reshape(1,-1)
        # loss = sklearn.metrics.mean_poisson_deviance(y_true_in[0],y_pred_clipped)

        # > RSS Loss
        temp_1 = np.power(np.array(y_pred_clipped) - np.array(y_true),2)
        temp_2 = np.array(exposure_df).reshape(-1,)
        if temp_1.shape != temp_2.shape:
            raise Exception(f"{temp_1.shape} != {temp_2.shape}")
        loss = np.mean(temp_1 * temp_2)

        return loss
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X).reshape(-1,1)
        return y_pred
    
    def accuracy(self, X, y_true):
        """Compute accuracy"""
        y_pred = self.predict(X)
        # y_true_labels = np.argmax(y_true, axis=1)
        y_true_labels = np.array(y_true).reshape(-1,1)
        return np.mean(y_pred == y_true_labels)

    def poisson_deviance(self, X, y_true, exposure_df):
        """Compute mean poisson deviance"""
        y_pred_in = self.predict(X).reshape(1,-1)
        y_pred_clipped = np.array(np.clip(y_pred_in, 1e-12, None, dtype=np.float64))
        y_true_in = np.array(y_true).reshape(1,-1)

        temp_1 = y_true_in[0]
        temp_2 = np.array(exposure_df).reshape(-1,)
        if temp_1.shape != temp_2.shape:
            raise Exception(f"{temp_1.shape} != {temp_2.shape}")
        # try:

        a = y_pred_clipped[0]
        b = y_true_in[0]
        return np.mean(a-b*np.log(a) + np.log(factorial(b)))

        # return sklearn.metrics.mean_poisson_deviance(y_true_in[0],y_pred_clipped[0], sample_weight=temp_2)
        # except Exception as e:
            # print(e)
            # return 0


# Train the network
print("\nTraining Neural Network with Backpropagation")

df_training = pd.read_csv("data/claims_train_scaled.csv")
# > Exclude Categorical Variables
# X = df_training.loc[:, ~df_training.columns.isin(['ClaimNb','Region','VehGas','VehBrand','Area'])] # excludes categorical variables

# > Include Categorical Variables
no_claims_exposure = np.sum(df_training[df_training['ClaimNb']==0].loc[:,'Exposure'], axis=0)
positive_claims_exposure = np.sum(df_training[df_training['ClaimNb']>0].loc[:,'Exposure'], axis=0)
ratio_exposure = no_claims_exposure/positive_claims_exposure
# print(ratio_exposure)
# > NOTE: technically turns data exposure_train_df and exposure_val_df into sample weights (which exposure is), but too lazy to change var name
# df_no_claims = df_training[df_training['ClaimNb']==0].copy()
# df_positive_claims = df_training[df_training['ClaimNb']>0].copy()


# df_positive_claims.loc[:,'Exposure'] = df_positive_claims.loc[:,'Exposure'] * ratio_exposure
# df_training[df_training['ClaimNb']>0] = df_positive_claims.copy()
# > NOTE END

X = df_training.loc[:, df_training.columns != 'ClaimNb']
X = pd.get_dummies(X, columns=['Region','VehGas','VehBrand','Area'], drop_first=True, dtype=float)

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

# Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = 1 
learning_rate = 0.05
num_epochs = 100

# Initialize network
model = NN(input_dim, hidden_dim, output_dim, learning_rate)

# Training loop
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred_train = model.forward(X_train)
    
    # Compute loss
    train_loss = model.compute_loss(y_train, y_pred_train, exposure_train_df)
    
    # Backward pass
    model.backward(X_train, y_train, exposure_train_df)
    
    # Update weights
    model.update_weights()
    
    # Compute validation loss and accuracy
    y_pred_val = model.forward(X_val)
    val_loss = model.compute_loss(y_val, y_pred_val, exposure_val_df)
    
    train_acc = model.poisson_deviance(X_train, y_train, exposure_train_df) 
    val_acc = model.poisson_deviance(X_val, y_val, exposure_val_df) 
    
    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print("~"*125)
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train RSS Loss: {train_loss:.4f} | Val RSS Loss: {val_loss:.4f} | "
              f"Train MPDv: {train_acc:.4f} | Val MPDv: {val_acc:.4f} | "
              f"Average Weight (layer 1): {np.mean(model.W1):.4f} | "
              f"Average Weight (layer 2): {np.mean(model.W2):.4f} | "
              f"Average Bias (layer 1): {np.mean(model.b1):.4f} | "
              f"Average Bias (layer 2): {np.mean(model.b2):.4f} | "
              f"Net Gradient Change (layer 1): {np.mean(np.abs(model.dLdW1)):.4f} | "
              f"Net Gradient Change (layer 2): {np.mean(np.abs(model.dLdW2)):.4f} | "
              )

print("\nTraining Complete!")
print(f"Final Training MPDv: {train_accuracies[-1]:.4f}")
print(f"Final Validation MPDv: {val_accuracies[-1]:.4f}")


# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RSS Loss curve
axes[0].plot(train_losses, label='Training Loss', linewidth=2)
axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 2])
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(train_accuracies, label='Training MPDv', linewidth=2)
axes[1].plot(val_accuracies, label='Validation MPDv', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MPDv', fontsize=12)
axes[1].set_title('Training and Validation MPDv', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 2])
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

y_val_pred = model.predict(X_val)
for i in range(len(y_val)):
    if i%5000 == 0:
        temp = str(y_val_pred[i][0])
        print(temp + " "*(20-len(temp)) + " | " + str(np.array(y_val)[i]))
print("Notable Results:")
notable_results = []
y_val = np.array(y_val).reshape(-1,1)
max_y_val_pred = max(y_val_pred)
for i in range(len(y_val)):
    if y_val[i] > 1 or y_val_pred[i] > 1:
        notable_results.append((str(y_val_pred[i][0]), str(np.array(y_val)[i])))
notable_results = sorted(notable_results, key=lambda x:x[0])
notable_results = sorted(notable_results, key=lambda x:x[1])
for x in enumerate(notable_results):
    index = x[0]
    i = x[1]
    if (index % (len(notable_results)/30)) == 0:
        temp = i[0]
        print(temp + " "*(20-len(temp)) + " | " + i[1])

print("Max: " + str(max_y_val_pred))

def try_block(claim_num):
    if not len(y_val_pred[y_val == claim_num]) == 0:
        print(f"Claims = {claim_num} / Avg Prediction: " + str(np.mean(y_val_pred[y_val == claim_num])))
    else:
        print(f"Claims = {claim_num} / No such values in validation set.")
for i in range(6):
    try_block(i)