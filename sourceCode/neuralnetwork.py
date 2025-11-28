import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma

error_count = 0

def initialize_weights_normal(mean, std, shape):
    return np.random.normal(mean, std, shape)

# def relu(z):
#     res = z.copy()
#     res = np.where(res<0, 0)
#     return res

# def relu_derivative(z):
#     res = z.copy()
#     res = np.where(res>=0, 1, res)
#     res = np.where(res<0, 0, res)
#     return res

def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    try:
        return (z > 0).astype(float)
    except Exception as e:
        print(e + " /// ERROR COUNT: " + error_count)
        return 0

def elu(x, alpha=1): 
    res = x.copy()
    res = np.where(res<0, alpha*np.exp(res)-1, res)
    return res

def elu_derivative(x,alpha=1):
    res = x.copy()
    res = np.where(res>=0, 1, res)
    res = np.where(res<0, alpha*np.exp(res), res)
    return res

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
        # Hidden layer
        self.z1 = (X @ self.W1) + self.b1 # dim: batch_size x hidden_dim
        self.a1 = elu(self.z1) # Exponential Linear Activation
        
        # Output layer
        self.z2 = (self.a1 @ self.W2) + self.b2
        self.y_pred = np.array(self.z2).reshape((-1,)) # Linear Activation
        
        return self.y_pred

    def backward(self, X, y_true):
        """
        Backward pass - compute gradients
        
        Args:
            X: Input data of shape (batch_size, input_dim)
            y_true: True labels of shape (batch_size, output_dim)
        """
        batch_size = X.shape[0]
        
        # >>> Output layer gradients
        # RSS Cost Function Derivative
        dLdyhat = self.y_pred - y_true # dim: output_dim x batch_size

        # MPDv Cost Function Derivative
        # y_pred_clipped = np.clip(self.y_pred, 1e-12, None, dtype=np.float64)
        y_pred_clipped = self.y_pred
        # y_max = np.maximum(y_pred_clipped,y_true)
        # y_min = np.minimum(y_pred_clipped,y_true)
        y_max = y_true
        y_min = y_pred_clipped
        dLdyhat = 2*(1-(y_max/y_min))
        # print(dLdyhat)
        # exit()

        dLdz2 = dLdyhat # because dYhatdz2 is an identity matrix, derivative of linear activation function

        self.dLdW2 = (self.a1.T @ dLdz2) / batch_size  # dim: batch_size x hidden_dim
        self.db2 = np.sum(dLdz2, axis=0) / batch_size  # Shape: (1, output_dim) # untouched

        # >>> Hidden layer gradients
        dLda1 = np.matmul(np.array(dLdz2).reshape(-1,1), self.W2.T)
        # dLda1 = dLdz2 @ self.W2.T  # Backpropagate through W2 # dim: batch_size x hidden_dim
        dLdz1 = dLda1 * elu_derivative(self.z1)

        self.dLdW1 = (X.T @ dLdz1) / batch_size
        self.db1 = np.sum(dLdz1, axis=0) / batch_size  # Shape: (1, hidden_dim) # untouched 

    def update_weights(self):
        """Update weights using gradient descent"""
        self.W1 -= self.lr * self.dLdW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dLdW2.reshape(-1,1)
        self.b2 -= self.lr * self.db2

        print(self.W1)
        print(self.b1)
        print(self.W2)
        print(self.b2)

    def compute_loss(self, y_true, y_pred):
        y_pred_clipped = np.array(np.clip(y_pred, 1e-12, None, dtype=np.float64))

        # Idon'tknowwhich loss function
        # loss = -np.sum(np.array(y_true, dtype=np.float64) * np.log(y_pred_clipped)) / y_true.shape[0]

        # Mean Poisson Deviance Loss
        y_true_in = np.array(y_true).reshape(1,-1)
        loss = sklearn.metrics.mean_poisson_deviance(y_true_in[0],y_pred_clipped)
        return loss
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X).reshape(-1,1)
        # return np.argmax(y_pred)
        return y_pred
    
    def accuracy(self, X, y_true):
        """Compute accuracy"""
        y_pred = self.predict(X)
        # y_true_labels = np.argmax(y_true, axis=1)
        y_true_labels = np.array(y_true).reshape(-1,1)
        return np.mean(y_pred == y_true_labels)

    def poisson_deviance(self, X, y_true):
        y_pred_in = self.predict(X).reshape(1,-1)
        y_pred_clipped = np.array(np.clip(y_pred_in, 1e-12, None, dtype=np.float64))
        y_true_in = np.array(y_true).reshape(1,-1)
        # print(y_pred_clipped)
        # print(y_true_in)

        return sklearn.metrics.mean_poisson_deviance(y_true_in[0],y_pred_clipped[0])


# Train the network
print("\nTraining Neural Network with Backpropagation")

# X_train = np.array([[1,2],[4,3],[2,1],[5,7],[1,5]], dtype=float)
# y_train = np.array([[1],[2],[3],[4],[5]], dtype=float)
# X_val = np.array([[1,3],[2,5]], dtype=float)
# y_val = np.array([[1],[2]], dtype=float)

df_training = pd.read_csv("data/claims_train_final.csv")
df_training = df_training.drop(columns=['IDpol'])
# X = df_training.loc[:, df_training.columns != 'ClaimNb']
X = df_training.loc[:, ~df_training.columns.isin(['ClaimNb', 'Region','VehGas','VehBrand','Area'])]
Y = df_training.loc[:,"ClaimNb"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 8
output_dim = 1 
learning_rate = 0.1
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
    train_loss = model.compute_loss(y_train, y_pred_train)
    
    # Backward pass
    model.backward(X_train, y_train)
    
    # Update weights
    model.update_weights()
    
    # Compute validation loss and accuracy
    y_pred_val = model.forward(X_val)
    val_loss = model.compute_loss(y_val, y_pred_val)
    
    train_acc = model.poisson_deviance(X_train, y_train) # form. acc
    val_acc = model.poisson_deviance(X_val, y_val) # form. acc
    
    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train MPDv: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print("\nTraining Complete!")
print(f"Final Training MPDv: {train_accuracies[-1]:.4f}")
print(f"Final Validation MPDv: {val_accuracies[-1]:.4f}")


# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(train_losses, label='Training Loss', linewidth=2)
axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(train_accuracies, label='Training MPDv', linewidth=2)
axes[1].plot(val_accuracies, label='Validation MPDv', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MPDv', fontsize=12)
axes[1].set_title('Training and Validation MPDv', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# print( sklearn.metrics.mean_poisson_deviance([2,1],[3,4], sample_weight=[1,1]) )
# print( sklearn.metrics.mean_poisson_deviance([2],[3], sample_weight=[1]) )


# from scipy.special import xlogy

# y_true = 1
# y_pred = 1

# print( 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred) )


