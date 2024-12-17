

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))

def mae(x,y):
    return np.mean(np.abs((x-y)))

df = pd.read_csv('SeoulBikeData.csv')
df.index = df.index + 1
df

rent_org = df['rent']
plt.figure(figsize=(15, 1)) 
plt.plot(rent_org[-100*24:])
plt.title('last 100 days')
plt.show()

# Setting

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

time_period = 30
rent = df[df['hour'] == 18]
rent = df['rent'][:-time_period]
rent_monthly = rent[-time_period:]
maxrent, minrent = np.max(rent), np.min(rent)
scaled_rent = (rent - minrent) / (maxrent - minrent)  

def generate_data(seq_length):
    X = []
    y = []
    for i in range(len(rent) - seq_length):
        X.append(scaled_rent[i:i + seq_length])
        y.append(scaled_rent[i + seq_length])  
    return np.array(X), np.array(y)

X, y = generate_data(time_period)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class SimpleRNN(nn.Module):
    def __init__(self, network, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size 
        if network == 'RNN':
            self.network = nn.RNN(input_size, hidden_size, batch_first=True)

        elif network == 'GRU':
            self.network = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.network(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state to zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Cell state
        
        # Pass input through LSTM layer
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out is the output, hn is the hidden state
        
        # Only use the hidden state from the last time step
        out = self.fc(out[:, -1, :])  # Take the last time step's output for prediction
        return out

def training(network):
    input_size, hidden_size, output_size = 1, 20, 1
    if (network == 'RNN') or (network ==  'GRU'):
        model = SimpleRNN(network,input_size, hidden_size, output_size)
    elif network == 'LSTM':
        model = SimpleLSTM(input_size, hidden_size, output_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        X_input = X.unsqueeze(2)
        outputs = model(X_input) 
        y_input = y.unsqueeze(1)
        loss = criterion(outputs, y_input)
    
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
    
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
    model.eval()

    with torch.no_grad():
        X_input = X.unsqueeze(2)  
        predictions = model(X_input)  # Predictions from the model
    
    # Convert predictions and actual values to NumPy
    predictions = predictions.numpy().flatten()  # Flatten to remove batch and sequence dimensions
    y_actual = y.numpy()
    
    # Inverse Min-Max scaling
    predictions_rescaled = predictions * (maxrent - minrent) + minrent
    y_rescaled = y_actual * (maxrent - minrent) + minrent
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_rescaled[-time_period:], label='Actual Rent', color='blue')
    # plt.plot(predictions_rescaled[-time_period:], label='Predicted Rent', color='red', linestyle='dashed')
    # plt.title(f'Rent Predictions vs Actual Rent ({network})')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Rent Value')
    # plt.legend()
    # plt.show()
    Xnew = []
    for i in range(time_period):
        Xnew.append(scaled_rent[len(rent)-2*time_period+i+1:len(rent)-time_period+i])
    Xnew = np.array(Xnew, dtype=np.float32)
    Xnew = torch.tensor(Xnew, dtype=torch.float32)
    
    with torch.no_grad():
        X_input = Xnew.unsqueeze(2)  
        predictions = model(X_input)  # Predictions from the model
    
    # Convert predictions and actual values to NumPy
    predictions = predictions.numpy().flatten()  # Flatten to remove batch and sequence dimensions
    y_actual = y.numpy()
    
    # Inverse Min-Max scaling
    predictions_rescaled = predictions * (maxrent - minrent) + minrent
    
    # np.save(f"{network}_{time_period}", predictions_rescaled)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(rent_org)-1000,len(rent_org)), rent_org[-1000:], label='Actual Rent', color='blue')
    # plt.plot(range(len(rent), len(rent)+time_period), predictions_rescaled, label='Predicted Rent', color='red', linestyle='dashed')
    
    # # Adding titles and labels
    # plt.title(f'Rent Predictions vs Actual Rent ({network})')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Rent Value')
    
    # plt.legend()
    
    # plt.show()
    return predictions_rescaled

# Prediction

## RNN
prediction_RNN = training('RNN')
print(rmse(rent_monthly, prediction_RNN))
print(mae(rent_monthly, prediction_RNN))


## LSTM
prediction_LSTM = training('LSTM')
print(rmse(rent_monthly, prediction_LSTM))
print(mae(rent_monthly, prediction_LSTM))

## GRU
prediction_GRU = training('GRU')
print(rmse(rent_monthly, prediction_GRU))
print(mae(rent_monthly, prediction_GRU))
RNN

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def rmse(x,y):
    return np.sqrt(np.mean((x-y)**2))

def mae(x,y):
    return np.mean(np.abs((x-y)))

df = pd.read_csv('SeoulBikeData.csv')
df.index = df.index + 1
df

rent_org = df['rent']
plt.figure(figsize=(15, 1)) 
plt.plot(rent_org[-100*24:])
plt.title('last 100 days')
plt.show()

# Setting

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

time_period = 30
rent = df[df['hour'] == 18]
rent = df['rent'][:-time_period]
rent_monthly = rent[-time_period:]
maxrent, minrent = np.max(rent), np.min(rent)
scaled_rent = (rent - minrent) / (maxrent - minrent)  

def generate_data(seq_length):
    X = []
    y = []
    for i in range(len(rent) - seq_length):
        X.append(scaled_rent[i:i + seq_length])
        y.append(scaled_rent[i + seq_length])  
    return np.array(X), np.array(y)

X, y = generate_data(time_period)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class SimpleRNN(nn.Module):
    def __init__(self, network, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size 
        if network == 'RNN':
            self.network = nn.RNN(input_size, hidden_size, batch_first=True)

        elif network == 'GRU':
            self.network = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.network(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state to zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Cell state
        
        # Pass input through LSTM layer
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out is the output, hn is the hidden state
        
        # Only use the hidden state from the last time step
        out = self.fc(out[:, -1, :])  # Take the last time step's output for prediction
        return out

def training(network):
    input_size, hidden_size, output_size = 1, 20, 1
    if (network == 'RNN') or (network ==  'GRU'):
        model = SimpleRNN(network,input_size, hidden_size, output_size)
    elif network == 'LSTM':
        model = SimpleLSTM(input_size, hidden_size, output_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        X_input = X.unsqueeze(2)
        outputs = model(X_input) 
        y_input = y.unsqueeze(1)
        loss = criterion(outputs, y_input)
    
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
    
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
    model.eval()

    with torch.no_grad():
        X_input = X.unsqueeze(2)  
        predictions = model(X_input)  # Predictions from the model
    
    # Convert predictions and actual values to NumPy
    predictions = predictions.numpy().flatten()  # Flatten to remove batch and sequence dimensions
    y_actual = y.numpy()
    
    # Inverse Min-Max scaling
    predictions_rescaled = predictions * (maxrent - minrent) + minrent
    y_rescaled = y_actual * (maxrent - minrent) + minrent
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_rescaled[-time_period:], label='Actual Rent', color='blue')
    # plt.plot(predictions_rescaled[-time_period:], label='Predicted Rent', color='red', linestyle='dashed')
    # plt.title(f'Rent Predictions vs Actual Rent ({network})')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Rent Value')
    # plt.legend()
    # plt.show()
    Xnew = []
    for i in range(time_period):
        Xnew.append(scaled_rent[len(rent)-2*time_period+i+1:len(rent)-time_period+i])
    Xnew = np.array(Xnew, dtype=np.float32)
    Xnew = torch.tensor(Xnew, dtype=torch.float32)
    
    with torch.no_grad():
        X_input = Xnew.unsqueeze(2)  
        predictions = model(X_input)  # Predictions from the model
    
    # Convert predictions and actual values to NumPy
    predictions = predictions.numpy().flatten()  # Flatten to remove batch and sequence dimensions
    y_actual = y.numpy()
    
    # Inverse Min-Max scaling
    predictions_rescaled = predictions * (maxrent - minrent) + minrent
    
    # np.save(f"{network}_{time_period}", predictions_rescaled)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(rent_org)-1000,len(rent_org)), rent_org[-1000:], label='Actual Rent', color='blue')
    # plt.plot(range(len(rent), len(rent)+time_period), predictions_rescaled, label='Predicted Rent', color='red', linestyle='dashed')
    
    # # Adding titles and labels
    # plt.title(f'Rent Predictions vs Actual Rent ({network})')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Rent Value')
    
    # plt.legend()
    
    # plt.show()
    return predictions_rescaled

# Prediction

## RNN
prediction_RNN = training('RNN')
print(rmse(rent_monthly, prediction_RNN))
print(mae(rent_monthly, prediction_RNN))


## LSTM
prediction_LSTM = training('LSTM')
print(rmse(rent_monthly, prediction_LSTM))
print(mae(rent_monthly, prediction_LSTM))

## GRU
prediction_GRU = training('GRU')
print(rmse(rent_monthly, prediction_GRU))
print(mae(rent_monthly, prediction_GRU))
