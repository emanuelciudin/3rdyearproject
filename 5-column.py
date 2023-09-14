# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from numpy import argmax
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
from collections import Counter


df_Y_AF = np.array(df_Y_AF)
df_X_AF1 = np.array(df_X_AF)
X = torch.from_numpy(df_X_AF1).float()
Y = torch.from_numpy(df_Y_AF).float()


# Define the dataset
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Split the data into training and validation sets

train_val_data, test_data, train_val_labels, test_labels = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 42)

# Create the dataset and data loader

dataset = ECGDataset(train_val_data, train_val_labels)

    #ORGINAL DATA DISTRIBUTION TEST SET
test_dataset = ECGDataset(test_data, test_labels)


data_loader = DataLoader(dataset, batch_size= 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size= 64, shuffle = False)

# Define the CNN
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()

        self.fc = nn.Linear(5, 1)
        self.fc_col = nn.Linear(64, 1)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.layers = [[],[],[],[],[]]

        for i in range(5):
          self.layers[i].append(nn.Conv1d(1, 16, kernel_size = 20, stride = 2, padding = 'valid'))
          self.layers[i].append(nn.BatchNorm1d(16))

          self.layers[i].append(nn.MaxPool1d(3,2))

          self.layers[i].append(nn.Conv1d(16, 64, kernel_size = 11, stride = 2, padding = 'valid'))
          self.layers[i].append(nn.BatchNorm1d(64))

          self.layers[i].append(nn.MaxPool1d(3,2))

          # self.layers[i].append(nn.MaxPool1d(3,2))
          self.layers[i].append(nn.Dropout(0.3))
          self.layers[i].append(nn.Linear(121, 16))
          self.layers[i].append(nn.Dropout(0.3))
          self.layers[i].append(nn.Linear(16, 1))

    def forward(self, x):
        x_cols = [[], [], [], [], []]
        for i in range(5):
          x_cols[i] = x[:,:,i].unsqueeze(1)

          x_cols[i] = self.layers[i][0](x_cols[i])
          x_cols[i] = self.act(x_cols[i])
          x_cols[i] = self.layers[i][1](x_cols[i])

          x_cols[i] = self.layers[i][2](x_cols[i])

          x_cols[i] = self.layers[i][3](x_cols[i])
          x_cols[i] = self.act(x_cols[i])
          x_cols[i] = self.layers[i][4](x_cols[i])

          x_cols[i] = self.layers[i][5](x_cols[i])

          x_cols[i] = self.layers[i][6](x_cols[i])
          x_cols[i] = self.layers[i][7](x_cols[i])

          x_cols[i] = self.layers[i][8](x_cols[i])
          x_cols[i] = self.layers[i][9](x_cols[i])
          

          x_cols[i] = self.fc_col(x_cols[i].view(-1,64))
          #print(x_cols[i].shape)
          #x_cols[i] = self.sigmoid(x_cols[i])
        x = torch.cat((*x_cols, ), 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Define the model and move it to the device

device = torch.device('cpu')
model = ECGClassifier()
model = model.to(device)
#model = model.float()

# Define the loss function and optimizer

# Train the model

n_splits = 5
n_epochs = 8
total_train = 0
correct_train = 0

skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=42)


# IMPLEMENTING CROSS-VALIDATION

for fold, (train_index, val_index) in enumerate(skf.split(dataset.data, dataset.labels)):
    print(f"Fold: {fold + 1}")
    # Create the dataset for training and validation

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    #print(train_dataset.shape)
    # Create the dataloader for training and validation

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False)

    # # IMPLEMENTING SMOTE OVERSAMPLING:
    X_train = []
    Y_train = []
    for i in range(len(train_dataset)):
        x, y = train_dataset[i]
        if isinstance(x, np.ndarray):
            # Append x if it already is a numpy array
            X_train.append(x.reshape(-1))
        else:
            # Convert x to a numpy array if it is not already and reshape it
            X_train.append(np.array(x).reshape(-1))
        Y_train.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # Initialise SMOTE Oversampling
    counterbf = Counter(Y_train)
    print('The class distribution before the oversampling is:', counterbf)


# Setting the model to training mode
    model.train()

    for epoch in range(n_epochs):
        print(f"Epoch: {epoch + 1}")
        train_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            # Forward pass

            outputs = model(data)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            predicted_train = []
            predicted_AF_train = 0
            true_labels_AF = 0
            for i in range(len(outputs)):
              if outputs[i]<0.3:
                predicted_train.append(0)
              else:
                predicted_train.append(1)
            predicted_train = np.array(predicted_train)
            labels = labels.detach().numpy()
            total_train += len(predicted_train)
            for i in range(len(predicted_train)):
              if predicted_train[i] == labels[i]:
                correct_train += 1
            for i in range(len(predicted_train)):
              if labels[i] > 0.99:
                true_labels_AF += 1
              if predicted_train[i] > 0.3 and labels[i] > 0.5:
                predicted_AF_train += 1

            # Backward and optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        #print('current epoch train loss is:', loss.item()*data.size(0))

        # Set the model to evaluation mode 
        model.eval()
        # Validate the model
        with torch.no_grad():
          val_loss = 0.0
          for i, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            # Forward pass
            outputs = model(data)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()*data.size(0)
        #print('current epoch validation loss is:', loss.item()*data.size(0))
    # Dividing total loss for the length of the loaders
    val_loss /= len(val_loader)
    train_loss /= len(train_loader)

    # Set the model back to training mode
    model.train()
    print ('Epoch [{}/{}], Training loss is: {:.4f}, Validation loss is {:.4f}'.format(epoch+1, n_epochs, train_loss, val_loss))
 
print('Train Accuracy of the model on the test input: {} %'.format(100 * correct_train / total_train), 'Accurately predicted AF percentage in training is:', predicted_AF_train*100/true_labels_AF)
#model.load_state_dict(torch.load('best_model.pth'))

y_predi = []
y_true = []

    # Test the model

with torch.no_grad():
   correct = 0
   total = 0
   for data, labels in test_loader:
       data, labels = data.to(device), labels.to(device)
       outputs = model(data)
       y_predi.extend(outputs.cpu().numpy().tolist())
       y_true.extend(labels.cpu().numpy().tolist())
       predicted = []
       for i in range(len(outputs)):
         if outputs[i] < 0.3:
           predicted.append(0)
         else:
           predicted.append(1)
       predicted = np.array(predicted)
       labels = labels.detach().numpy()
       total += len(predicted)
       for i in range(len(predicted)):
         if predicted[i] == labels[i]:
           correct += 1

   print('Test Accuracy of the model on the test input: {} %'.format(100 * correct / total))

# AUC Calculation


y_predi_AUC = []
y_true_AUC = []
y_predi_1 = np.array(y_predi, dtype=np.float32)
for i in range(len(y_true)):
  y_predi_AUC.append(y_predi_1[i])
  y_true_AUC.append(y_true[i])
auc = roc_auc_score(y_true_AUC, y_predi_AUC)
print("The Area under the ROC (AUROC) is:", auc)