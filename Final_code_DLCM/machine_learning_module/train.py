from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from pycm import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

from ML_utils import *
from hex_nn_models import *
from tetra_nn_models import *

############# Define function ###########
#----------------------------------------------------------


def train_model(model, control_seed=True):
  """The train model function, which controls the whole training process

  input
  model: model to train

  return
  trained model

  """
  if control_seed:
      set_seed(seed)
  # Default for cpu conditon  
  num_workers = 0
  if device == 'gpu':
      num_workers = 4

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss()

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  validation_loader = DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)


  for epoch in range(n_epochs):

      train_loss, train_accuracy = train(model, optimizer, criterion, train_loader, size, device)
      validation_loss, validation_accuracy = validate(model, criterion, validation_loader, size, device)
      
      print('------------------------------------------------------------------------------------')
      print('Current epoch: ', epoch)
      print('train loss: ', train_loss.item(), ' train accuracy: ', train_accuracy)
      print('validation_loss: ', validation_loss.item(), ' train accuracy: ', validation_accuracy)
      print('------------------------------------------------------------------------------------')

  ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
  model_save_name = 'DLCM' + st + '.pt'
  PATH = F"/content/gdrive/My Drive/{model_save_name}"

  # Save the model and all parameters
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, PATH)

  return model


############# Data loading part ###########
#-------------------------------------------------------------------------------

drive_path = "../data"
# select data
tetra = True
hex = False

############## Set parameters #############
#---------------------------------------------------
control_seed = True
seed = 66 # seed
lr = 1e-3 # learning rate
batch_size = 32
test_batch_size = 32
n_epochs = 200 # training epochs
weight_decay = 0.0 # Weight decay value for adam, refernce value: 0.0, 1e-3, 1e-4, 1e-5


############## Set element related parameters and load model #############
# Model refer table
# Tetra
# DLCM_tetra_10_2, DLCM_tetra_10_3, DLCM_tetra_10_4, DLCM_tetra_10_5, DLCM_tetra_10_6, DLCM_tetra_10_7
# Hex
# DLCM_hex_8_2, DLCM_hex_8_3, DLCM_hex_8_4, DLCM_hex_8_5, DLCM_hex_8_6
#---------------------------------------------------

# Default set to tetra 10 layer 6 model
size = 10 * 3
model_selected = DLCM_tetra_10_6().to(device)


if tetra:
    #tetra training data
    data_features = np.load(drive_path+"/train_data_tetra_10/train_features.npy")
    train_data_features = data_features
    # Labels
    data_label = np.load(drive_path + "/train_data_tetra_10/labels.npy")
    train_data_label = data_label
    # Neural network input size
    size = 10 * 3
    model_selected = DLCM_tetra_10_6().to(device)
if hex:
    # hex training data
    data_features = np.load(drive_path+"/train_data_hex_08/train_features.npy")
    train_data_features = data_features
    # Labels
    data_label = np.load(drive_path + "/train_data_hex_08/labels.npy")
    train_data_label = data_label
    # Neural network input size
    size = 8 * 3
    model_selected = DLCM_hex_8_5().to(device)


############# Dataset division part ###########
#--------------------------------------------------------
# Do dataset division
train_data_label_index = np.arange(0,train_data_label.shape[0],1)
shuffler = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(train_data_features, train_data_label_index)
indices = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]
X_train, y_train = torch.from_numpy(train_data_features[indices[0]]), torch.from_numpy(train_data_label[indices[0]])
X_val, y_val = torch.from_numpy(train_data_features[indices[1]]), torch.from_numpy(train_data_label[indices[1]])
print("View the shape of train set and validation set: ")
print("train", X_train.shape)
print("val", X_val.shape)


############# Normlization part ###########
#--------------------------------------------------------
# Calculate mean and std
mean = train_data_features.mean(axis=(0,1,2))
std = train_data_features.std(axis=(0,1,2))
print("View the overall mean and std: ")
print(train_data_features.shape)
print('mean', mean)
print('std', std)
# Do data normlization
X_train = Normlization(X_train.float(), mean, std)
X_val = Normlization(X_val.float(),mean, std)


# Construct training data
train_data = TensorDataset(X_train, y_train.long())
validate_data = TensorDataset(X_val, y_val.long())


############# Train model #############
#---------------------------------------------
model_trained = train_model(model_selected, control_seed)

############# Calculate Confusion Matrix #############
#----------------------------------------------
test_loader = DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=0)
y_pred, y_gt = evaluate(model_trained, test_loader, size, device)
cm = ConfusionMatrix(actual_vector=y_gt, predict_vector=y_pred) # Create CM From Data
print(cm)
