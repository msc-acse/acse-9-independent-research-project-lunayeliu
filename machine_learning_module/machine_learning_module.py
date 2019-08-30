!pip install pycm livelossplot
%pylab inline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
import livelossplot
from livelossplot import PlotLosses
from pycm import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
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

############# Data loading part ###########
#-------------------------------------------------------------------------------
# Mount the goole drive for data
from google.colab import drive
drive.mount('/content/gdrive/')
drive_path = "./gdrive/My Drive/Colab Notebooks/ml/data/cm"
# select data
tetra = True
hex = False

size = 8 * 3
if tetra:
    #tetra training data
    data_features = np.load(drive_path+"/train_data_tetra_10/train_features.npy")
    train_data_features = data_features
    # Labels
    data_label = np.load(drive_path + "/train_data_tetra_10/labels.npy")
    train_data_label = data_label
    # Neural network input size
    size = 10 * 3
if hex:
    # hex training data
    data_features = np.load(drive_path+"/train_label_hex_08/train_features.npy")
    train_data_features = data_features
    # Labels
    data_label = np.load(drive_path + "/train_label_hex_08/labels.npy")
    train_data_label = data_label
    # Neural network input size
    size = 8 * 3


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
# Construct training data
train_data = TensorDataset(X_train, y_train.long())
validate_data = TensorDataset(X_val, y_val.long())


############## Set parameters #############
#---------------------------------------------------
control_seed = True
seed = 66 # seed
lr = 1e-3 # learning rate
batch_size = 32
test_batch_size = 32
n_epochs = 200 # training epochs
weight_decay = 0.0 # Weight decay value for adam, refernce value: 0.0, 1e-3, 1e-4, 1e-5


############## Load model #############
# Model refer table
# Tetra
# DLCM_tetra_10_2, DLCM_tetra_10_3, DLCM_tetra_10_4, DLCM_tetra_10_5, DLCM_tetra_10_6, DLCM_tetra_10_7
# Hex
# DLCM_hex_8_2, DLCM_hex_8_3, DLCM_hex_8_4, DLCM_hex_8_5, DLCM_hex_8_6
#---------------------------------------------------
model_selected = DLCM_hex_8_5().to(device)

############# Train model #############
#---------------------------------------------
model_trained = train_model(model_selected)

############# Calculate Confusion Matrix #############
#----------------------------------------------
test_loader = DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=0)
y_pred, y_gt = evaluate(model_trained, test_loader)
cm = ConfusionMatrix(actual_vector=y_gt, predict_vector=y_pred) # Create CM From Data
print(cm)
