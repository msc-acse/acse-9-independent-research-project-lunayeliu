import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from torch.utils.data import TensorDataset, DataLoader

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

from ML_utils import *
from hex_nn_models import *
from tetra_nn_models import *

path = "../trained_model/"

# select data
tetra = True
hex = True

model_name_list = glob.glob(path+'*.pt')

# Model refer table
# Tetra
# DLCM_tetra_10_2, DLCM_tetra_10_3, DLCM_tetra_10_4, DLCM_tetra_10_5, DLCM_tetra_10_6, DLCM_tetra_10_7
# Hex
# DLCM_hex_8_2, DLCM_hex_8_3, DLCM_hex_8_4, DLCM_hex_8_5, DLCM_hex_8_6

if tetra:
    model_name = model_name_list[1]
    model_loaded =  DLCM_tetra_10_6()
    # Define test data

    x= np.array([[0.0, 0.0, 0.0],
         [1.1770690139471445, 0.0, 0.0],
         [0.0, 1.008221935884381, 0.0],
         [0.0, 0.0, 1.0742815356628406],
         [0.798555480574705, -0.29855548057470505, 0.0],
         [0.7640472671189913, 0.7640472671189913, 0.0],
         [-0.06675474221278772, 0.5667547422127878, 0.0],
         [-0.21765881660728761, -0.21765881660728761, 0.7176588166072876],
         [0.6869612116309823, 0.18696121163098234, 0.5],
         [0.08444066303194893, 0.584440663031949, 0.5]]).flatten()
    x = np.array([x])
    model_loaded.load_state_dict(torch.load(model_name, map_location=device)['model_state_dict'])
    model_loaded.to(device)
    model_loaded.eval()
    print('Model for tetra load successful.')
    
    x = torch.from_numpy(x).float()
    y = model_loaded(x)
    y_pred_tetra = F.log_softmax(y, dim=1).max(1)[1]
    

if hex:
    model_name = model_name_list[0]
    model_loaded =  DLCM_hex_8_7()
    # Define test data
    x = np.array([[-1.0, -1.0, -1.0],
                 [1.0, -1.0, -1.0],
                 [1.0, 1.0, -1.0],
                 [-1.0, 1.0, -1.0],
                 [-1.0, -1.0, 1.0],
                 [1.0, -1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [-1.0, 1.0, 1.0]]).flatten()
    x = np.array([x])

    model_loaded.load_state_dict(torch.load(model_name, map_location=device)['model_state_dict'])
    model_loaded.to(device)
    model_loaded.eval()
    print('Model for hex Load successful.')
    
    x = torch.from_numpy(x).float()
    y = model_loaded(x)
    y_pred_hex = F.log_softmax(y, dim=1).max(1)[1]
    


print("The predict number of integration point for the test hexahedron is: ", y_pred_hex[0].numpy())
print("The predict number of integration point for the test tetrahedron is: ", y_pred_tetra[0].numpy())
print('---------------------------------------------------------')
print("If you see this line, that means both the trainined models worked!")





