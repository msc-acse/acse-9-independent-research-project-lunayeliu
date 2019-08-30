import sys
sys.path.extend(['../utils','../integration_module','../FEM_module','../utils','../elements', '../machine_learning_module'])
import numpy as np
import os
import time
import datetime
from data_generation_tetra_10 import *

d = [0.1, 0.15, 0.20, 0.25, 0.30]
#d = [0.10]
a = -0.5
b = 1.0


N = 20
tol = 1e-3
n_max = 20
# Construct data structure holder
coordinate_list_stratified = []
label_list_stratified = []
coordinate_list_all = []
label_list_all = []

for i in range(len(d)):
    coordinate_list_stratified.append([])
    label_list_stratified.append([])
#print(coordinate_list_stratified)
#print(label_list_stratified)
print('Initialize successful.')

# Four dimensions data structure
# 1 - different d
# 2 - how many samples
# 3 - nodes
# 4 - x, y, z coordinates

print("Hi! You are running the tetrahedron element data generation test scripts.")
print("This script depends on FEM, Elements and Integration modules. ")
print("So if this script runs well, there should be no problem for your installation.")
print("------------------------------------------------------------------------------")
print("There are %d samples to generate in total." % (N*5))


undo = 0
for j in range(len(d)):
    for i in range(N):
        #r = random.random() * (b - a) + a
        coor  = generate_coordinate_tetra_10(random_func, d[j])
        plane_set = plane_generate_tetra_10(coor)

        while plane_judge(plane_set, range=(20, 160)) or central_point_judge(coor):
            #r = random.random() * (b - a) + a
            coor  = generate_coordinate_tetra_10(random_func, d[j])
            plane_set = plane_generate_tetra_10(coor)
            undo += 1
            print(undo)
        # Stratified features list 4-dimension: 5 * N * 8 * 3
        coordinate_list_stratified[j].append(coor)
        # All features list 3-dimension:  (5*N) * 8 * 3
        coordinate_list_all.append(coor)
        element = tetra_10(coor)
        print('regenerate times: ', undo)
        optimal_num = find_optimal_number(element, tol=tol, n_max = n_max)
        # Stratified label list 2-dimension: 5 * N
        label_list_stratified[j].append(optimal_num)
        # All label list 1-dimension:  (5*N)
        label_list_all.append(optimal_num)

    print("d equals ", d[j]," finished.")


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
output_path = "./train_label_%.2f_"%(d[0])+st
os.mkdir(output_path)
np.save(output_path + "/train_features", np.array(coordinate_list_all))
np.save(output_path + "/labels", np.array(label_list_all))
np.save(output_path + "/train_features_stratified", np.array(coordinate_list_stratified))
np.save(output_path + "/labels_stratified", np.array(label_list_stratified))
