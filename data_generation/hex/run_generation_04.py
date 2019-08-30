import numpy as np
import os
import time
import datetime
from data_generation_hex_8 import *
sys.path.extend(['../../utils','../../integration_module','../../FEM_module','../../utils','../../elements'])


#d = [0.1 * n for n in range(1,6)]
d = [0.4]
a = -0.5
b = 1.0

N = 2000
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
print(coordinate_list_stratified)
print(label_list_stratified)


# Four dimensions data structure
# 1 - different d
# 2 - how many samples
# 3 - nodes
# 4 - x, y, z coordinates

undo = 0
for j in range(len(d)):
    for i in range(N):
        #r = random.random() * (b - a) + a
        coor  = generate_coordinate(random_func, d[j])
        plane_set = plane_generate(coor)

        while plane_judge(plane_set):
            #r = random.random() * (b - a) + a
            coor  = generate_coordinate(random_func, d[j])
            plane_set = plane_generate(coor)
            undo += 1
        # Stratified features list 4-dimension: 5 * N * 8 * 3
        coordinate_list_stratified[j].append(coor)
        # All features list 3-dimension:  (5*N) * 8 * 3
        coordinate_list_all.append(coor)

        print(undo)
        optimal_num = find_optimal_number(coor, tol=tol, n_max = n_max)
        # Stratified label list 2-dimension: 5 * N
        label_list_stratified[j].append(optimal_num)
        # All label list 1-dimension:  (5*N)
        label_list_all.append(optimal_num)

    print("d equals ", d[j]," finished.")


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
output_path = "./train_label"+st
os.mkdir(output_path)
np.save(output_path + "/train_features", np.array(coordinate_list_all))
np.save(output_path + "/labels", np.array(label_list_all))
np.save(output_path + "/train_features_stratified", np.array(coordinate_list_stratified))
np.save(output_path + "/labels_stratified", np.array(label_list_stratified))
