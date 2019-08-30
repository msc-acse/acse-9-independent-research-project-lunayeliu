import numpy as np

class hex_8:
    def __init__(self, coordinates=None):
        if coordinates == None:
            self.nodal_coor = {1:[-1.0,-1.0,-1.0],
                               2:[1.0,-1.0,-1.0],
                               3:[1.0,1.0,-1.0],
                               4:[-1.0,1.0,-1.0],
                               5:[-1.0,-1.0,1.0],
                               6:[1.0,-1.0,1.0],
                               7:[1.0,1.0,1.0],
                               8:[-1.0,1.0,1.0]}
        else:
            self.nodal_coor = dict()
            for i in range(1, 9):
                self.nodal_coor[i] = coordinates[i-1]
