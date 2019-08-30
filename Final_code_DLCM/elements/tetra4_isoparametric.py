import numpy as np

class tetra_4_isoparametric:
    def __init__(self, coordinates=None):
    """4-nodes isoparametric tetrahedron element
    """
        self.npe = 4  #number of vertices
        self.V = -1   # volume of the element
        self.N = np.zeros(self.npe)
        self.DNR = np.zeros(self.npe)
        self.DNS = np.zeros(self.npe)
        self.DNT = np.zeros(self.npe)
        self.coefficient_matrix = np.zeros([self.npe, self.npe])
        self.coefficient_calculated = False

        if coordinates == None:
            self.nodal_coor = {0:[0.0, 0.0, 0.0],
                               1:[1.0, 0.0, 0.0],
                               2:[0.0, 1.0, 0.0],
                               3:[0.0, 0.0, 2.5],
                              }
        else:
            self.nodal_coor = dict()
            for i in range(self.npe):
                self.nodal_coor[i] = coordinates[i]

        self.nodal_values = np.array([item for item in self.nodal_coor.values()])


    def shape_function(self, r, s, t):
        """note that L1 = 1 - r - s - t, L2 = r, L3 = s, L4 = t"""

        self.N[0] = 1.0 - r - s - t
        self.N[1] = r
        self.N[2] = s
        self.N[3] = t

        return self.N

    def shape_derivative_r(self, r, s, t):
        """Shape function derivative to r"""
        self.DNR[0] = -1.0
        self.DNR[1] = 1.0
        self.DNR[2] = 0.0
        self.DNR[3] = 0.0

        return self.DNR

    def shape_derivative_s(self, r, s, t):
        """Shape function derivative to s"""
        self.DNS[0] = -1.0
        self.DNS[1] = 0.0
        self.DNS[2] = 1.0
        self.DNS[3] = 0.0

        return self.DNS

    def shape_derivative_t(self, r, s, t):
        """Shape function derivative to t"""
        self.DNT[0] = -1.0
        self.DNT[1] = 0.0
        self.DNT[2] = 0.0
        self.DNT[3] = 1.0

        return self.DNT
