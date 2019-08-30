import numpy as np

class tetra_4:
    def __init__(self, coordinates=None):
        """4-nodes isoparametric tetrahedron element
        """
        self.npe = 4
        self.V = -1
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



    def volume(self):
        """Calculate the volume of CST element
        """
        coor_matrix = np.ones([self.npe, self.npe])

        for i in range(self.npe):
            coor_matrix[i,1:] = self.nodal_values[i,:]

        self.V = sl.det(coor_matrix)/6.0

        return self.V

    def _coefficient_calculation(self):
        """Calculate coefficient matrix for polynominal shape function
        """
        print(self.nodal_values)
        jj = 1
        for i in range(self.npe):
            j = (i+1)%self.npe
            k = (i+2)%self.npe
            l = (i+3)%self.npe


            # X values
            x_j = self.nodal_values[j,0]
            x_k = self.nodal_values[k,0]
            x_l = self.nodal_values[l,0]
            # y values
            y_j = self.nodal_values[j,1]
            y_k = self.nodal_values[k,1]
            y_l = self.nodal_values[l,1]
            # z values
            z_j = self.nodal_values[j,2]
            z_k = self.nodal_values[k,2]
            z_l = self.nodal_values[l,2]

            # compute a_i
            a_matrix = np.array([[x_j, y_j, z_j],
                                 [x_k, y_k, z_k],
                                 [x_l, y_l, z_l],
                                ])

            self.coefficient_matrix[i, 0] = jj*sl.det(a_matrix)


            # compute b_i
            b_matrix = np.array([[1.0, y_j, z_j],
                                 [1.0, y_k, z_k],
                                 [1.0, y_l, z_l],
                                ])

            self.coefficient_matrix[i, 1] = -jj*sl.det(b_matrix)


            # compute c_i
            c_matrix = np.array([[x_j, 1.0, z_j],
                                 [x_k, 1.0, z_k],
                                 [x_l, 1.0, z_l],
                                ])

            self.coefficient_matrix[i, 2] = -jj*sl.det(c_matrix)


            # compute d_i
            d_matrix = np.array([[x_j, y_j, 1.0],
                                 [x_k, y_k, 1.0],
                                 [x_l, y_l, 1.0],
                                ])

            self.coefficient_matrix[i, 3] = -jj*sl.det(d_matrix)

            jj = jj *(-1)
        self.coefficient_calculated = True

    def shape_function(self, x, y, z):
        """polynominal Shape function """
        if not self.coefficient_calculated:
            self._coefficient_calculation()
        self.volume()

        L = np.dot(self.coefficient_matrix, np.array([1.0,x,y,z]))/(6.0*self.V)

        for i,item in enumerate(L):
            self.N[i] = item

        return self.N

    def shape_derivate_r(self):
        """Shape function derivative to r"""
        if not self.coefficient_calculated:
            self._coefficient_calculation()
        self.DNR[0] = self.coefficient_matrix[0,1]
        self.DNR[1] = self.coefficient_matrix[1,1]
        self.DNR[2] = self.coefficient_matrix[2,1]
        self.DNR[3] = self.coefficient_matrix[3,1]

    def shape_derivate_s(self):
        """Shape function derivative to s"""
        if not self.coefficient_calculated:
            self._coefficient_calculation()
        self.DNS[0] = self.coefficient_matrix[0,2]
        self.DNS[1] = self.coefficient_matrix[1,2]
        self.DNS[2] = self.coefficient_matrix[2,2]
        self.DNS[3] = self.coefficient_matrix[3,2]

    def shape_derivate_t(self):
        """Shape function derivative to t"""
        if not self.coefficient_calculated:
            self._coefficient_calculation()
        self.DNT[0] = self.coefficient_matrix[0,3]
        self.DNT[1] = self.coefficient_matrix[1,3]
        self.DNT[2] = self.coefficient_matrix[2,3]
        self.DNT[3] = self.coefficient_matrix[3,3]

def strain_matrix_CST(element=tetra_4()):
    """Calculate strain matrix for linear elasticity"""
    element.volume()
    element._coefficient_calculation()
    print('coe',element.coefficient_matrix)
    B = np.zeros([6, 3*element.npe])

    for i in range(element.npe):
        # Column-1
        B[0,i*3] = element.coefficient_matrix[i,1]
        B[4,i*3] = element.coefficient_matrix[i,3]
        B[5,i*3] = element.coefficient_matrix[i,2]

        # Column-2
        B[1,i*3+1] = element.coefficient_matrix[i,2]
        B[3,i*3+1] = element.coefficient_matrix[i,3]
        B[5,i*3+1] = element.coefficient_matrix[i,1]

        # Column-3
        B[2,i*3+2] = element.coefficient_matrix[i,3]
        B[3,i*3+2] = element.coefficient_matrix[i,2]
        B[4,i*3+2] = element.coefficient_matrix[i,1]

    return B/(6*element.V)

def stiffness_matrix_CST(element=tetra_4()):
    """Calculate stiffness matrix for linear elasticity"""
    element.volume()
    B = strain_matrix_CST(element)
    D = material()
    print('B')
    print(B)
    print('V',element.V)
    return element.V * np.dot(np.dot(np.transpose(B),D),B)
