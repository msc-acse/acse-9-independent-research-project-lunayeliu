import sys
import scipy.linalg as sl
import numpy as np
import math
import random
import os
from hex8 import *



class FEM_Linear(object):
    """FEM class for linear elasticity"""
    def __init__(self, element):
        self.element = element
        self.std_element = hex_8()
        self.nu = 0.3
        self.E = 1e5
        self.G = 0.5*self.E/(1+self.nu)
        self.D = np.diag(np.array([1 for d in range(6)]))

    def poly_shape(self, coor):
        """Construct the shape matrix using polynominal basic function
        N_mat: 3 * (3 * 8)
        """
        N_mat = -1
        for pt in self.std_element.nodal_coor:
            N = 1
            for dim in range(3):
                N *= (1 + self.std_element.nodal_coor[pt][dim] * coor[dim])
            N_i = np.diag(np.array([N for d in range(3)]))
            # Handle the first matrix
            if type(N_mat) == int:
                N_mat = N_i
            else:
                # stack the matrix from each nodal
                N_mat = np.hstack((N_mat, N_i))
        return N_mat/8.0


    def shape_deriv(self, coor):
        """Calculate the shape function derivative for Jacobian
        """
        N_deriv_std = np.array([x for x in self.std_element.nodal_coor.values()])
        size = N_deriv_std.shape
        for i in range(size[0]):
            temp = N_deriv_std[i, :].copy()
            for j in range(size[1]):
                ind_1 = (j+4)%3
                ind_2 = (j+4+1)%3
                N_deriv_std[i, j] = temp[j]*1.0\
                                   *(1+temp[ind_1]*coor[ind_1])\
                                   *(1+temp[ind_2]*coor[ind_2])
        #print("N_deriv_std", N_deriv_std)
        N_deriv_std = N_deriv_std / 8.0
        return N_deriv_std

    def Jacobian(self, coor):
        """Calculate the Jacobian to do transformation
        between standard coordinates and reference coordinates"""

        ori_coor = np.array([x for x in self.element.nodal_coor.values()])
        N_deriv_std = self.shape_deriv(coor)
        return np.dot(np.transpose(N_deriv_std), ori_coor), N_deriv_std

    def poly_shape_deriv(self, coor, J=None,N_deriv_std=None):
        """Construct the shape derivative matrix using polynominal basic function
        N_mat_deriv: 6 * (3 * 8)
        The matrix structure is consistent to
        e_xx
        e_yy
        e_zz
        e_yz
        e_xz
        e_xy
        as the strain on right hand side
        """
        N_mat_deriv = -1
        if J is not None and N_deriv_std is not None:
            J = J
            N_deriv_std = N_deriv_std
        else:
            # Calculate Jacobian matrix and shape derivative matrix in standard coordinate
            J, N_deriv_std = self.Jacobian(coor)

        size = N_deriv_std.shape

        for i in range(size[0]):
            N_i = np.zeros((6, 3))

            # Calculate Jacobian matrix and shape derivative matrix in original coordinate(x,y,z)
            N_deriv_ori = sl.solve(J, N_deriv_std[i]) # direct method

            # Fill in the strain-displacement matrix
            for dim in range(3):
                val = N_deriv_ori[dim]
                if dim == 0:
                    N_i[0, 0] = val
                    N_i[4, 2] = val
                    N_i[5, 1] = val
                if dim == 1:
                    N_i[1, 1] = val
                    N_i[3, 2] = val
                    N_i[5, 0] = val
                if dim == 2:
                    N_i[2, 2] = val
                    N_i[3, 1] = val
                    N_i[4, 0] = val
            # Handle the first matrix
            if type(N_mat_deriv) == int:
                N_mat_deriv = N_i
            else:
                # stack the matrix from each nodal
                N_mat_deriv = np.hstack((N_mat_deriv, N_i))
        return N_mat_deriv

    def material_mat(self):
        """Construct the isotropic material matrix"""
        C_11 = self.E*(1-self.nu)/((1-2*self.nu)*(1+self.nu))
        C_12 = self.E*self.nu/((1-2*self.nu)*(1+self.nu))
        C_11_C12 = self.G
        D = np.diag(np.array([1 for d in range(6)]))

        D[0,0] = C_11
        D[1,1] = C_11
        D[2,2] = C_11
        D[3,3] =  self.G
        D[4,4] =  self.G
        D[5,5] =  self.G

        D[0,1] = C_12
        D[0,2] = C_12
        D[1,2] = C_12

        D[1,0] = C_12
        D[2,0] = C_12
        D[2,1] = C_12

        self.D = D
        return D

    def integrand(self, x, y, z):
        """Constrcut the BDB integrand for linear elasticity problem"""
        J, N_deriv_std = self.Jacobian([x,y,z])
        det_J=sl.det(J)
        Mat_B = self.poly_shape_deriv([x,y,z], J=J, N_deriv_std=N_deriv_std)
        BD = np.dot(np.transpose(Mat_B), self.D)
        return det_J*np.dot(BD, Mat_B)
