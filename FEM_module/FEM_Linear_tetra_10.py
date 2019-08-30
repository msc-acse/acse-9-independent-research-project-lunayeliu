import sys
import numpy as np
import scipy as sp
import math
import scipy.linalg as sl
from tetra10 import *


def N_matrix(coor, element=tetra_10()):
    """Construct the N matrix for shape function

    input
    coor: coordinate values for the element

    return
    N matrix
    """
    N = np.zeros([3,element.npe*3])
    N_value_list = element.shape_function(coor[0], coor[1], coor[2])

    for i, N_value in enumerate(N_value_list):
        N[0,i],N[1,i+1],N[2,i+2] = N_value, N_value, N_value
    return N

def N_derivative_local(coor, element=tetra_10()):
    """Construct the N derivative matrix for shape function

    input
    coor: coordinate values for the element

    return
    N derivative matrix

    """
    N_derivative = np.zeros([3,element.npe])

    DNR = element.shape_derivative_r(coor[0], coor[1], coor[2])
    DNS = element.shape_derivative_s(coor[0], coor[1], coor[2])
    DNT = element.shape_derivative_t(coor[0], coor[1], coor[2])

    for i in range(element.npe):
        N_derivative[0, i] = DNR[i]
        N_derivative[1, i] = DNS[i]
        N_derivative[2, i] = DNT[i]

    return N_derivative

def Jacobian(coor, element = tetra_10()):
    """Cpmpute the jacobian matrix
    from x,y,z physic coordinate to standard r,s,t coordinate
     """
    N_derivative = N_derivative_local(coor, element)
    J = np.dot(N_derivative, element.nodal_values)
    return J

def N_derivative_global(coor, element=tetra_10()):
    """Compute the N derivative in global coordinate system"""
    N_derivative_global = np.zeros([3,element.npe])
    J = Jacobian(coor, element)
    N_derivative = N_derivative_local(coor, element)
    for i in range(element.npe):
        N_derivative_global[:,i] = sl.solve(J,N_derivative[:,i])
    return N_derivative_global

def Strain_matrix_global(coor,element=tetra_10()):
    """Compute the strain matrix in global coordinates"""
    B = np.zeros([6, 3*element.npe])
    N_derivative_global_value = N_derivative_global(coor,element)

    for i in range(element.npe):
        # Column-1
        B[0,i*3] = N_derivative_global_value[0,i]
        B[4,i*3] = N_derivative_global_value[2,i]
        B[5,i*3] = N_derivative_global_value[1,i]

        # Column-2
        B[1,i*3+1] = N_derivative_global_value[1,i]
        B[3,i*3+1] = N_derivative_global_value[2,i]
        B[5,i*3+1] = N_derivative_global_value[0,i]

        # Column-3
        B[2,i*3+2] = N_derivative_global_value[2,i]
        B[3,i*3+2] = N_derivative_global_value[1,i]
        B[4,i*3+2] = N_derivative_global_value[0,i]

    return B

def material():
    """Construct the material matrix
    Here a identity matrix is used for simplicity
    """
    return np.diag(np.array([1 for d in range(6)]))

def Integrand(p,q,k):
    """
    The integrand of stiffness matrix in hextrahedron coordinate
    """
    r,s,t,jac = PQKtoRST([p,q,k]) # From base tet element to base hex
    J = Jacobian([r,s,t], element=test_element) # From original tet to base tet element
    B = Strain_matrix_global([r,s,t], element=test_element)
    D = material()
    return jac*sl.det(J)*np.dot(np.dot(np.transpose(B),D),B)

def PQKtoRST(coor_pqk):
    """From pqk base hextrahedron coordinate
    to rst base tetrahedron coordinate"""

    p,q,k = coor_pqk[0], coor_pqk[1], coor_pqk[2]
    r = (1+p)/2.0
    s = (1-p)*(1+q)/4.0
    t = (1-p)*(1-q)*(1+k)/8.0
    jac = (1-p)*(1-p)*(1-q)/64.0

    return r,s,t,jac

class FEM_Linear:
    """The base class for linear elasticity problem using tetrahedron element"""
    def __init__(self, element):
        self.element = element

    def integrand(self, p,q,k):
        r,s,t,jac = PQKtoRST([p,q,k]) # From base tet element to base hex
        J = Jacobian([r,s,t], element=self.element) # From original tet to base tet element
        B = Strain_matrix_global([r,s,t], element=self.element)
        D = material()
        return jac*sl.det(J)*np.dot(np.dot(np.transpose(B),D),B)
