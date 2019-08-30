from error_estimation import *
from Integration import *
from FEM_Linear_hex import *
from geometry import *
import random


def random_func(a = -0.5, b=1.0):
    """Generate random factor r among [a, b]"""
    return random.random() * (b - a) + a

def generate_coordinate(random_func, d):
    """Generate different r for each node"""
    r = [random_func() for i in range(7)]
    coordinate = [[0.0, 0.0, 0.0],\
              [1.0 + r[0]*d, 0.0, 0.0],\
              [1.0 + r[1]*d, 1.0 + r[1]*d, r[1]*d],\
              [r[2]*d, 1.0 + r[2]*d, 0.0],\
              [r[3]*d, r[3]*d, 1.0 + r[3]*d],\
              [1.0 + r[4]*d, r[4]*d, 1.0 + r[4]*d],\
              [1.0 + r[5]*d, 1.0 + r[5]*d, 1.0 + r[5]*d],\
              [r[6]*d, 1.0 + r[6]*d, 1.0 + r[6]*d]]
    return coordinate

def plane_generate(coor):
    """
    Generate the adjacent plane pair set for hexahedron 10

    plane1 [pt1, pt2, pt5]
    plane2 [pt2, pt3, pt6]
    plane3 [pt3, pt4, pt7]
    plane4 [pt1, pt4, pt5]
    plane5 [pt5, pt6, pt8]
    plane6 [pt1, pt2, pt3]
    """
    plane1 = [coor[0], coor[1], coor[4]]
    plane2 = [coor[1], coor[2], coor[5]]
    plane3 = [coor[2], coor[3], coor[6]]
    plane4 = [coor[0], coor[3], coor[4]]
    plane5 = [coor[4], coor[5], coor[7]]
    plane6 = [coor[0], coor[1], coor[2]]

    plane_set = [[plane1, plane5],[plane1, plane2],[plane1, plane4],[plane1, plane6],\
                 [plane3, plane2],[plane3, plane5],[plane3, plane6],[plane3, plane4],\
                 [plane2, plane5],[plane2, plane6],[plane4, plane5],[plane4, plane6]]

    return plane_set


def find_optimal_number(element, tol=1e-3, n_max=20):
    """Function to find the number of integration points which firstly reach the requirement
    input
    element: The element to use
    tol: error tolerance
    n_max: the number of integration points which represent the exact value
    """
    integrand_inner = FEM_Linear(element).integrand

    res_ref = TensorQuad(integrand_inner, [[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],n=n_max,w=None).integrate()
    for n in range(1, n_max):
        res = TensorQuad(integrand_inner, [[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]],n=n,w=None).integrate()
        err = error_estimation(res, res_ref)
        print("error", err)
        if err <= tol:
            return n
    print("No optimal number!")
    return 0
