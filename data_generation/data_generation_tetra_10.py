import sys
sys.path.extend(['../utils','../integration_module','../FEM_module','../utils'])
from error_estimation import *
from Integration import *
from FEM_Linear_tetra_10 import *
from geometry import *
import random

def random_func(a = -0.5, b=1.0):
    """Generate random factor r among [a, b]"""
    return random.random() * (b - a) + a

def plane_generate_tetra_10(coor):
    """
    Generate the adjacent plane pair set for tetra 10

    plane1 [pt0, pt1, pt2]
    plane2 [pt0, pt1, pt3]
    plane3 [pt0, pt2, pt3]
    plane4 [pt1, pt2, pt3]

    """
    plane1 = [coor[0], coor[1], coor[2]]
    plane2 = [coor[0], coor[1], coor[3]]
    plane3 = [coor[0], coor[2], coor[3]]
    plane4 = [coor[1], coor[2], coor[3]]

    plane_set = [[plane1, plane2],[plane1, plane3],[plane1, plane4],\
                 [plane2, plane3],[plane2, plane4],[plane3, plane4],\
                 ]

    return plane_set

def generate_coordinate_tetra_10(random_func, d):
    """Generate different r for each node"""
    r = [random_func() for i in range(9)]

    # mid-node should get a positive value
    r[3] = random_func(a=0.0)
    r[4] = random_func(a=0.0)
    r[5] = random_func(a=0.0)
    r[6] = random_func(a=0.0)
    r[7] = random_func(a=0.0)
    r[8] = random_func(a=0.0)

    coordinate = [[0.0, 0.0, 0.0],\
              [1.0 + r[0]*d, 0.0, 0.0],\
              [0.0, 1.0 + r[1]*d, 0.0],\
              [0.0, 0.0, 1.0 + r[2]*d],\
              [0.5 + r[3]*d, -r[3]*d, 0.0],\
              [0.5 + r[4]*d, 0.5+r[4]*d, 0.0],\
              [-r[5]*d, 0.5 + r[5]*d, 0.0],\
              [-r[6]*d, -r[6]*d, 0.5+r[6]*d],\
              [0.5+r[7]*d, r[7]*d, 0.5],\
              [r[8]*d, 0.5+r[8]*d, 0.5]
              ]
    return coordinate


def central_point_judge(coor):
    """Judge whether central points in the tetrahedron
    are proper placed
                      ^ t
                  |
                  |

                3 o
                  | \
               E3 |   \ E5
                7 o     o 9
                  |      \
           8 o    |        \
          E4    0 o -- o --- o 2  --> s
              E0 /   E2 6
               /
           4 o     o 5 E1
            /
          /
        o 1
        r
    """
    flag = False
    #print(coor)
    # Point 4 is between pt0 and pt1
    if line_function_judge(coor[4][:2], coor[0][:2], coor[1][:2]):
        #print('pt4 pass')
        pass
    else:
        #print("pt4 false")
        return True
    # Point 5 is between pt1 and pt2
    if line_function_judge(coor[5][:2], coor[1][:2], coor[2][:2]):
        #print('pt5 pass')
        pass
    else:
        #print("pt5 false")
        return True
    # Point 6 is between pt0 and pt2
    if line_function_judge(coor[6][:2], coor[0][:2], coor[2][:2]):
        #print('pt6 pass')
        pass
    else:
        #print("pt6 false")
        return True
    # Point 7 is between pt0 and pt3
    if line_function_judge(coor[7][1:], coor[0][1:], coor[3][1:]):
        #print('pt7 pass')
        pass
    else:
        #print("pt7 false")
        return True
    # Point 8 is between pt1 and pt3
    if line_function_judge((coor[8][0],coor[8][2]), (coor[1][0],coor[1][2]), (coor[3][0],coor[3][2])):
        #print('pt8 pass')
        pass
    else:
        #print("pt8 false")
        return True
    # Point 9 is between pt2 and pt3
    if line_function_judge(coor[9][1:], coor[2][1:], coor[3][1:]):
        #print('pt9 pass')
        pass
    else:
        #print("pt9 false")
        return True
    return flag

def line_function_judge(pt_target, pt1, pt2):
    """Line function, two points version
    (y_2-y_1)x-(x_2-x_1)y = (y_2-y_1)x_1-(x_2-x_1)y_1
    Exclue RHS =0
    if LHS/RHS > 1
    condition accepted, then return True

    """
    if (pt2[1]-pt1[1])==0 and (pt2[0]-pt1[0])==0:
        print("pt2, pt1, Overlap points!")
        return False

    if (pt2[1]-pt_target[1])==0 and (pt2[0]-pt_target[0])==0:
        print("pt_target, pt2, Overlap points!")
        return False

    if (pt_target[1]-pt1[1])==0 and (pt_target[0]-pt1[0])==0:
        print("pt_target, pt1, Overlap points!")
        return False

    RHS = (pt2[1]-pt1[1])*pt1[0] - (pt2[0]-pt1[0])*pt1[1]

    # Handle the line pass origin condition
    if RHS == 0:
        if pt1[0]==0 and pt1[1]==0:
            if pt_target[0] <= 0 or pt_target[1] <= 0:
                return True

        if pt2[0]==0 and pt2[1]==0:
            if pt_target[0] <= 0 or pt_target[1] <= 0:
                return True
        #print("RHS")
        return False


    LHS = (pt2[1]-pt1[1])*pt_target[0] - (pt2[0]-pt1[0])*pt_target[1]

    if LHS/RHS >= 1:
        #print(LHS/RHS)
        return True
    else:
        return False

def find_optimal_number(element, tol=1e-3, n_max=20):
    """Find the optimal integration number of a element"""

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
