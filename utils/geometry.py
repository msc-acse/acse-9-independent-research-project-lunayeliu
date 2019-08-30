import numpy as np
import scipy as sp
import math


def normal_func_parameter(plane):
    """
    Calculate coefficients for plane function

       pt1 = [x1, y1, z1]
       pt2 = [x2, y2, z2]
       pt3 = [x3, y3, z3]

       vector1 = [x2 - x1, y2 - y1, z2 - z1]
       vector2 = [x3 - x1, y3 - y1, z3 - z1]
    """
    pt1 = plane[0]
    pt2 = plane[1]
    pt3 = plane[2]
    vector1 = [pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2]]
    vector2 = [pt3[0] - pt1[0], pt3[1] - pt1[1], pt3[2] - pt1[2]]

    cross_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1],\
                     -1 * (vector1[0] * vector2[2] - vector1[2] * vector2[0]),\
                     vector1[0] * vector2[1] - vector1[1] * vector2[0]]
    a = cross_product[0]
    b = cross_product[1]
    c = cross_product[2]
    d = - (cross_product[0] * pt1[0] + cross_product[1] * pt1[1] + cross_product[2] * pt1[2])
    return a, b, c, d


def angle_plane(a1, b1, c1, a2, b2, c2):
    """
    Function to calculate angle between adjacent faces using plane coefficients
    """
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    #print("Angle is ", A, " degree")
    return A


    # -------------------------
    # plane1 [pt1, pt2, pt5]
    # plane2 [pt2, pt3, pt6]
    # plane3 [pt3, pt4, pt7]
    # plane4 [pt1, pt4, pt5]
    # plane5 [pt5, pt6, pt8]
    # plane6 [pt1, pt2, pt3]
    # -------------------------
    # 1-5, 1-2, 1-4, 1-6
    # 3-2, 3-5, 3-6, 3-4
    # 2-5, 2-6
    # 4-5, 4-6
    # -------------------------

    # Calculate angles between planes
def calculate_angle(plane1, plane2):
    """Calculate the angle between two planes
    """
    para1 = normal_func_parameter(plane1)
    para2 = normal_func_parameter(plane2)
    angle = angle_plane(para1[0], para1[1], para1[2], para2[0], para2[1], para2[2])

    return angle


def plane_judge(plane_set, range=(20, 160)):
    """Judge whether the angle between two plane is inside the range
    """
    flag = False
    for plane_pair in plane_set:
        ang = calculate_angle(plane_pair[0], plane_pair[1])
        if ang <= range[0] or ang >= range[1]:
            flag = True
            break
    return flag
