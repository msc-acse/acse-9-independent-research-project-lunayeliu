import numpy as np

class tetra_10:
    def __init__(self, coordinates=None):
        """
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
        self.npe = 10
        self.N = np.zeros(self.npe)
        self.DNR = np.zeros(self.npe)
        self.DNS = np.zeros(self.npe)
        self.DNT = np.zeros(self.npe)

#         if coordinates == None:
#             self.nodal_coor = {0:[0.0, 0.0, 0.0],
#                                1:[1.0, 0.0, 0.0],
#                                2:[0.0, 1.0, 0.0],
#                                3:[0.0, 0.0, 1.0],
#                                4:[0.5, 0.0, 0.0],
#                                5:[0.5, 0.5, 0.0],
#                                6:[0.0, 0.5, 0.0],
#                                7:[0.0, 0.0, 0.5],
#                                8:[0.5, 0.0, 0.5],
#                                9:[0.0, 0.5, 0.5],
#                               }
#         else:
#             self.nodal_coor = dict()
#             for i in range(npe):
#                 self.nodal_coor[i] = coordinates[i]
        self.nodal_coor = {0:[0.0, 0.0, 0.0],
                           1:[1.0, 0.0, 0.0],
                           2:[0.0, 1.0, 0.0],
                           3:[0.0, 0.0, 1.0],
                           4:[0.5, 0.0, 0.0],
                           5:[0.5, 0.5, 0.0],
                           6:[0.0, 0.5, 0.0],
                           7:[0.0, 0.0, 0.5],
                           8:[0.5, 0.0, 0.5],
                           9:[0.0, 0.5, 0.5],
                          }
        if coordinates is not None:
            self.nodal_values = coordinates
        else:
            self.nodal_values = np.array([item for item in self.nodal_coor.values()])



    def shape_function(self, r, s, t):
        """note that L1 = 1 - r - s - t, L2 = r, L3 = s, L4 = t"""
        L2 = r
        L3 = s
        L4 = t

        L1 = 1.0 - L2 - L3 - L4
        self.N[0] = L1 * (2.0 * L1 - 1.0)
        self.N[1] = L2 * (2.0 * L2 - 1.0)
        self.N[2] = L3 * (2.0 * L3 - 1.0)
        self.N[3] = L4 * (2.0 * L4 - 1.0)
        self.N[4] = 4.0 * L1 * L2
        self.N[5] = 4.0 * L2 * L3
        self.N[6] = 4.0 * L3 * L1
        self.N[7] = 4.0 * L1 * L4
        self.N[8] = 4.0 * L2 * L4
        self.N[9] = 4.0 * L3 * L4
        return self.N

    def shape_derivative_r(self, r, s, t):
        """derivative to r"""
        nrst = 1. - r - s - t

        self.DNR[0] = 1. - 4. * nrst
        self.DNR[1] = -1. + 4. * r
        self.DNR[2] = 0.
        self.DNR[3] = 0.
        self.DNR[4] = -4. * r + 4. * nrst
        self.DNR[5] = 4. * s
        self.DNR[6] = -4. * s
        self.DNR[7] = -4. * t
        self.DNR[8] = 4. * t
        self.DNR[9] = 0.

        return self.DNR

    def shape_derivative_s(self, r, s, t):
        """derivative to s"""
        nrst = 1. - r - s - t

        self.DNS[0] = 1. - 4. * nrst
        self.DNS[1] = 0.
        self.DNS[2] = -1. + 4. * s
        self.DNS[3] = 0.
        self.DNS[4] = -4. * r
        self.DNS[5] = 4. * r
        self.DNS[6] = -4. * s + 4. * nrst
        self.DNS[7] = -4. * t
        self.DNS[8] = 0.
        self.DNS[9] = 4. * t

        return self.DNS

    def shape_derivative_t(self, r, s, t):
        """derivative to t"""

        nrst = 1. - r - s - t

        self.DNT[0] = 1. - 4. * nrst
        self.DNT[1] = 0.
        self.DNT[2] = 0.
        self.DNT[3] = -1. + 4. * t
        self.DNT[4] = -4. * r
        self.DNT[5] = 0.
        self.DNT[6] = -4. * s
        self.DNT[7] = 4. * nrst - 4. * t
        self.DNT[8] = 4. * r
        self.DNT[9] = 4. * s

        return self.DNT
