# ============================================================================
# This file is derived from the ST24R (STAR) MATLAB Screw Theory Toolbox.
#
# Original MATLAB code:
#   Copyright (C) 2001–2020
#   Dr. Jose M. Pardos-Gotor
#
# This Python adaptation:
#   Copyright (C) 2026
#   Vibhu Sharma
#
# License:
#   GNU Lesser General Public License v3.0 (LGPL-3.0)
#
# This file is distributed in compliance with the LGPL. You may modify and
# redistribute this file under the terms of the LGPL v3.0.
# ============================================================================

import numpy as np

np.set_printoptions(suppress=False, precision=5)

class SkewOps:
    """
    Class to convert between axis vector and skew symmetric matrix representations.
    """

    def __init__(self):
        pass
    

    @staticmethod
    def axis_to_skew(w:np.array):
        """
        Returns a skew symmetric matrix r 3x3 from the vector 3x1 w[a1;a2;a3;].
           |0  -a3  a2| 
        r =|a3   0 -a1|
           |-a2 a1   0|
        It is useful to transform a vector cross product to a matrix product, in the case of 3D vectors.
        Cross = a X b" is the same as "Cross = axis2skew(a) * b

        Args:
            w (np.array): A vector whose skew matrix is to be derived (3X1)

        Returns:
            r(np.array): A 3x3 skew symmetric matrix 
        """
        w = np.asarray(w).reshape(3,) 
        r = np.array([[0, -w[2], w[1]],[w[2], 0, -w[0]], [-w[1], w[0], 0]])
        return r
    

    @staticmethod
    def skew_to_axis(r:np.array) -> np.array:
        """
        Returns a vector 3x1 w[a1;a2;a3;] from a skew symmetric matrix r 3x3 .
           |0  -a3  a2| 
        r =|a3   0 -a1|
           |-a2 a1   0|

        Args:
            r (np.array): A 3x3 skew symmetric matrix

        Returns:
            np.array: A 3x1 vector
        """
        w = np.array([[r[2,1]], [r[0,2]], [r[1,0]]]).reshape(3,)
        return w
