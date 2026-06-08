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
from .skewops import SkewOps

np.set_printoptions(suppress=False, precision=5)

class ExpMapping:
    """
    Class for Exponential Mapping operations in screw theory.
    Contains methods to compute the exponential map from axis-angle representation to rotation matrix
    and from twist-magnitude representation to homogeneous transformation matrix.
    """

    def __init__(self):
        pass

    @staticmethod
    def exp_axang(axang: np.array) -> np.array:
        """
        Exponential map from axis-angle representation to rotation matrix.
        
        Args:
            axang (np.array): A 4x1 vector representing the axis of rotation multiplied by the angle of rotation in radians.

        Returns:
            np.array: A 3x3 rotation matrix.
        """
        axang = np.asarray(axang, dtype=float).reshape(4,)

        w = axang[0:3]
        t = axang[3]

        Ws = SkewOps.axis_to_skew(w)

        R = np.eye(3) + Ws * np.sin(t) + (Ws @ Ws) * (1 - np.cos(t)) 
        return R
    

    @staticmethod
    def exp_screw(TwMag: np.array) -> np.array:
        """
        Exponential map from twist-magnitude representation to homogeneous transformation matrix.

        Args:
            TwMag (np.array): Twist and Magnitude vector (7x1) [v; W; theta]

        Returns:
            np.array: 4x4 Homogeneous transformation matrix.
        """

        TwMag = np.asarray(TwMag, dtype=float).reshape(7,)

        v = TwMag[0:3]
        w = TwMag[3:6]
        Th = TwMag[6]

        if np.linalg.norm(w) < 1e-6:
            r = np.eye(3)
            p = v* Th

        else:
            r = ExpMapping.exp_axang(np.hstack((w.T, Th)))
            p = (np.eye(3) - r) @ (np.cross(w,v))
        
        H = np.eye(4)
        H[0:3, 0:3] = r
        H[0:3, 3] = p
        return H