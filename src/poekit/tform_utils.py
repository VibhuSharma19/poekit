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
from .rotation_utils import RotationUtils

np.set_printoptions(suppress=False, precision=5)

class TformUtils:
    """
    Utility functions for transformation matrix conversions.
    """

    def __init__(self):
        pass


    @staticmethod
    def tform_2_adjoint(tform: np.array) -> np.array:
        """
        Convert a transformation matrix to adjoint representation.

        Args:
            tform (np.array): 4x4 transformation matrix
        
        Returns:
            np.array: 6x6 adjoint matrix
        """

        tform = np.asarray(tform, dtype=float).reshape(4,4)

        r = tform[0:3, 0:3]
        p = tform[0:3, 3].reshape(3,1)

        tform = np.asarray(tform, dtype=float)
        R = tform[0:3, 0:3]
        p = tform[0:3, 3]

        Ad = np.zeros((6, 6), dtype=float)
        Ad[0:3, 0:3] = R
        Ad[3:6, 3:6] = R
        Ad[0:3, 3:6] = SkewOps.axis_to_skew(p) @ R

        return Ad
    

    @staticmethod
    def tform_2_spvec(tform: np.array) -> np.array:
        """
        Convert a transformation matrix to spatial vector representation.

        Args:
            tform (np.array): 4x4 transformation matrix
        
        Returns:
            np.array: 6x1 spatial vector
        """

        tform = np.asarray(tform, dtype=float).reshape(4,4)

        R = tform[0:3, 0:3]
        p = tform[0:3, 3]

        axis, angle = RotationUtils.rotm_2_axang(R)   
        wtheta = axis * angle

        spvec = np.zeros(6)
        spvec[0:3] = wtheta
        spvec[3:6] = p

        return spvec
    

    @staticmethod
    def tform_2_twist(tform: np.array) -> np.array:
        """
        Convert a transformation matrix to twist representation.

        Args:
            tform (np.array): 4x4 transformation matrix
        
        Returns:
            np.array: 6x1 twist vector
        """

        tform = np.asarray(tform, dtype=float).reshape(4,4)

        r = tform[0:3, 0:3]
        p = tform[0:3, 3]
        
        w = SkewOps.skew_to_axis(r)

        tw = np.concatenate([p, w.reshape(3,)]).reshape(6,1)

        return tw
    
    
    @staticmethod
    def tform_2_xpluc(tform: np.array) -> np.array:
        """
        Convert a transformation matrix to Plucker coordinates representation.
        
        Args:
            tform (np.array): 4x4 transformation matrix
        
        Returns:
            np.array: 6x6 Plucker coordinates matrix
        """

        H = np.asarray(tform, dtype=float)

        R = H[0:3, 0:3]
        p = H[0:3, 3]

        X = np.zeros((6, 6), dtype=float)
        X[0:3, 0:3] = R
        X[3:6, 3:6] = R
        X[3:6, 0:3] = SkewOps.axis_to_skew(p) @ R

        return X
    

    @staticmethod
    def trvP_2_tform(p: np.ndarray) -> np.ndarray:
        """
        Convert a translation vector to a transformation matrix.

        Args:
            p (np.ndarray): 3x1 translation vector

        Returns:
            np.ndarray: 4x4 transformation matrix
        """

        p = np.asarray(p, dtype=float).reshape(3,)
        Hp = np.eye(4)
        Hp[0:3, 3] = p
        return Hp


    @staticmethod
    def trvX_2_tform(p: float) -> np.ndarray:
        """
        Convert a translation along the X-axis to a transformation matrix.
        
        Args:
            p (float): translation distance along the X-axis
        
        Returns:
            np.ndarray: A 4x4 transformation matrix
        """

        Hxp = np.eye(4)
        Hxp[0, 3] = float(p)
        return Hxp


    @staticmethod
    def trvY_2_tform(p: float) -> np.ndarray:
        """
        Convert a translation along the Y-axis to a transformation matrix.

        Args:
            p (float): translation distance along the Y-axis

        Returns:
            np.ndarray: A 4x4 transformation matrix
        """

        Hyp = np.eye(4)
        Hyp[1, 3] = float(p)
        return Hyp


    @staticmethod
    def trvZ_2_tform(p: float) -> np.ndarray:
        """
        Convert a translation along the Z-axis to a transformation matrix.

        Args:
            p (float): translation distance along the Z-axis

        Returns:
            np.ndarray: A 4x4 transformation matrix
        """

        Hzp = np.eye(4)
        Hzp[2, 3] = float(p)
        return Hzp

    
    @staticmethod
    def xpluc_2_tform(X: np.ndarray) -> np.ndarray:
        """
        Convert Plucker coordinates representation to transformation matrix.

        Args:
            X (np.ndarray): 6x6 Plucker coordinates matrix

        Returns:
            np.ndarray: 4x4 transformation matrix
        """

        X = np.asarray(X, dtype=float).reshape(6,6)

        H = np.zeros((4, 4), dtype=float)
        H[3, 3] = 1.0

        R = X[0:3, 0:3]
        skew_r_Rt = X[4:6, 0:3] @ R.T   
        p = SkewOps.skew_to_axis(skew_r_Rt).reshape(3,)

        H[0:3, 0:3] = R
        H[0:3, 3] = p

        return H