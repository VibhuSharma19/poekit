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
from .tform_utils import TformUtils
from .rotation_utils import RotationUtils

np.set_printoptions( suppress=False, precision=5)

class TwistUtils:
    """
    Utility functions for twist representation conversions.
    """

    def __init__(self):
        pass


    @staticmethod
    def joint_2_twist(axis: np.ndarray, point: np.ndarray, joint_type: str) -> np.ndarray:
        """
        Convert joint axis and a point on that axis to a 6-vector twist [v; w].
    
        - axis:  shape (3,) or (3,1)
        - point: shape (3,) or (3,1)
        - joint_type: 'rot' or 'prism'
        Returns: numpy array shape (6,)
        """
    
        w = np.asarray(axis, dtype=float).reshape(3,)
        q = np.asarray(point, dtype=float).reshape(3,)
    
        if joint_type == 'rot':
            # revolute: v = -w x q
            v = -np.cross(w, q)             # shape (3,)
            tw = np.hstack((v, w))          # shape (6,)
        elif joint_type == 'prism':
            tw = np.hstack((w, np.zeros(3,)))
        else:
            tw = np.zeros(6,)
    
        return tw
    

    @staticmethod
    def motion_2_twist(h: np.array) -> np.array:
        """
        Convert motion vector to twist representation.

        Args:
            h (np.ndarray): 4x4 homogeneous transformation matrix
        
        Returns:
            np.array: 6x1 twist vector
        """

        h = np.asarray(h, dtype=float).reshape(4,4)

        r = h[0:3, 0:3]
        p = h[0:3, 3]
        th = RotationUtils.rotation_angle(r)
        w = RotationUtils.rotation_axis(r, th).reshape(3,)

        if np.isclose(th, 0):
            t = np.linalg.norm(p)

            if np.isclose(t, 0):
                v = np.array([0,0,0])
            else:   
                v = p / t
            
        else:
            ws = SkewOps.axis_to_skew(w)
            A = (np.eye(3)-r)@ws + w @ np.transpose(w) *th
            v = np.linalg.solve(A, p)

        return np.concatenate((v, w)).reshape(6,1)
    

    @staticmethod
    def twist_2_joint(tw: np.array) -> tuple:
        """
        Convert twist representation to joint axis and point.

        Args:
            tw (np.array): 6x1 twist vector
        Returns:
            tuple: (axis, point, joint_type)
        """

        tw = np.asarray(tw, dtype=float).reshape(6,)

        v = tw[0:3]
        w = tw[3:6]

        if np.linalg.norm(w) < 1e-06:
            axis = v / np.linalg.norm(v)
            q = np.array([0,0,0])
            joint_type = 'tra'
        else:
            axis = w / np.linalg.norm(w)
            q = np.cross(w,v) / np.linalg.norm(w)**2
            joint_type = 'rot'

        return (axis, q, joint_type)
    

    @staticmethod
    def twist_2_tform(tw: np.array) -> np.array:
        """
        Convert twist representation to transformation matrix.

        Args:
            tw (np.array): 6x1 twist vector
        Returns:
            np.array: 4x4 transformation matrix
        """

        tw = np.asarray(tw, dtype=float).reshape(6,1)
        v = tw[0:3].reshape(3,)
        w = tw[3:6].reshape(3,)

        ws = SkewOps.axis_to_skew(w)

        tform = np.array([[ws[0,0], ws[0,1], ws[0,2], v[0]], 
                          [ws[1,0], ws[1,1], ws[1,2], v[1]], 
                          [ws[2,0], ws[2,1], ws[2,2], v[2]], 
                          [   0   ,    0   ,    0   ,  1 ]])
        return tform
    

    @staticmethod
    def twist_axis(tw: np.array) -> np.array:
        """
        Extract the axis of rotation from a twist representation.

        Args:
            tw (np.array): 6x1 twist vector
        Returns:
            np.array: A 3x1 vector representing the axis of rotation
        """

        tw = np.asarray(tw, dtype=float).reshape(6,)

        v = tw[0:3]
        w = tw[3:6]

        if np.linalg.norm(w) < 1e-06:
            axis = v / np.linalg.norm(v)
            q = np.array([0,0,0])
        else:
            axis = w / np.linalg.norm(w)
            q = np.cross(w,v) / np.linalg.norm(w)**2

        return np.column_stack((q, w))


    @staticmethod
    def twist_bracket(tw1: np.array, tw2: np.array) -> np.array:
        """
        Compute the Lie bracket of two twists.

        Args:
            tw1 (np.array): 6x1 twist vector
            tw2 (np.array): 6x1 twist vector
        Returns:
            np.array: 6x1 twist vector representing the Lie bracket [tw1, tw2]
        """

        tw1 = np.asarray(tw1, dtype=float).reshape(6,)
        tw2 = np.asarray(tw2, dtype=float).reshape(6,)

        E1 = TwistUtils.twist_2_tform(tw1)
        E2 = TwistUtils.twist_2_tform(tw2)

        xi = TformUtils.tform_2_twist(E1 @ E2 - E2 @ E1)

        return xi
    

    @staticmethod
    def twist_magnitude(tw: np.array) -> float:
        """
        Compute the magnitude of a twist.

        Args:
            tw (np.array): 6x1 twist vector
        Returns:
            float: magnitude of the twist
        """

        tw = np.asarray(tw, dtype=float).reshape(6,)

        v = tw[0:3]
        w = tw[3:6]

        if np.linalg.norm(w) < 1e-06:
            mag = np.linalg.norm(v)
        else:
            mag = np.linalg.norm(w)

        return mag
    

    @staticmethod
    def twist_pitch(tw: np.array) -> float:
        """
        Compute the pitch of a twist.

        Args:
            tw (np.array): 6x1 twist vector
        Returns:
            float: pitch of the twist
        """

        tw = np.asarray(tw, dtype=float).reshape(6,)

        v = tw[0:3]
        w = tw[3:6]

        if np.linalg.norm(w) < 1e-06:
            pitch = np.inf
        else:
            pitch = np.dot(w, v) / np.linalg.norm(w)**2

        return pitch 