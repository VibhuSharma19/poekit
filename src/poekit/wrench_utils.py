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

class WrenchUtils:
    """
    Utility functions for wrench conversions.
    """

    def __init__(self):
        pass


    @staticmethod
    def link_2_wrench(ForceAxis: np.ndarray, ForcePoint: np.ndarray, forceType: str) -> np.ndarray:
        """
        Convert force axis and point to wrench representation.

        Args:
            ForceAxis (np.ndarray): 3x1 vector representing force axis or moment axis
            ForcePoint (np.ndarray): 3x1 vector representing point of application
            forceType (str): "rot" for pure moment, "tra" for translational force

        Returns:
            np.ndarray: 6x1 wrench vector [f; T]
        """

        ForceAxis = np.asarray(ForceAxis, dtype=float).reshape(3,)
        ForcePoint = np.asarray(ForcePoint, dtype=float).reshape(3,)

        if forceType == "rot":
            return np.concatenate([np.zeros(3), ForceAxis])

        elif forceType == "tra":
            moment = -np.cross(ForceAxis, ForcePoint)
            return np.concatenate([ForceAxis, moment])

        else:
            return np.zeros(6)


    @staticmethod
    def wrench_axis(phi: np.ndarray) -> np.ndarray:
        """
        Obtain the line of action (point and direction) of a wrench.

        Args:
            phi (np.ndarray): 6x1 wrench vector [f; T]

        Returns:
            np.ndarray: 3x2 matrix where first column is a point on the line of action,
                        second column is the direction of the line of action
        """

        phi = np.asarray(phi, dtype=float).reshape(6,)
        f = phi[0:3]
        T = phi[3:6]

        if np.linalg.norm(f) < 1e-06:
            q = np.zeros(3)
            w = T / np.linalg.norm(T)
        else:
            f_norm2 = np.dot(f, f)
            q = np.cross(f, T) / f_norm2
            w = f / np.sqrt(f_norm2)

        return np.column_stack((q, w))


    @staticmethod
    def wrench_magnitude(xphi: np.ndarray) -> float:
        """
        Obtain the magnitude of a wrench.

        Args:
            xphi (np.ndarray): 6x1 wrench vector [f; T]

        Returns:
            float: magnitude of the wrench
        """

        xphi = np.asarray(xphi, dtype=float).reshape(6,)
        f = xphi[0:3]
        T = xphi[3:6]

        if np.linalg.norm(f) < 1e-06:
            return float(np.linalg.norm(T))
        else:
            return float(np.linalg.norm(f))


    @staticmethod
    def wrench_pitch(phi: np.ndarray) -> float:
        """
        Obtain the pitch of a wrench.

        Args:
            phi (np.ndarray): 6x1 wrench vector [f; T]

        Returns:
            float: pitch of the wrench
        """

        phi = np.asarray(phi, dtype=float).reshape(6,)
        f = phi[0:3]
        T = phi[3:6]

        f_norm2 = np.dot(f, f)

        if f_norm2 < 1e-6:
            return float("inf")

        return float(np.dot(f, T) / f_norm2)