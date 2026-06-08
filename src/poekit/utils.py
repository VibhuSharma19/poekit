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
from .tform_utils import TformUtils
from .expmapping import ExpMapping     
from .rotation_utils import RotationUtils

np.set_printoptions(suppress=False, precision=5)

class Utils:
    """
    General utility functions for screw theory operations.
    """

    def __init__(self):
        pass


    @staticmethod
    def jointmag_2_limits(TheTheory: float, LimitPos: float, LimitNeg: float) -> float:
        """
        Saturate theoretical joint magnitude within physical limits.

        Args:
            TheTheory (float): theoretical joint magnitude
            LimitPos (float): positive joint limit
            LimitNeg (float): negative joint limit

        Returns:
            float: saturated joint magnitude
        """

        vals = np.array([LimitPos, TheTheory, LimitNeg], dtype=float)
        vals_sorted = np.sort(vals)
        return vals_sorted[1]  


    @staticmethod
    def spavec_2_crf(v: np.ndarray) -> np.ndarray:
        """
        Convert a spatial vector to cross product Force representation.

        Args:
            v (np.ndarray): 6x1 spatial vector

        Returns:
            np.ndarray: 6x6 cross product Force matrix
        """

        v = np.asarray(v, dtype=float).reshape(6,)
        v1, v2, v3, v4, v5, v6 = v

        crf = np.array([
            [ 0,   -v3,   v2,    0,   -v6,   v5],
            [ v3,    0,  -v1,   v6,     0,  -v4],
            [-v2,   v1,    0,  -v5,   v4,     0],
            [ 0,     0,    0,    0,   -v3,   v2],
            [ 0,     0,    0,   v3,     0,  -v1],
            [ 0,     0,    0,  -v2,   v1,     0]
        ], dtype=float)

        return crf


    @staticmethod
    def spavec_2_crm(v: np.ndarray) -> np.ndarray:
        """
        Convert a spatial vector to cross product Motion representation.

        Args:
            v (np.ndarray): 6x1 spatial vector

        Returns:
            np.ndarray: 6x6 cross product Motion matrix
        """

        v = np.asarray(v, dtype=float).reshape(6,)
        v1, v2, v3, v4, v5, v6 = v

        crm = np.array([
            [ 0,   -v3,   v2,    0,    0,    0],
            [ v3,    0,  -v1,    0,    0,    0],
            [-v2,   v1,    0,    0,    0,    0],
            [ 0,   -v6,   v5,    0,   -v3,   v2],
            [ v6,    0,  -v4,   v3,    0,   -v1],
            [-v5,   v4,    0,   -v2,   v1,    0]
        ], dtype=float)

        return crm


    @staticmethod
    def Aij_2_adjoint(i: float, j: float, TwMag: np.array) -> np.ndarray:
        """
        Computes ADJOINT TRANSFORMATION for a list of twists-mag. Use in SE(3).
        Notation useful for Link Jacobian (mobile). Notation useful for Christofell Symbols. Use in SE(3).

        Args:
            i (float): joint number
            j (float): joint number
            TwMag (np.array): 7xN matrix of N twists-magnitudes

        Returns:
            np.ndarray: 6x6 adjoint transformation matrix
        """

        AI = np.eye(6)
        AZ = np.zeros((6, 6))

        if i < j:
            return AZ
        elif i == j:
            return AI
        else:
            PoE = ExpMapping.exp_screw(TwMag[:, j])       
            for k in range(j + 1, i):
                PoE = PoE @ ExpMapping.exp_screw(TwMag[:, k])
            return np.linalg.solve(TformUtils.tform_2_adjoint(PoE), AI)


    @staticmethod
    def intersect_lines3D(Axes: np.ndarray, Points: np.ndarray) -> np.ndarray:
        """
        Find the intersection point of two 3D lines defined by Plucker coordinates.

        Args:
            Axes (np.ndarray): 3x2 matrix of direction vectors of the lines
            Points (np.ndarray): 3x2 matrix of points on the lines

        Returns:
            np.ndarray: 3x1 intersection point, or [inf, inf, inf] if lines are parallel
        """
        Axes = np.asarray(Axes, dtype=float).reshape(3,2)
        Points = np.asarray(Points, dtype=float).reshape(3,2)

        c = Points[:, 0]
        e = Axes[:, 0]

        d = Points[:, 1]
        f = Axes[:, 1]

        g = d - c
        Cfg = np.cross(f, g)
        Cfe = np.cross(f, e)

        if np.linalg.norm(Cfe) < 1e-6:
            return np.array([np.inf, np.inf, np.inf])

        if np.dot(Cfg, Cfe) >= 0:
            return c + (np.linalg.norm(Cfg) / np.linalg.norm(Cfe)) * e
        else:
            return c - (np.linalg.norm(Cfg) / np.linalg.norm(Cfe)) * e


    @staticmethod
    def minus_poseEul(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """
        Compute the difference between two poses represented in Euler angles.
        
        Pose = [trvX trvY trvZ rotX rotY rotZ]. Returns a pose difference expresed as Euler coordinates with the same structure as the inputs: 
            pose difference = pose2 - pose1
        expressed in the same reference frame as both input poses.
        The formulation is as follows:
            Position difference = [trvX1-trvX2 trvY1-trvY2 trvZ1-trvZ2]
            Rotation difference = R12 = R01' * R02

        Args:
            pose1 (np.ndarray): 6x1 pose vector
            pose2 (np.ndarray): 6x1 pose vector

        Returns:
            np.ndarray: 6x1 pose difference vector
        """

        pose1 = np.asarray(pose1, float).reshape(6,)
        pose2 = np.asarray(pose2, float).reshape(6,)

        vtcpS = pose2[0:3] - pose1[0:3]

        R01 = RotationUtils.eul_2_rotm(pose1[3:6], seq="XYZ")
        R02 = RotationUtils.eul_2_rotm(pose2[3:6], seq="XYZ")

        R12 = R01.T @ R02
        axis, angle = RotationUtils.rotm_2_axang(R12)

        wstS = R01 @ (axis * angle)

        return np.concatenate([vtcpS, wstS])
    
    
    @staticmethod
    def project_perp(v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Project vector v onto the plane perpendicular to vector w.

        Args:
            v (np.ndarray): 3x1 vector to be projected
            w (np.ndarray): 3x1 vector defining the normal to the plane

        Returns:
            np.ndarray: 3x1 projected vector
        """
        v = np.asarray(v, dtype=float).reshape(3,1)
        w = np.asarray(w, dtype=float).reshape(3,1)

        w = w / np.linalg.norm(w)
        projection = v - (w @ w.T) @ v
        return projection
