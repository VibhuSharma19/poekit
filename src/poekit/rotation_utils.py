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

class RotationUtils:
    """
    Utility class for rotation operations in 3D space. 
    """


    def __init__(self):
        pass


    @staticmethod
    def rotation_angle(r: np.array) -> float:
        """
        Find the angle "theta" of a rotation matrix.
        Use in SO(3).

        Args:
            r (np.array): 3x3 rotation matrix

        Returns:
            float: rotation angle in radians
        """

        r = np.asarray(r, dtype=float).reshape(3,3)
        trace_r = np.trace(r)
        theta = np.arccos( (trace_r - 1) / 2 )
        return theta
    
    
    @staticmethod
    def rotation_axis(r: np.array, theta: float) -> np.array:
        """
        Find the axis "w" of a rotation matrix.
        Use in SO(3).

        Args:
            r (np.array): 3x3 rotation matrix
            theta (float): rotation angle in radians
        Returns:
            np.array: A 3x1 vector representing the axis of rotation
        """

        r = np.asarray(r, dtype=float).reshape(3,3)
        sin_theta = np.sin(theta)

        if np.isclose(sin_theta, 0):
            w = np.array([[0],[0],[0]])
        else:
            w = np.array([[r[2,1] - r[1,2]], 
                          [r[0,2] - r[2,0]], 
                          [r[1,0] - r[0,1]]]) / (2 * sin_theta)
        return w
    
    
    @staticmethod
    def rotm_2_axang(r: np.array) -> tuple:
        """
        Convert a rotation matrix to axis-angle representation.

        Args:
            r (np.array): 3x3 rotation matrix
        
        Returns:
            tuple: (axis (np.array), angle (float))
        """

        r = np.asarray(r, dtype=float).reshape(3,3)
        theta = RotationUtils.rotation_angle(r)
        w = RotationUtils.rotation_axis(r, theta).reshape(3,)

        return (w, theta)
    
    
    @staticmethod
    def rotX_to_tform(th: float) -> np.array:
        """
        Convert a rotation about the X-axis to a transformation matrix.

        Args:
            th (float): rotation angle in radians
        Returns:
            np.array: A 4x4 transformation matrix
        """
        
        r = np.array([[1, 0, 0],
                      [0, np.cos(th), -np.sin(th)],
                      [0, np.sin(th),  np.cos(th)]])
        
        tform = np.eye(4)
        tform[0:3, 0:3] = r
        return tform
    
    
    @staticmethod
    def rotZ_to_tform(th: float) -> np.array:
        """
        Convert a rotation about the Z-axis to a transformation matrix.

        Args:
            th (float): rotation angle in radians
        Returns:
            np.array: A 4x4 transformation matrix
        """
        
        r = np.array([[ np.cos(th), -np.sin(th), 0],
                      [ np.sin(th),  np.cos(th), 0],
                      [     0,           0,      1]])
        
        tform = np.eye(4)
        tform[0:3, 0:3] = r
        return tform
    
    
    @staticmethod
    def rotY_to_tform(th: float) -> np.array:
        """
        Convert a rotation about the Y-axis to a transformation matrix.

        Args:
            th (float): rotation angle in radians
        Returns:
            np.array: A 4x4 transformation matrix
        """
        
        r = np.array([[ np.cos(th), 0, np.sin(th)],
                      [     0,      1,     0     ],
                      [-np.sin(th), 0, np.cos(th)]])
        
        tform = np.eye(4)
        tform[0:3, 0:3] = r
        return tform


    @staticmethod
    def quat_to_eulZYX(Q: np.ndarray) -> np.ndarray:
        """
        Convert a quaternion [q0, q1, q2, q3] into ZYX Euler angles:
            Z = yaw (psi)
            Y = pitch (theta)
            X = roll (phi)
        All angles returned in radians.

        MATLAB equivalent: quat2eulZYX(Q)
        """

        Q = np.asarray(Q, dtype=float).reshape(4,)
        q0, q1, q2, q3 = Q

        # Yaw (Z rotation)
        psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

        # Pitch (Y rotation)
        theta = np.arcsin(2*(q0*q2 - q3*q1))

        # Roll (X rotation)
        phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))

        return np.array([psi, theta, phi])
    

    @staticmethod
    def eul_2_rotm(eul: np.array, seq:str = "ZYX") -> np.array:
        """
        Convert Euler angles to rotation matrix.

        Args:
            eul (np.array): 3x1 vector of Euler angles in radians
            seq (str, optional): Rotation sequence: 'ZYX', 'ZYZ', or 'XYZ'. Defaults to "ZYX".

        Returns:
            np.array: 3x3 rotation matrix
        """

        eul = np.asarray(eul, dtype=float)

        # Normalize input to shape (N,3)
        if eul.ndim == 1:
            eul = eul.reshape(1, 3)
            squeeze_output = True
        else:
            squeeze_output = False

        N = eul.shape[0]

        ct = np.cos(eul)
        st = np.sin(eul)

        R = np.zeros((3, 3, N), dtype=float)
        seq = seq.upper()

        if seq == "ZYX":
            for i in range(N):
                cz, cy, cx = ct[i]
                sz, sy, sx = st[i]

                R[:, :, i] = np.array([
                    [cy * cz,              cy * sz,               -sy],
                    [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy],
                    [cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy]
                ])

        elif seq == "ZYZ":
            for i in range(N):
                cz, cy, cz2 = ct[i]
                sz, sy, sz2 = st[i]

                R[:, :, i] = np.array([
                    [cz2 * cy * cz - sz2 * sz, -sz2 * cy * cz - cz2 * sz, sy * cz],
                    [cz2 * cy * sz + sz2 * cz, -sz2 * cy * sz + cz2 * cz, sy * sz],
                    [-cz2 * sy,                sz2 * sy,                 cy]
                ])

        elif seq == "XYZ":
            for i in range(N):
                cx, cy, cz = ct[i]
                sx, sy, sz = st[i]

                R[:, :, i] = np.array([
                    [cy * cz,             -cy * sz,               sy],
                    [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                    [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz,  cx * cy]
                ])

        else:
            raise ValueError("Invalid Euler sequence. Use 'ZYX', 'ZYZ', or 'XYZ'.")

        if squeeze_output:
            return R[:, :, 0]

        return R





#th = np.pi / 4
#r_x = RotationUtils.rotX_to_tform(th)
#r_y = RotationUtils.rotY_to_tform(th)
#r_z = RotationUtils.rotZ_to_tform(th)
#w = RotationUtils.rotation_axis(r_x[0:3,0:3], th)
#angle = RotationUtils.rotation_angle(r_x[0:3,0:3])
#e = RotationUtils.quat_to_eulZYX([0.9239, 0.3827, 0, 0])  # 45 deg about X
#print("Rotation about X:\n", r_x)
#print("Axis:\n", w)
#print("Angle (rad): ", angle)
#print("Euler ZYX (rad): ", e)