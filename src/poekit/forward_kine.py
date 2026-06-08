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
from .expmapping import ExpMapping

np.set_printoptions(suppress=False, precision=5)

class ForwardKine:
    """
    Class for computing forward kinematics using different methods.
    """

    def __init__(self):
        pass

    @staticmethod
    def forward_kinematics_poe(TwMag: np.ndarray) -> np.ndarray:
        """
        Compute the forward kinematics using the Product of Exponentials formula.

        Args:
            TwMag (np.ndarray): 7xn array where each column is a twist (6x1) with joint magnitude (1x1) at the end.

        Raises:
            ValueError: If TwMag is not of shape (7, n).

        Returns:
            np.ndarray: 4x4 transformation matrix representing the end-effector pose.
        """

        TwMag = np.asarray(TwMag, dtype=float)
        if TwMag.ndim != 2 or TwMag.shape[0] != 7:
            raise ValueError("TwMag must be shape (7, n)")

        n = TwMag.shape[1]
        # start with first transform
        HstR = ExpMapping.exp_screw(TwMag[:,0])
        for i in range(1, n):
            HstR = HstR @ ExpMapping.exp_screw(TwMag[:, i])
        return HstR