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
from .utils import Utils
from numpy.linalg import solve
from .expmapping import ExpMapping

np.set_printoptions(suppress=False, precision=5)

class Jacobian:
    """
    Class for computing different types of Jacobians using POE methods. 
    Contains methods to compute jacobian in space frame, link frame and tool frame.
    """

    def __init__(self):
        pass

    
    @staticmethod
    def GeoJacobianL(TwMag: np.ndarray, Hsli0: np.ndarray, Li: int) -> np.ndarray:
        """
        Spatial Jacobian expressed at link Li frame.

        Args:
            TwMag  : (7, n) twist+angle matrix
            Hsli0  : (4,4) transform from space to link-i frame at home
            Li     : link index (int)

        Returns:
            JslT : (6, n)
        """
        n = TwMag.shape[1]
        JslT = np.zeros((6, n))

        AdHsli0 = TformUtils.tform_2_adjoint(Hsli0)

        for j in range(n):
            Aij = Utils.Aij_2_adjoint(Li, j + 1, TwMag) 
            JslT[:, j] = solve(
                AdHsli0,
                Aij @ TwMag[0:6, j]
            )

        return JslT
    

    @staticmethod
    def GeoJacobianS(TwMag: np.ndarray) -> np.ndarray:
        """
        Spatial Jacobian expressed in the space frame.

        Args:
            TwMag : (7, n)

        Returns:
            JstS : (6, n)
        """
        n = TwMag.shape[1]
        JstS = np.zeros((6, n))

        JstS[:, 0] = TwMag[0:6, 0]

        PoE = ExpMapping.exp_screw(TwMag[:, 0])

        for i in range(1, n):
            JstS[:, i] = TformUtils.tform_2_adjoint(PoE) @ TwMag[0:6, i]
            PoE = PoE @ ExpMapping.exp_screw(TwMag[:, i])

        return JstS
    
    
    @staticmethod
    def GeoJacobianT(TwMag: np.ndarray, Hst0: np.ndarray) -> np.ndarray:
        """
        Spatial Jacobian expressed in the tool frame.

        Args:
            TwMag : (7, n)
            Hst0  : (4,4) home configuration

        Returns:
            JstT : (6, n)
        """
        n = TwMag.shape[1]
        JstT = np.zeros((6, n))

        PoE = Hst0.copy()

        for i in reversed(range(n)):
            PoE = ExpMapping.exp_screw(TwMag[:, i]) @ PoE
            JstT[:, i] = solve(
                TformUtils.tform_2_adjoint(PoE),
                TwMag[0:6, i]
            )

        return JstT
