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
from .twist_utils import TwistUtils
from .utils import Utils
from .inertia_utils import InertiaUtils


class CoriolisUtils:
    """
    Utility class for computing Coriolis and centrifugal terms in robotic dynamics.
    """

    def christoffel(TwMag, LiMas, i, j, k):
        """
        Compute Christoffel term Γ_ijk

        Parameters:
        TwMag : (7,n) twist + theta
        LiMas : (7,n) link mass properties
        i,j,k : indices (0-based)

        Returns:
        dMt : scalar
        """

        n = TwMag.shape[1]
        m = max(i, j)

        dMt = 0.0

        for l in range(m, n):
            Hsli0 = TformUtils.trvP_2_tform(LiMas[0:3, l])
            Ii = LiMas[3:6, l]
            mi = LiMas[6, l]

            ImS = InertiaUtils.link_inertia_S(Hsli0, Ii, mi)

            # --- First term ---
            term1 = TwistUtils.twist_bracket(
                Utils.Aij_2_adjoint(k, i, TwMag) @ TwMag[0:6, i],
                TwMag[0:6, k]
            ).T

            term1 = term1 @ Utils.Aij_2_adjoint(l, k, TwMag).T
            term1 = term1 @ ImS
            term1 = term1 @ Utils.Aij_2_adjoint(l, j, TwMag)
            term1 = term1 @ TwMag[0:6, j]

            # --- Second term ---
            term2 = TwMag[0:6, i].T @ Utils.Aij_2_adjoint(l, i, TwMag).T
            term2 = term2 @ ImS
            term2 = term2 @ Utils.Aij_2_adjoint(l, k, TwMag)

            term2 = term2 @ TwistUtils.twist_bracket(
                Utils.Aij_2_adjoint(k, j, TwMag) @ TwMag[0:6, j],
                TwMag[0:6, k]
            )

            dMt += term1 + term2

        return dMt
    
    def coriolis_Aij(TwMag, LiMas, Thetap):
        """
        Compute Coriolis matrix C(q, qdot)

        Parameters:
        TwMag  : (7,n)
        LiMas  : (7,n)
        Thetap : (n,) joint velocities

        Returns:
        Ctdt : (n,n)
        """
        n = TwMag.shape[1]
        Ctdt = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                cij = 0.0

                for k in range(n):
                    c_ijk = CoriolisUtils.christoffel(TwMag, LiMas, i, j, k)
                    c_ikj = CoriolisUtils.christoffel(TwMag, LiMas, i, k, j)
                    c_kji = CoriolisUtils.christoffel(TwMag, LiMas, k, j, i)

                    term = (c_ijk + c_ikj - c_kji) * Thetap[k]
                    cij += term

                Ctdt[i, j] = 0.5 * cij

        return Ctdt