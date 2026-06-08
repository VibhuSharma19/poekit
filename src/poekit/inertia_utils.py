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
import sympy as sp
from .skewops import SkewOps
from .tform_utils import TformUtils
from .utils import Utils
from .jacobian import Jacobian


class InertiaUtils:
    """
    Utility class for inertia-related computations, including spatial inertia matrix calculations.
    """

    @staticmethod
    def link_inertia_B(CMi, Ii, mi):
        """
        Spatial inertia in body frame.

        Parameters:
        CMi : (3,) center of mass vector
        Ii  : (3,3) inertia tensor about COM
        mi  : scalar mass

        Returns:
        ImB : (6,6) spatial inertia matrix
        """
        CM = SkewOps.axis_to_skew(CMi)

        upper_left = Ii + mi * (CM @ CM.T)
        upper_right = mi * CM
        lower_left = mi * CM.T
        lower_right = mi * np.eye(3)

        ImB = np.block([
            [upper_left, upper_right],
            [lower_left, lower_right]
        ])

        return ImB
    
    @staticmethod
    def link_inertia_S(Hsli0, Ii, mi):
        """
        Spatial inertia in space frame.

        Parameters:
        Hsli0 : (4,4) homogeneous transform
        Ii    : (3,) principal inertia [Ix, Iy, Iz]
        mi    : scalar mass

        Returns:
        ImS : (6,6)
        """

        I_mat = np.diag(Ii)

        Im = np.block([
            [mi * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), I_mat]
        ])

        # Adjoint transformation
        AdHsli0 = np.linalg.inv(TformUtils.tform_2_adjoint(Hsli0))

        ImS = AdHsli0.T @ Im @ AdHsli0

        return ImS
    
    def M_inertia_Aij(TwMag, LiMas):
        """
        Joint-space inertia matrix using Aij formulation.
        """
        n = TwMag.shape[1]
        Mt = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                k = max(i, j)

                for l in range(k, n):
                    Hsli0 = TformUtils.trvP_2_tform(LiMas[0:3, l])
                    Ii = LiMas[3:6, l]
                    mi = LiMas[6, l]

                    term = TwMag[0:6, i].T @ Utils.Aij_2_adjoint(l, i, TwMag).T
                    term = term @ InertiaUtils.link_inertia_S(Hsli0, Ii, mi)
                    term = term @ Utils.Aij_2_adjoint(l, j, TwMag) @ TwMag[0:6, j]

                    Mt[i, j] += term

        return Mt
    

    def M_inertia_Aij_sym(TwMag, LiMas):
        n = TwMag.shape[1]

        # Symbolic joint variables
        theta = sp.symbols(f't1:{n+1}')

        TwMagSym = sp.Matrix.vstack(
            sp.Matrix(TwMag[0:6, :]),
            sp.Matrix(theta)
        )

        MtSym = sp.zeros(n, n)

        for i in range(n):
            for j in range(n):
                k = max(i, j)

                for l in range(k, n):
                    Hsli0 = TformUtils.trvP_2_tform(LiMas[0:3, l])
                    Ii = LiMas[3:6, l]
                    mi = LiMas[6, l]

                    term = TwMagSym[0:6, i].T * Utils.Aij_2_adjoint(l, i, TwMagSym).T
                    term = term * InertiaUtils.link_inertia_S(Hsli0, Ii, mi)
                    term = term * Utils.Aij_2_adjoint(l, j, TwMagSym) * TwMagSym[0:6, j]

                    MtSym[i, j] += term[0]

        MtSym = sp.simplify(MtSym)

        return MtSym
    
    def M_inertia_Jsl(TwMag, LiMas):
        """
        Joint-space inertia using spatial Jacobian (left Jacobian formulation).
        """
        
        n = TwMag.shape[1]
        Mt = np.zeros((n, n))

        for i in range(n):
            Hsli0 = TformUtils.trvP_2_tform(LiMas[0:3, i])
            Ii = LiMas[3:6, i]
            mi = LiMas[6, i]

            I_mat = np.diag(Ii)

            Mi = np.block([
                [mi * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), I_mat]
            ])

            Jsl = Jacobian.GeoJacobianL(TwMag, Hsli0, i)

            Mt += Jsl.T @ Mi @ Jsl

        return Mt