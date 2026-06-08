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
from .tform_utils import TformUtils
from .inertia_utils import InertiaUtils
from .utils import Utils
from .forward_kine import ForwardKine
from .wrench_utils import WrenchUtils
from .jacobian import Jacobian



class PotentialUtils:
    """
    Utility class for computing potential energy and gravity vector in robotic dynamics.
    """

    @staticmethod
    def N_potential_Aij(TwMag, LiMas, PoAcc):
        """
        Compute gravity vector using Aij formulation.

        Parameters:
        TwMag : (7,n)
        LiMas : (7,n)
        PoAcc : (3,) gravity acceleration vector

        Returns:
        Nt : (n,)
        """
        n = TwMag.shape[1]

        # Spatial gravity vector (only linear acceleration)
        Eg = np.concatenate([PoAcc, np.zeros(3)])

        Nt = np.zeros(n)

        for i in range(n):
            for l in range(i, n):
                Hsli0 = TformUtils.trvP_2_tform(LiMas[0:3, l])
                Ii = LiMas[3:6, l]
                mi = LiMas[6, l]

                ImS = InertiaUtils.link_inertia_S(Hsli0, Ii, mi)

                term = TwMag[0:6, i].T @ Utils.Aij_2_adjoint(l, i, TwMag).T
                term = term @ ImS
                term = term @ Utils.Aij_2_adjoint(l, 0, TwMag)  
                term = term @ Eg

                Nt[i] -= term

        return Nt
    
    @staticmethod
    def N_potential_sym(TwMag, LiMas, PoAcc):
        """
        Symbolic gravity vector via potential energy differentiation.

        Parameters:
        TwMag : (7,n)
        LiMas : (7,n)
        PoAcc : (3,) gravity acceleration vector    

        Returns:
        Nt : (n,)
        """
        
        n = TwMag.shape[1]

        # Symbolic joint variables
        theta = sp.symbols(f't1:{n+1}')
        ThSym = sp.Matrix(theta)

        TwMagSym = sp.Matrix.vstack(
            sp.Matrix(TwMag[0:6, :]),
            ThSym.T
        )

        V = sp.zeros(3, 1)

        for i in range(n):
            Hsli0 = TformUtils.trvP_2_tform(LiMas[0:3, i])

            Hslit = ForwardKine.forward_kinematics_poe(TwMagSym[:, 0:i+1]) * sp.Matrix(Hsli0)

            mi = LiMas[6, i]
            pos = Hslit[0:3, 3]

            # Equivalent to diag(m*g)*position
            V += mi * sp.Matrix(PoAcc).multiply_elementwise(pos)

        NtSym = sp.zeros(n, 1)

        for i in range(n):
            NtSym[i] = sum(sp.diff(V[j], ThSym[i]) for j in range(3))

        NtSym = sp.simplify(NtSym)

        Nt = np.array(NtSym.subs(dict(zip(theta, TwMag[6, :]))), dtype=float).flatten()

        return -Nt
    
    @staticmethod
    def N_potential_wrench(TwMag, LiMas, PoAcc):
        """
        Gravity via wrench propagation.
        """
        n = TwMag.shape[1]

        MagnG = np.linalg.norm(PoAcc)
        AxisG = PoAcc / MagnG

        Wrench = np.zeros((6, n))

        for i in range(n):
            Hsli0 = TformUtils.trvP_2_tform(LiMas[0:3, i])
            Hslit = ForwardKine.forward_kinematics_poe(TwMag[:, 0:i+1]) @ Hsli0

            mi = LiMas[6, i]

            Wrench[:, i] = mi * MagnG * WrenchUtils.link_2_wrench(
                AxisG,
                Hslit[0:3, 3],
                forcetype='tra'
            )

        Nt = np.zeros(n)

        for i in range(n):
            JstS = np.zeros((6, n))
            JstS[:, 0:i+1] = Jacobian.GeoJacobianS(TwMag[:, 0:i+1])

            Ntnew = JstS.T @ Wrench[:, i]
            Nt -= Ntnew

        return Nt