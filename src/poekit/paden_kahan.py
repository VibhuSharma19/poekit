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
from .utils import Utils

np.set_printoptions(suppress=False, precision=5)

def _ensure_col(x, length=None):
    """Return array as column vector (n,1). If length provided, checks length."""
    a = np.asarray(x, dtype=float).reshape(-1)
    if length is not None and a.size != length:
        raise ValueError(f"Expected length {length}, got {a.size}")
    return a.reshape(a.size, 1)

class PadenKahan:
    """
    Static implementations of Paden-Kahan subproblems for POE IK.
    """
    _EPS = 1e-6
    
    def __init__(self):
        pass


    def paden_kahan_1(x1, pp, pk):
        """
        PK1: Theta such that exp(E^Theta) * pp = pk for a pure rotation screw E = [v; w]
        Returns scalar theta (float).
        """
        x1 = np.asarray(x1, dtype=float).reshape(6,)
        v1 = _ensure_col(x1[0:3], 3)
        w1 = _ensure_col(x1[3:6], 3)

        pp = _ensure_col(pp, 3)
        pk = _ensure_col(pk, 3)

        wnorm2 = (w1.T @ w1).item()
        if wnorm2 < PadenKahan._EPS:
            raise ValueError("paden_kahan_1: axis norm too small (not a rotation).")

        r1 = np.cross(w1.flatten(), v1.flatten()).reshape(3, 1) / wnorm2

        u = pp - r1
        up = u - w1 * (w1.T @ u).item()

        v = pk - r1
        vp = v - w1 * (w1.T @ v).item()

        num = (w1.T @ np.cross(up.flatten(), vp.flatten()).reshape(3, 1)).item()
        den = (up.T @ vp).item()

        theta = float(np.arctan2(num, den))
        return theta


    def paden_kahan_2(x1, x2, pp, pk):
        """
        PK2: Solve for two consecutive rotations:
            exp(E1^th1) * exp(E2^th2) * pp = pk
        Returns 2x2 array [[t11,t21],[t12,t22]]; returns zeros matrix when no solution.
        """
        x1 = np.asarray(x1, dtype=float).reshape(6,)
        x2 = np.asarray(x2, dtype=float).reshape(6,)
        v1 = _ensure_col(x1[0:3], 3); w1 = _ensure_col(x1[3:6], 3)
        v2 = _ensure_col(x2[0:3], 3); w2 = _ensure_col(x2[3:6], 3)

        pp = _ensure_col(pp, 3)
        pk = _ensure_col(pk, 3)

        w1_norm2 = (w1.T @ w1).item()
        w2_norm2 = (w2.T @ w2).item()
        if w1_norm2 < PadenKahan._EPS or w2_norm2 < PadenKahan._EPS:
            return np.zeros((2, 2), dtype=float)

        w1u = w1 / np.sqrt(w1_norm2)
        w2u = w2 / np.sqrt(w2_norm2)

        r1 = np.cross(w1u.flatten(), v1.flatten()).reshape(3, 1) / (1.0)  # because w1u already unit
        r2 = np.cross(w2u.flatten(), v2.flatten()).reshape(3, 1) / (1.0)

        axes = np.column_stack((w1u.flatten(), w2u.flatten()))
        points = np.column_stack((r1.flatten(), r2.flatten()))
        
        pr = Utils.intersect_lines3D(axes, points)

        pr = np.asarray(pr, dtype=float).reshape(-1)
        if np.isinf(pr[0]):
            return np.zeros((2, 2), dtype=float)
        pr = pr.reshape(3, 1)

        u = pp - pr
        v = pk - pr

        Cw1w2 = np.cross(w1u.flatten(), w2u.flatten()).reshape(3, 1)
        w1Tw2 = (w1u.T @ w2u).item()

        denom = (w1Tw2 ** 2 - 1.0)
        if abs(denom) < PadenKahan._EPS:
            return np.zeros((2, 2), dtype=float)

        w2Tu = (w2u.T @ u).item()
        w1Tu = (w1u.T @ u).item()
        w1Tv = (w1u.T @ v).item()
        w2Tv = (w2u.T @ v).item()

        a = ((w1Tw2) * w2Tu - w1Tv) / denom
        b = ((w1Tw2) * w1Tv - w2Tu) / denom

        denomC = (Cw1w2.T @ Cw1w2).item()
        if denomC < PadenKahan._EPS:
            return np.zeros((2, 2), dtype=float)

        g2 = abs((np.linalg.norm(u)**2 - a**2 - b**2 - 2*a*b*w1Tw2) / denomC)
        if g2 < -1e-12:
            return np.zeros((2, 2), dtype=float)
        g = np.sqrt(max(g2, 0.0))

        pc = pr + a*w1u + b*w2u + g*Cw1w2
        pd = pr + a*w1u + b*w2u - g*Cw1w2

        m = pc - pr
        n = pd - pr

        def proj_perp(x, w):
            return x - w * (w.T @ x).item()

        up = proj_perp(u, w2u)
        vp = proj_perp(v, w1u)

        m2p = proj_perp(m, w2u)
        m1p = proj_perp(m, w1u)
        n2p = proj_perp(n, w2u)
        n1p = proj_perp(n, w1u)

        t21 = float(np.arctan2((w2u.T @ np.cross(up.flatten(), m2p.flatten()).reshape(3,1)).item(),
                               (up.T @ m2p).item()))
        t11 = float(np.arctan2((w1u.T @ np.cross(m1p.flatten(), vp.flatten()).reshape(3,1)).item(),
                               (m1p.T @ vp).item()))

        t22 = float(np.arctan2((w2u.T @ np.cross(up.flatten(), n2p.flatten()).reshape(3,1)).item(),
                               (up.T @ n2p).item()))
        t12 = float(np.arctan2((w1u.T @ np.cross(n1p.flatten(), vp.flatten()).reshape(3,1)).item(),
                               (n1p.T @ vp).item()))

        return np.array([[t11, t21], [t12, t22]], dtype=float)


    def paden_kahan_3(x1, pp, pk, de):
        """
        PK3: Solve theta such that ||exp(E^theta)*pp - pk|| = de
        Returns array([theta1, theta2]) (two possible angles).
        """
        x1 = np.asarray(x1, dtype=float).reshape(6,)
        v1 = _ensure_col(x1[0:3], 3)
        w1 = _ensure_col(x1[3:6], 3)

        pp = _ensure_col(pp, 3)
        pk = _ensure_col(pk, 3)

        wnorm2 = (w1.T @ w1).item()
        if wnorm2 < PadenKahan._EPS:
            raise ValueError("paden_kahan_3: axis norm too small (not a rotation).")

        w1u = w1 / np.sqrt(wnorm2)
        r1 = np.cross(w1u.flatten(), v1.flatten()).reshape(3, 1) / 1.0

        u = pp - r1
        up = u - w1u * (w1u.T @ u).item()
        nup = np.linalg.norm(up)

        v = pk - r1
        vp = v - w1u * (w1u.T @ v).item()
        nvp = np.linalg.norm(vp)

        num = (w1u.T @ np.cross(up.flatten(), vp.flatten()).reshape(3,1)).item()
        den = (up.T @ vp).item()
        alfa1 = float(np.arctan2(num, den))

        d_parallel_sq = ((w1u.T @ (pp - pk))**2).item()

        dep2 = float(de*de - d_parallel_sq)

        denom = 2.0 * nup * nvp
        if denom < PadenKahan._EPS:
            return np.array([alfa1, alfa1], dtype=float)

        beta = (nup*nup + nvp*nvp - dep2) / denom
        beta = float(np.clip(beta, -1.0, 1.0))
        beta1 = float(np.arccos(beta))

        return np.array([alfa1 - beta1, alfa1 + beta1], dtype=float)
    
    
