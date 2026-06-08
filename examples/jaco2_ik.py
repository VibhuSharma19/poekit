"""
Inverse kinematics example for the Kinova Jaco2 7-DOF manipulator.

This example demonstrates the use of poekit for computing
PK sub-problems based Inverse kinematics.

The method and flow of IK is taken from Exercise in ST24R Matlab toolbox by Dr. Jose M. Pardos-Gotor
"""

import numpy as np
from poekit.expmapping import ExpMapping
from poekit.forward_kine import ForwardKine
from poekit.paden_kahan import PadenKahan
from poekit.rotation_utils import RotationUtils
from poekit.tform_utils import TformUtils
from poekit.utils import Utils

np.set_printoptions(precision=5, suppress=False)

# ============================================================
#   JACO2 MECHANICAL CONSTANTS  (AT HOME CONFIGURATION)
# ============================================================

po = np.array([[0.0], [0.0], [0.0]])
pk = np.array([[0.0], [0.0], [0.2755]])
pr = np.array([[0.0], [0.0], [-0.1345]])
pf = np.array([[0.0], [-0.0098], [0.1766]])
pp = np.array([[0.0], [-0.0098], [-0.0872]])

A_X = np.array([[1.0], [0.0], [0.0]])
A_Y = np.array([[0.0], [1.0], [0.0]])
A_Z = np.array([[0.0], [0.0], [1.0]])


Joint = ["rot"] * 7
joint_limits = np.array([[-2*np.pi, 2*np.pi],[0.261*np.pi, 1.7389*np.pi],
                        [-2*np.pi, 2*np.pi], [0.1667*np.pi, 1.833*np.pi], 
                        [-2*np.pi, 2*np.pi], [0.361*np.pi, 1.6389*np.pi], 
                        [-2*np.pi, 2*np.pi]])

Point = np.column_stack((po, pk, pk, pr, pf, pf, pf)) 
Axis  = np.column_stack((-A_Z, -A_Y, A_Z, A_Y, -A_Z, -A_Y, A_Z))  

Twist = np.zeros((6, 7), dtype=float)
for i in range(7):
    w = Axis[:, i].reshape(3,)
    q = Point[:, i].reshape(3,)
    v = -np.cross(w, q)           
    Twist[:, i] = np.hstack((v, w))

# ============================================================
#   STEP 1: FORWARD KINEMATICS FOR RANDOM MAGNITUDES
# ============================================================
Mag = (np.random.rand(7) - np.random.rand(7)) * np.pi
Mag_row = np.asarray(Mag).reshape(1, -1)   
Mag_row = np.array([
    Utils.jointmag_2_limits(Mag[i], joint_limits[i, 1], joint_limits[i, 0])
    for i in range(7)
]).reshape(1, 7)

TwMag = np.vstack((Twist, Mag_row))

HstR = ForwardKine.forward_kinematics_poe(TwMag)

Hst0 = TformUtils.trvP_2_tform(pp) @ RotationUtils.rotX_to_tform(np.pi) @ RotationUtils.rotZ_to_tform(-np.pi/2)
noap = HstR @ Hst0

print("\n--- STEP 1: Forward Kinematics Target ---\n")
print(noap)

# ============================================================
#   STEP 2: INVERSE KINEMATICS (SCREW THEORY)
#           Using Paden–Kahan Subproblems
# ===========================================================

Theta = np.zeros((16, 7))

# ------------------------------
# 2.0 SEED VALUES USING PK1
# ------------------------------

target_point = noap[0:3, 3].reshape(3, 1) 
t1 = PadenKahan.paden_kahan_1(Twist[:, 0].reshape(6,), np.array([1.0, 0.0, 0.0]), target_point)
t3 = PadenKahan.paden_kahan_1(Twist[:, 2].reshape(6,), np.array([1.0, 0.0, 0.0]), target_point)

Theta[0:8, 2] = float(t3)         
Theta[8:16, 0] = float(t1)

# ------------------------------
# 2.1 SOLVE THETA4 USING PK3
# ------------------------------

pf_h = np.vstack((pf.reshape(3, 1), np.array([[1.0]])))
noapHst0pf = noap @ np.linalg.solve(Hst0, pf_h)   
pk1p = noapHst0pf[0:3, :].reshape(3,)

de = float(np.linalg.norm(pk1p - pk.reshape(3,)))

t4_vals = PadenKahan.paden_kahan_3(Twist[:, 3].reshape(6,), pf.reshape(3,), pk.reshape(3,), de)

t4a = float(t4_vals[0])
t4b = float(t4_vals[1])

Theta[0:4, 3] = t4a               
Theta[4:8, 3] = t4b
Theta[8:12, 3] = t4a
Theta[12:16, 3] = t4b

# ------------------------------
# 2.2 SOLVE THETA1 & THETA2 (PK2)
# ------------------------------

for base_idx in [0, 4]:

    twmag3 = np.concatenate((Twist[:, 2].reshape(6,), np.array([Theta[base_idx, 2]])))
    twmag4 = np.concatenate((Twist[:, 3].reshape(6,), np.array([Theta[base_idx, 3]])))

    pf1pt = ExpMapping.exp_screw(twmag3) @ ExpMapping.exp_screw(twmag4) @ pf_h
    pf1p = pf1pt[0:3, :].reshape(3,)

    t1t2_mat = np.asarray(
        PadenKahan.paden_kahan_2(
            Twist[:, 0].reshape(6,), 
            Twist[:, 1].reshape(6,), 
            pf1p, 
            pk1p
        )
    ).reshape(2,2)

    Theta[base_idx + 0, 0:2] = t1t2_mat[0, :]
    Theta[base_idx + 1, 0:2] = t1t2_mat[0, :]
    Theta[base_idx + 2, 0:2] = t1t2_mat[1, :]
    Theta[base_idx + 3, 0:2] = t1t2_mat[1, :]

# ------------------------------
# 2.3 SOLVE THETA2 & THETA3 (PK2 reversed order)
# ------------------------------

for base_idx in [8, 12]:
    twmag4 = np.concatenate((Twist[:, 3].reshape(6,), np.array([Theta[base_idx, 3]])))
    pf2pt = ExpMapping.exp_screw(twmag4) @ pf_h
    pf2p = pf2pt[0:3, :].reshape(3,)

    twmag1 = np.concatenate((Twist[:, 0].reshape(6,), np.array([Theta[base_idx, 0]])))
    pk3pt = np.linalg.solve(ExpMapping.exp_screw(twmag1), noapHst0pf)
    pk3p = pk3pt[0:3, :].reshape(3,)

    t2t3_mat = PadenKahan.paden_kahan_2(Twist[:, 1].reshape(6,), Twist[:, 2].reshape(6,), pf2p, pk3p)
    t2t3_mat = np.asarray(t2t3_mat).reshape(2, 2)

    Theta[base_idx + 0, 1:3] = t2t3_mat[0, :]
    Theta[base_idx + 1, 1:3] = t2t3_mat[0, :]
    Theta[base_idx + 2, 1:3] = t2t3_mat[1, :]
    Theta[base_idx + 3, 1:3] = t2t3_mat[1, :]
    
# ------------------------------
# 2.4 SOLVE THETA5 & THETA6 (PK2)
# ------------------------------

pp_h = np.vstack((pp.reshape(3, 1), np.array([[1.0]])))
noapHst0pp = noap @ np.linalg.solve(Hst0, pp_h)

for i in range(0, 16, 2):
    pk3pt = np.linalg.solve(ExpMapping.exp_screw(np.concatenate((Twist[:, 0].reshape(6,), np.array([Theta[i, 0]])))), noapHst0pp)
    pk3pt = np.linalg.solve(ExpMapping.exp_screw(np.concatenate((Twist[:, 1].reshape(6,), np.array([Theta[i, 1]])))), pk3pt)
    pk3pt = np.linalg.solve(ExpMapping.exp_screw(np.concatenate((Twist[:, 2].reshape(6,), np.array([Theta[i, 2]])))), pk3pt)
    pk3pt = np.linalg.solve(ExpMapping.exp_screw(np.concatenate((Twist[:, 3].reshape(6,), np.array([Theta[i, 3]])))), pk3pt)
    pk3p = pk3pt[0:3, :].reshape(3,)

    t56 = PadenKahan.paden_kahan_2(Twist[:, 4].reshape(6,), Twist[:, 5].reshape(6,), pp.reshape(3,), pk3p)
    t56 = np.asarray(t56).reshape(2, 2)
    Theta[i + 0:i + 2, 4:6] = t56

# ------------------------------
# 2.5 SOLVE THETA7 USING PK1
# ------------------------------

noapHst0po = noap @ np.linalg.solve(Hst0, np.array([[1.0], [0.0], [0.0], [1.0]]))

for i in range(16):
    pk4pt = noapHst0po.copy()
    for j in range(6):
        A = ExpMapping.exp_screw(np.concatenate((Twist[:, j].reshape(6,), np.array([Theta[i, j]]))))
        pk4pt = np.linalg.solve(A, pk4pt)

    pk4p = pk4pt[0:3, :].reshape(3,)
    Theta[i, 6] = float(PadenKahan.paden_kahan_1(Twist[:, 6].reshape(6,), np.array([1.0, 0.0, 0.0]), pk4p))

print("\n--- STEP 2: IK Solutions ---")
print(Theta)

# ============================================================
#   STEP 3: VERIFY ALL 16 SOLUTIONS USING FORWARD KINEMATICS
# ============================================================

print("\n--- STEP 3: Forward Kine Check of IK Solutions ---\n")

for i in range(16):
    TwMagi = np.vstack((Twist, Theta[i, :].reshape(1, -1)))   # (7,7)
    Hchk = ForwardKine.forward_kinematics_poe(TwMagi) @ Hst0
    err = np.linalg.norm(Hchk - noap)
    print(f"\nSolution {i + 1}: err={err:.5e}")
    print(Hchk)