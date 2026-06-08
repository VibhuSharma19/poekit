"""
Forward kinematics example for the Kinova Jaco2 7-DOF manipulator.

This example demonstrates the use of poekit for computing
POE-based forward kinematics.
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


## ============================================================
##   BUILD TWIST MATRIX (6×7)
## ============================================================

Point = np.column_stack((po, pk, pk, pr, pf, pf, pf))   # shape (3,7)
Axis  = np.column_stack((-A_Z, -A_Y, A_Z, A_Y, -A_Z, -A_Y, A_Z))  # shape (3,7)

Twist = np.zeros((6, 7), dtype=float)
for i in range(7):
    w = Axis[:, i].reshape(3,)
    q = Point[:, i].reshape(3,)
    v = -np.cross(w, q)           
    Twist[:, i] = np.hstack((v, w))

print("\n--- Arm Twist ---\n")
print(Twist)

# ============================================================
#   OBTAIN RANDOM MAGNITUDES
# ============================================================

Mag = (np.random.rand(7) - np.random.rand(7)) * np.pi
Mag_row = np.asarray(Mag).reshape(1, -1)   
Mag_row = np.array([
    Utils.jointmag_2_limits(Mag[i], joint_limits[i, 1], joint_limits[i, 0])
    for i in range(7)
]).reshape(1, 7)

TwMag = np.vstack((Twist, Mag_row))   
print("\n--- Joint Magnitudes ---\n")         
print(Mag_row)

# ============================================================
#   GET FORWARD KINEMATICS
# ============================================================

HstR = ForwardKine.forward_kinematics_poe(TwMag)

Hst0 = TformUtils.trvP_2_tform(pp) @ RotationUtils.rotX_to_tform(np.pi) @ RotationUtils.rotZ_to_tform(-np.pi/2)
noap = HstR @ Hst0

print("\n--- Forward Kinematics ---\n")
print(noap)