"""
Geometric Jacobian example for the Kinova Jaco2 7-DOF manipulator.

This example demonstrates the use of poekit for computing
Geometric Space Jacobian
"""

import numpy as np
from poekit.utils import Utils
from poekit.jacobian import Jacobian

np.set_printoptions(precision=5, suppress=False)

# ============================================================
#   JACO2 MECHANICAL CONSTANTS  (AT HOME CONFIGURATION)
# ============================================================

po = np.array([[0.0], [0.0], [0.0]])
ps = np.array([[0.0], [0.0], [0.176]])
pe = np.array([[0.0], [0.0], [0.476]])
pp = np.array([[0.0], [0.0], [0.796]])
pf = np.array([[0.0], [0.0], [1.126]])

A_X = np.array([[1.0], [0.0], [0.0]])
A_Y = np.array([[0.0], [1.0], [0.0]])
A_Z = np.array([[0.0], [0.0], [1.0]])

Joint = ["rot"] * 7
joint_limits = np.array([[-2*np.pi, 2*np.pi],[0.261*np.pi, 1.7389*np.pi],
                        [-2*np.pi, 2*np.pi], [0.1667*np.pi, 1.833*np.pi], 
                        [-2*np.pi, 2*np.pi], [0.361*np.pi, 1.6389*np.pi], 
                        [-2*np.pi, 2*np.pi]])


Point = np.column_stack((po, ps, pe, pe, pp, pp, pp))   
Axis  = np.column_stack((-A_Z, A_Y, -A_Z, A_Y, -A_Z, A_Y, -A_Z))  

Twist = np.zeros((6, 7), dtype=float)
for i in range(7):
    w = Axis[:, i].reshape(3,)
    q = Point[:, i].reshape(3,)
    v = -np.cross(w, q)          
    Twist[:, i] = np.hstack((v, w))

Mag = (np.random.rand(7) - np.random.rand(7)) * np.pi
Mag_row = np.asarray(Mag).reshape(1, -1)   
Mag_row = np.array([
    Utils.jointmag_2_limits(Mag[i], joint_limits[i, 1], joint_limits[i, 0])
    for i in range(7)
]).reshape(1, 7)

TwMag = np.vstack((Twist, Mag_row))

j_space = Jacobian.GeoJacobianS(TwMag)
print("Geometric Jacobian in Space Frame:\n", j_space)