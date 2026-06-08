<div align="center">

<img src="https://raw.githubusercontent.com/VibhuSharma19/poekit/main/icon/logo.png" width="150" height="150" />

#  poekit

![PyPI](https://img.shields.io/pypi/v/poekit) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/poekit?logo=python) ![License](https://img.shields.io/pypi/l/poekit)
![Research](https://img.shields.io/badge/Research-Robotics-success) ![POE](https://img.shields.io/badge/Formulation-POE-orange) ![Screw%20Theory](https://img.shields.io/badge/Theory-Screw%20Theory-purple) ![Geometry](https://img.shields.io/badge/Geometry-SE(3)%20%7C%20SO(3)-blue)

## Screw Theory Toolkit for Python

#### 🌀 Product of Exponentials (POE) • 🔩 Screw Theory • 🌐 SE(3) • 📐 Robot Kinematics

Built for **Robotics Research, Education, and Simulation**

🔬 Inspired and adapted from the **ST24R MATLAB Toolbox**

---

### T = exp([S]θ) M

</div>

---

## 🔍 Overview

**poekit** is a professional Python toolkit for robot modeling and analysis based on the **Product of Exponentials (POE)** formulation, **Screw Theory**, and rigid-body motion in **SE(3)**. Designed for researchers, students, and robotics engineers, it provides modern geometric tools for kinematics, dynamics, and motion analysis.

The package includes comprehensive utilities for twists, wrenches, Jacobians, rigid-body transformations, and robot dynamics, enabling rapid development, education, simulation, and robotics research.

---

## ✨ Features

- 🔩 Screw Theory Operations
- 🌀 Twist Algebra & Motion Operations
- ⚙️ Wrench & Force-Moment Operations
- 🌐 SE(3) Rigid-Body Transformations
- 🧭 SO(3) Rotation Representations
- 🤖 Robot Kinematics & Dynamics
- 📐 Spatial and Body Jacobians
- 🎯 Paden–Kahan Inverse Kinematics Subproblems

---

## 📦 Installation

### PyPI

```bash
pip install poekit
```

### From Source

```bash
git clone https://github.com/VibhuSharma19/poekit.git

cd poekit

pip install -e .
```

---

## 🚀 Quick Example

```python
from poekit.forward_kine import ForwardKine

T = ForwardKine.forward_kinematics_poe(TwMag)

print(T)
```

---

## 📂 Project Structure

```bash
poekit
│
├── examples
│   ├── jaco2_fk.py                       # Example: Forward kinematics analysis of the Kinova Jaco2 manipulator
│   ├── jaco2_ik.py                       # Example: Inverse kinematics analysis of the Kinova Jaco2 manipulator
│   └── jaco2_jacobian.py                 # Example: Jacobian computation and analysis of the Kinova Jaco2 manipulator
│
└── src
    └── poekit
        ├── coriolis_utils.py             # Coriolis and centrifugal effects in robot dynamics
        ├── expmapping.py                 # Exponential map and Lie group operations
        ├── forward_kine.py               # Forward kinematics using the Product of Exponentials formulation
        ├── inertia_utils.py              # Inertia and mass matrix computation
        ├── jacobian.py                   # Geometric Jacobian computation in spatial and body forms
        ├── paden_kahan.py                # Paden–Kahan subproblem solvers
        ├── potential_utils.py            # Potential energy and gravity vector computation
        ├── rotation_utils.py             # SO(3) rotation representations and operators
        ├── skewops.py                    # Skew-symmetric matrix operators
        ├── tform_utils.py                # SE(3) rigid-body transformation utilities
        ├── twist_utils.py                # Twist algebra and screw motion utilities
        ├── utils.py                      # Common geometric and mathematical helper functions
        └── wrench_utils.py               # Wrench algebra and force-moment utilities
```

---

## 🧠 Mathematical Foundations

**poekit** is built around:

- 🌀 Product of Exponentials (POE)
- 🔩 Screw Theory
- 🌐 SO(3) & SE(3)
- 🌀 Twists & ⚙️ Wrenches
- 🔄 Adjoint Transformations
- 📐 Geometric Jacobians
- 🎯 Paden–Kahan Subproblems

---

## 🎓 Core Concepts

```text
Twist            ξ = (v, ω)

Screw Axis       ω

Exponential Map  e^[Sθ]

Rigid Motion     T ∈ SE(3)

Adjoint          Ad(T)

Jacobian         J
```

---

## 📚 References

### 📖 Primary Literature

**José M. Pardos-Gotor**

*Screw Theory in Robotics: An Illustrated and Practicable Introduction to Modern Mechanics*

### 🔗 Software Reference

[**ST24R (STAR) MATLAB Toolbox**](https://github.com/DrPardosGotor/Screw-Theory-in-Robotics)

### 🤖 Robot Platform

**Kinova Jaco2 7-DOF Manipulator**

---

## 🏗️ Relationship to ST24R

This package contains Python implementations derived from concepts and algorithms presented in the **ST24R (STAR) MATLAB Screw Theory Toolbox**, developed by **Dr. José M. Pardos-Gotor**.

The software architecture has been redesigned for Python while preserving the underlying mathematical formulations and educational philosophy of the original toolbox.

---

## 📜 License

GNU Lesser General Public License v3.0 (**LGPL-3.0**)

See:

- [`LICENSE`](LICENSE)
- [`COPYING.LESSER`](COPYING.LESSER)

for complete licensing information.

---

## 👤 Author

**Vibhu Sharma**

🎓 M.Tech (Gold Medalist) — Automation and Robotics

🔗 GitHub: https://github.com/VibhuSharma19

📧 Email: 1999vibhusharma@gmail.com

---

<div align="center">

##  Geometry. Motion. Robotics.

*Modern Screw Theory and Product of Exponentials tools for Python.*

</div>
