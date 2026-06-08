import numpy as np
from poekit.rotation_utils import RotationUtils


def test_rotation_angle_identity():

    theta = RotationUtils.rotation_angle(np.eye(3))

    assert np.isclose(theta, 0.0)


def test_rotation_axis_z90():

    R = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    axis = RotationUtils.rotation_axis(
        R,
        np.pi/2
    )

    expected = np.array([[0], [0], [1]])

    assert np.allclose(axis, expected)


def test_rotX_to_tform_shape():

    T = RotationUtils.rotX_to_tform(np.pi/4)

    assert T.shape == (4, 4)


def test_rotY_to_tform_shape():

    T = RotationUtils.rotY_to_tform(np.pi/4)

    assert T.shape == (4, 4)


def test_rotZ_to_tform_shape():

    T = RotationUtils.rotZ_to_tform(np.pi/4)

    assert T.shape == (4, 4)


def test_rotation_determinant():

    T = RotationUtils.rotZ_to_tform(np.pi/3)

    R = T[:3, :3]

    assert np.isclose(
        np.linalg.det(R),
        1.0
    )


def test_rotm_to_axang_identity():

    w, theta = RotationUtils.rotm_2_axang(np.eye(3))

    assert np.isclose(theta, 0.0)
    assert np.allclose(w, np.zeros(3))


def test_quat_to_eulZYX_identity():

    q = np.array([1, 0, 0, 0])

    eul = RotationUtils.quat_to_eulZYX(q)

    assert np.allclose(eul, [0, 0, 0])


def test_eul_to_rotm_identity():

    R = RotationUtils.eul_2_rotm([0, 0, 0])

    assert np.allclose(R, np.eye(3))