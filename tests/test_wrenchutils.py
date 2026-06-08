import numpy as np

from poekit.wrench_utils import WrenchUtils


class TestWrenchUtils:

    def test_rotational_wrench(self):
        """
        Pure moment wrench.
        """

        axis = np.array([0, 0, 1])
        point = np.array([0, 0, 0])

        phi = WrenchUtils.link_2_wrench(
            axis,
            point,
            "rot"
        )

        expected = np.array([
            0, 0, 0,
            0, 0, 1
        ])

        assert np.allclose(phi, expected)

    def test_translational_wrench(self):
        """
        Force applied at offset point.
        """

        force = np.array([1, 0, 0])
        point = np.array([0, 1, 0])

        phi = WrenchUtils.link_2_wrench(
            force,
            point,
            "tra"
        )

        expected_moment = -np.cross(force, point)

        assert np.allclose(phi[:3], force)
        assert np.allclose(phi[3:], expected_moment)

    def test_invalid_force_type(self):

        phi = WrenchUtils.link_2_wrench(
            [1, 0, 0],
            [0, 0, 0],
            "bad"
        )

        assert np.allclose(phi, np.zeros(6))

    def test_wrench_axis_pure_moment(self):

        phi = np.array([
            0, 0, 0,
            0, 0, 5
        ])

        axis = WrenchUtils.wrench_axis(phi)

        direction = axis[:, 1]

        expected = np.array([0, 0, 1])

        assert np.allclose(direction, expected)

    def test_wrench_axis_pure_force(self):

        phi = np.array([
            3, 0, 0,
            0, 0, 0
        ])

        axis = WrenchUtils.wrench_axis(phi)

        direction = axis[:, 1]

        expected = np.array([1, 0, 0])

        assert np.allclose(direction, expected)

    def test_wrench_axis_shape(self):

        phi = np.array([
            1, 2, 3,
            4, 5, 6
        ])

        axis = WrenchUtils.wrench_axis(phi)

        assert axis.shape == (3, 2)

    def test_magnitude_force(self):

        phi = np.array([
            3, 4, 0,
            0, 0, 0
        ])

        mag = WrenchUtils.wrench_magnitude(phi)

        assert np.isclose(mag, 5)

    def test_magnitude_moment(self):

        phi = np.array([
            0, 0, 0,
            0, 0, 5
        ])

        mag = WrenchUtils.wrench_magnitude(phi)

        assert np.isclose(mag, 5)

    def test_pitch_pure_force(self):

        phi = np.array([
            1, 0, 0,
            0, 0, 0
        ])

        pitch = WrenchUtils.wrench_pitch(phi)

        assert np.isclose(pitch, 0)

    def test_pitch_pure_moment(self):

        phi = np.array([
            0, 0, 0,
            0, 0, 1
        ])

        pitch = WrenchUtils.wrench_pitch(phi)

        assert pitch == np.inf

    def test_pitch_general_case(self):

        phi = np.array([
            1, 0, 0,
            2, 0, 0
        ])

        pitch = WrenchUtils.wrench_pitch(phi)

        assert np.isclose(pitch, 2)

    def test_force_wrench_shape(self):

        phi = WrenchUtils.link_2_wrench(
            [1, 0, 0],
            [0, 0, 0],
            "tra"
        )

        assert phi.shape == (6,)

    def test_line_of_action_consistency(self):

        force = np.array([0, 0, 1])
        point = np.array([1, 0, 0])

        phi = WrenchUtils.link_2_wrench(
            force,
            point,
            "tra"
        )

        axis = WrenchUtils.wrench_axis(phi)

        direction = axis[:, 1]

        assert np.allclose(
            direction,
            force / np.linalg.norm(force)
        )

    def test_pitch_scaling_invariance(self):

        phi = np.array([
            1, 0, 0,
            2, 0, 0
        ])

        p1 = WrenchUtils.wrench_pitch(phi)
        p2 = WrenchUtils.wrench_pitch(5 * phi)

        assert np.isclose(p1, p2)