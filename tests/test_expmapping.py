import numpy as np

from poekit.expmapping import ExpMapping


class TestExpMapping:

    def test_exp_axang_identity(self):
        """Zero rotation should produce identity."""

        axang = np.array([0, 0, 1, 0])

        R = ExpMapping.exp_axang(axang)

        assert np.allclose(R, np.eye(3))


    def test_exp_axang_z90(self):
        """90 deg rotation about Z."""

        axang = np.array([0, 0, 1, np.pi / 2])

        R = ExpMapping.exp_axang(axang)

        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])

        assert np.allclose(R, expected, atol=1e-6)


    def test_exp_axang_orthogonality(self):
        """R^T R = I"""

        axang = np.array([1, 0, 0, np.pi / 3])

        R = ExpMapping.exp_axang(axang)

        assert np.allclose(
            R.T @ R,
            np.eye(3),
            atol=1e-6
        )


    def test_exp_axang_det_one(self):
        """det(R)=1"""

        axang = np.array([0, 1, 0, np.pi / 5])

        R = ExpMapping.exp_axang(axang)

        assert np.isclose(
            np.linalg.det(R),
            1.0,
            atol=1e-6
        )


    def test_exp_screw_pure_translation_x(self):

        TwMag = np.array([
            1, 0, 0,
            0, 0, 0,
            2
        ])

        H = ExpMapping.exp_screw(TwMag)

        expected = np.eye(4)
        expected[0, 3] = 2

        assert np.allclose(H, expected)


    def test_exp_screw_pure_translation_y(self):

        TwMag = np.array([
            0, 1, 0,
            0, 0, 0,
            3
        ])

        H = ExpMapping.exp_screw(TwMag)

        assert np.isclose(H[1, 3], 3)


    def test_exp_screw_shape(self):

        TwMag = np.array([
            0, 0, 0,
            0, 0, 1,
            np.pi / 2
        ])

        H = ExpMapping.exp_screw(TwMag)

        assert H.shape == (4, 4)


    def test_exp_screw_bottom_row(self):

        TwMag = np.array([
            0, 0, 0,
            0, 0, 1,
            np.pi / 4
        ])

        H = ExpMapping.exp_screw(TwMag)

        assert np.allclose(
            H[3, :],
            [0, 0, 0, 1]
        )


    def test_exp_screw_rotation_part_valid(self):

        TwMag = np.array([
            0, 0, 0,
            0, 0, 1,
            np.pi / 3
        ])

        H = ExpMapping.exp_screw(TwMag)

        R = H[:3, :3]

        assert np.isclose(
            np.linalg.det(R),
            1.0,
            atol=1e-6
        )


    def test_exp_screw_zero_motion(self):
        """
        exp([S]0)=I
        """

        TwMag = np.array([
            0, 0, 0,
            0, 0, 1,
            0
        ])

        H = ExpMapping.exp_screw(TwMag)

        assert np.allclose(
            H,
            np.eye(4)
        )