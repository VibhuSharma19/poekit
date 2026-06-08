import numpy as np

from poekit.twist_utils import TwistUtils


class TestTwistUtils:

    def test_joint_to_twist_revolute_origin(self):

        axis = np.array([0, 0, 1])
        point = np.array([0, 0, 0])

        tw = TwistUtils.joint_2_twist(
            axis,
            point,
            "rot"
        )

        expected = np.array([
            0, 0, 0,
            0, 0, 1
        ])

        assert np.allclose(tw, expected)


    def test_joint_to_twist_prismatic(self):

        axis = np.array([1, 0, 0])

        tw = TwistUtils.joint_2_twist(
            axis,
            np.zeros(3),
            "prism"
        )

        expected = np.array([
            1, 0, 0,
            0, 0, 0
        ])

        assert np.allclose(tw, expected)


    def test_joint_to_twist_invalid_type(self):

        tw = TwistUtils.joint_2_twist(
            [1, 0, 0],
            [0, 0, 0],
            "invalid"
        )

        assert np.allclose(
            tw,
            np.zeros(6)
        )


    def test_twist_magnitude_revolute(self):

        tw = np.array([
            0, 0, 0,
            0, 0, 2
        ])

        mag = TwistUtils.twist_magnitude(tw)

        assert np.isclose(mag, 2)


    def test_twist_magnitude_prismatic(self):

        tw = np.array([
            3, 4, 0,
            0, 0, 0
        ])

        mag = TwistUtils.twist_magnitude(tw)

        assert np.isclose(mag, 5)


    def test_twist_pitch_pure_rotation(self):

        tw = np.array([
            0, 0, 0,
            0, 0, 1
        ])

        pitch = TwistUtils.twist_pitch(tw)

        assert np.isclose(
            pitch,
            0.0
        )


    def test_twist_pitch_prismatic(self):

        tw = np.array([
            1, 0, 0,
            0, 0, 0
        ])

        pitch = TwistUtils.twist_pitch(tw)

        assert np.isinf(pitch)


    def test_twist_axis_revolute_shape(self):

        tw = np.array([
            0, 0, 0,
            0, 0, 1
        ])

        axis = TwistUtils.twist_axis(tw)

        assert axis.shape == (3, 2)


    def test_twist_axis_prismatic_shape(self):

        tw = np.array([
            1, 0, 0,
            0, 0, 0
        ])

        axis = TwistUtils.twist_axis(tw)

        assert axis.shape == (3, 2)


    def test_twist_to_joint_revolute(self):

        tw = np.array([
            0, 0, 0,
            0, 0, 1
        ])

        axis, q, joint = TwistUtils.twist_2_joint(tw)

        assert joint == "rot"

        assert np.allclose(
            axis,
            [0, 0, 1]
        )


    def test_twist_to_joint_prismatic(self):

        tw = np.array([
            1, 0, 0,
            0, 0, 0
        ])

        axis, q, joint = TwistUtils.twist_2_joint(tw)

        assert joint == "tra"


    def test_twist_to_tform_shape(self):

        tw = np.array([
            1, 2, 3,
            4, 5, 6
        ])

        H = TwistUtils.twist_2_tform(tw)

        assert H.shape == (4, 4)


    def test_twist_to_tform_bottom_row(self):

        tw = np.array([
            1, 2, 3,
            4, 5, 6
        ])

        H = TwistUtils.twist_2_tform(tw)

        assert np.allclose(
            H[3, :],
            [0, 0, 0, 1]
        )


    def test_twist_bracket_self_zero(self):

        tw = np.array([
            1, 2, 3,
            0, 0, 1
        ])

        bracket = TwistUtils.twist_bracket(
            tw,
            tw
        )

        assert np.allclose(
            bracket,
            np.zeros((6, 1))
        )


    def test_twist_bracket_shape(self):

        tw1 = np.array([
            1, 0, 0,
            0, 0, 1
        ])

        tw2 = np.array([
            0, 1, 0,
            0, 1, 0
        ])

        bracket = TwistUtils.twist_bracket(
            tw1,
            tw2
        )

        assert bracket.shape == (6, 1)

    def test_motion_to_twist_identity(self):

        H = np.eye(4)

        xi = TwistUtils.motion_2_twist(H)

        expected = np.zeros(6)

        assert np.allclose(
            xi.flatten(),
            expected
        )


    def test_motion_to_twist_translation(self):

        H = np.eye(4)
        H[:3,3] = [1,0,0]
    
        xi = TwistUtils.motion_2_twist(H)
    
        expected = np.array([
            1,0,0,
            0,0,0
        ])
    
        assert np.allclose(
            xi.flatten(),
            expected
        )