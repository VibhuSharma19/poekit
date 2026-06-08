import numpy as np

from poekit.tform_utils import TformUtils


class TestTformUtils:

    def test_translation_vector_to_tform(self):

        p = np.array([1, 2, 3])

        H = TformUtils.trvP_2_tform(p)

        expected = np.eye(4)
        expected[:3, 3] = p

        assert np.allclose(H, expected)

    def test_trvX(self):

        H = TformUtils.trvX_2_tform(5)

        assert np.isclose(H[0, 3], 5)

    def test_trvY(self):

        H = TformUtils.trvY_2_tform(5)

        assert np.isclose(H[1, 3], 5)

    def test_trvZ(self):

        H = TformUtils.trvZ_2_tform(5)

        assert np.isclose(H[2, 3], 5)

    def test_translation_identity_rotation(self):

        p = np.array([1, 2, 3])

        H = TformUtils.trvP_2_tform(p)

        assert np.allclose(
            H[:3, :3],
            np.eye(3)
        )

    def test_adjoint_identity(self):

        H = np.eye(4)

        Ad = TformUtils.tform_2_adjoint(H)

        assert np.allclose(
            Ad,
            np.eye(6)
        )

    def test_adjoint_shape(self):

        H = np.eye(4)

        Ad = TformUtils.tform_2_adjoint(H)

        assert Ad.shape == (6, 6)

    def test_spatial_vector_identity(self):

        H = np.eye(4)

        sp = TformUtils.tform_2_spvec(H)

        assert np.allclose(
            sp,
            np.zeros(6)
        )

    def test_spatial_vector_translation(self):

        H = np.eye(4)
        H[:3, 3] = [1, 2, 3]

        sp = TformUtils.tform_2_spvec(H)

        assert np.allclose(
            sp[3:],
            [1, 2, 3]
        )

    def test_twist_from_identity(self):

        H = np.eye(4)

        tw = TformUtils.tform_2_twist(H)

        assert tw.shape == (6, 1)

    def test_homogeneous_last_row(self):

        H = TformUtils.trvP_2_tform([1, 2, 3])

        assert np.allclose(
            H[3, :],
            [0, 0, 0, 1]
        )