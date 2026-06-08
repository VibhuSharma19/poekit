import numpy as np

from poekit.utils import Utils


class TestUtils:

    # =====================================================
    # jointmag_2_limits
    # =====================================================

    def test_joint_limit_inside_range(self):

        q = Utils.jointmag_2_limits(
            TheTheory=0.5,
            LimitPos=1.0,
            LimitNeg=-1.0
        )

        assert np.isclose(q, 0.5)

    def test_joint_limit_upper_saturation(self):

        q = Utils.jointmag_2_limits(
            TheTheory=10.0,
            LimitPos=1.0,
            LimitNeg=-1.0
        )

        assert np.isclose(q, 1.0)

    def test_joint_limit_lower_saturation(self):

        q = Utils.jointmag_2_limits(
            TheTheory=-10.0,
            LimitPos=1.0,
            LimitNeg=-1.0
        )

        assert np.isclose(q, -1.0)

    # =====================================================
    # spavec_2_crf
    # =====================================================

    def test_crf_shape(self):

        v = np.arange(1, 7)

        crf = Utils.spavec_2_crf(v)

        assert crf.shape == (6, 6)

    def test_crf_zero_vector(self):

        crf = Utils.spavec_2_crf(np.zeros(6))

        assert np.allclose(crf, np.zeros((6, 6)))

    # =====================================================
    # spavec_2_crm
    # =====================================================

    def test_crm_shape(self):

        v = np.arange(1, 7)

        crm = Utils.spavec_2_crm(v)

        assert crm.shape == (6, 6)

    def test_crm_zero_vector(self):

        crm = Utils.spavec_2_crm(np.zeros(6))

        assert np.allclose(crm, np.zeros((6, 6)))

    # =====================================================
    # Aij_2_adjoint
    # =====================================================

    def test_Aij_identity(self):

        TwMag = np.zeros((7, 3))

        A = Utils.Aij_2_adjoint(
            1,
            1,
            TwMag
        )

        assert np.allclose(A, np.eye(6))

    def test_Aij_zero_case(self):

        TwMag = np.zeros((7, 3))

        A = Utils.Aij_2_adjoint(
            1,
            2,
            TwMag
        )

        assert np.allclose(A, np.zeros((6, 6)))

    # =====================================================
    # intersect_lines3D
    # =====================================================

    def test_intersect_lines_known_case(self):

        axes = np.array([
            [1, 0],
            [0, 1],
            [0, 0]
        ])

        points = np.array([
            [0, 0],
            [0, 0],
            [0, 0]
        ])

        p = Utils.intersect_lines3D(
            axes,
            points
        )

        expected = np.array([0, 0, 0])

        assert np.allclose(p, expected)

    def test_intersect_parallel_lines(self):

        axes = np.array([
            [1, 1],
            [0, 0],
            [0, 0]
        ])

        points = np.array([
            [0, 0],
            [0, 1],
            [0, 0]
        ])

        p = Utils.intersect_lines3D(
            axes,
            points
        )

        assert np.all(np.isinf(p))

    # =====================================================
    # minus_poseEul
    # =====================================================

    def test_minus_pose_identity(self):

        pose = np.zeros(6)

        diff = Utils.minus_poseEul(
            pose,
            pose
        )

        assert np.allclose(
            diff,
            np.zeros(6)
        )

    def test_minus_pose_translation_only(self):

        pose1 = np.zeros(6)

        pose2 = np.array([
            1,
            2,
            3,
            0,
            0,
            0
        ])

        diff = Utils.minus_poseEul(
            pose1,
            pose2
        )

        expected = np.array([
            1,
            2,
            3,
            0,
            0,
            0
        ])

        assert np.allclose(
            diff,
            expected,
            atol=1e-6
        )

    # =====================================================
    # project_perp
    # =====================================================

    def test_project_perp_x_to_yz_plane(self):

        v = np.array([
            1,
            2,
            3
        ])

        w = np.array([
            1,
            0,
            0
        ])

        p = Utils.project_perp(v, w)

        expected = np.array([
            [0],
            [2],
            [3]
        ])

        assert np.allclose(
            p,
            expected
        )

    def test_project_perp_parallel_vector(self):

        v = np.array([
            1,
            0,
            0
        ])

        w = np.array([
            1,
            0,
            0
        ])

        p = Utils.project_perp(v, w)

        assert np.allclose(
            p,
            np.zeros((3, 1))
        )

    def test_project_perp_orthogonal_vector(self):

        v = np.array([
            0,
            1,
            0
        ])

        w = np.array([
            1,
            0,
            0
        ])

        p = Utils.project_perp(v, w)

        assert np.allclose(
            p,
            v.reshape(3, 1)
        )

    def test_projected_vector_is_perpendicular(self):

        v = np.array([
            3,
            4,
            5
        ])

        w = np.array([
            1,
            2,
            3
        ])

        p = Utils.project_perp(v, w)

        dot_product = (
            p.T @ (
                w.reshape(3, 1)
                / np.linalg.norm(w)
            )
        )

        assert np.isclose(
            dot_product,
            0,
            atol=1e-6
        )

    def test_crf_crm_duality(self):

        v = np.random.rand(6)

        crm = Utils.spavec_2_crm(v)
        crf = Utils.spavec_2_crf(v)

        assert np.allclose(
            crf,
            -crm.T
        )

    def test_projection_idempotent(self):

        v = np.random.rand(3)
        w = np.array([1, 2, 3])

        p1 = Utils.project_perp(v, w)
        p2 = Utils.project_perp(p1.flatten(), w)

        assert np.allclose(
            p1,
            p2
        )