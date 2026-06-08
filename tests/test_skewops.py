import numpy as np

from poekit.skewops import SkewOps


class TestSkewOps:

    def test_axis_to_skew_shape(self):
        w = np.array([1, 2, 3])

        S = SkewOps.axis_to_skew(w)

        assert S.shape == (3, 3)

    def test_axis_to_skew_known_case(self):
        w = np.array([1, 2, 3])

        expected = np.array([
            [0, -3,  2],
            [3,  0, -1],
            [-2, 1,  0]
        ])

        S = SkewOps.axis_to_skew(w)

        assert np.allclose(S, expected)

    def test_skew_is_skew_symmetric(self):
        w = np.random.rand(3)

        S = SkewOps.axis_to_skew(w)

        assert np.allclose(S.T, -S)

    def test_round_trip_conversion(self):
        w = np.array([4.5, -2.1, 0.8])

        S = SkewOps.axis_to_skew(w)

        recovered = SkewOps.skew_to_axis(S)

        assert np.allclose(w, recovered)

    def test_zero_vector(self):
        w = np.zeros(3)

        S = SkewOps.axis_to_skew(w)

        assert np.allclose(S, np.zeros((3, 3)))

    def test_cross_product_property(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        lhs = np.cross(a, b)

        rhs = SkewOps.axis_to_skew(a) @ b

        assert np.allclose(lhs, rhs)