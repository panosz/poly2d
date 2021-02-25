import numpy as np
import numpy.testing as nt
import poly2d.poly2dfit as pf


def test_Poly2D():
    A = np.zeros((6, 4))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    Pa = pf.Poly2D(A)

    def explicit_poly(x, y):
        return x**5 + 3*x**4*y**3 - 2*x*y**3 + 7

    rng = np.random.default_rng(seed=0)
    x, y = rng.random([2, 10])

    result = Pa(x, y)

    desired = explicit_poly(x, y)

    nt.assert_allclose(result, desired)
