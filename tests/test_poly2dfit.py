import numpy as np
import numpy.testing as nt
import poly2d.poly2dfit as pf
import pytest


def explicit_poly1(x, y):
    return x**5 + 3*x**4*y**3 - 2*x*y**3 + 7

def explicit_poly2(x, y):
    return x*2*y*3 - x*y**4

def explicit_poly3(x, y):
    return x*2*y*3 - x*y**4 + 1e-5*x +1e-5*y


RNG = np.random.default_rng(seed=0)


def test_Poly2D():
    A = np.zeros((6, 4))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    Pa = pf.Poly2D(A)

    x, y = RNG.random([2, 10])

    result = Pa(x, y)

    desired = explicit_poly1(x, y)

    nt.assert_allclose(result, desired)


def test_Poly2D_fit():
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 50
    x_test, y_test = 100*RNG.random(size=(2, n_samples)) - 50

    z_s = explicit_poly2(x_sample, y_sample)
    z_t = explicit_poly2(x_test, y_test)

    poly = pf.Poly2D.fit(x_sample, y_sample, z_s, nx=4, ny=5)

    z_f = poly(x_test, y_test)

    nt.assert_allclose(z_f, z_t)


def test_Poly2D_fit_zero_cc_constraint():
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 5
    x_test, y_test = 100*RNG.random(size=(2, n_samples)) - 5

    z_s = explicit_poly2(x_sample, y_sample)
    z_t = explicit_poly2(x_test, y_test)

    poly = pf.Poly2D.fit(x_sample, y_sample, z_s, nx=4, ny=5, constraint="zero_cc")

    z_f = poly(x_test, y_test)

    nt.assert_allclose(z_f, z_t)

    assert poly.c[0, 0] == 0
    assert poly(0, 0) == 0


def test_Poly2D_fit_zero_grad_constraint():
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 5
    x_test, y_test = 100*RNG.random(size=(2, n_samples)) - 5

    z_s = explicit_poly3(x_sample, y_sample)
    z_t = explicit_poly3(x_test, y_test)

    poly = pf.Poly2D.fit(x_sample, y_sample, z_s, nx=4, ny=5, constraint="zero_grad")

    z_f = poly(x_test, y_test)

    nt.assert_allclose(z_f, z_t, rtol=1e-5, atol=1e-5)

    assert poly.der_x(1)(0, 0) == 0
    assert poly.der_y(1)(0, 0) == 0


def test_Poly2D_fit_unknown_constraint():
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 5

    z_s = explicit_poly3(x_sample, y_sample)

    with pytest.raises(pf.Poly2D.UnknownFitConstraintOption):
        pf.Poly2D.fit(x_sample,
                      y_sample,
                      z_s,
                      nx=4,
                      ny=5,
                      constraint="unknown_constraint")


#  def test_Poly2D_fit_few_samples():
    #  n_samples = 3
    #  x_sample, y_sample = 10*RNG.random(size=(2, n_samples)) - 5
    #  x_test, y_test = 10*RNG.random(size=(2, n_samples)) - 5

    #  z_s = explicit_poly2(x_sample, y_sample)
    #  z_t = explicit_poly2(x_test, y_test)

    #  poly = pf.Poly2D.fit(x_sample, y_sample, z_s, nx=4, ny=5)

    #  z_f = poly(x_test, y_test)

    #  nt.assert_allclose(z_f, z_t)

    #  assert False


