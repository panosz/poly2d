"""
Tests for pol2dfit module.
"""
import numpy as np
import numpy.testing as nt
import pytest
import poly2d.poly2dfit as pf


def explicit_poly1(x, y):
    return x**5 + 3*x**4*y**3 - 2*x*y**3 + 7


def derivative_x2_y2_explicit_poly1(x, y):
    return 216*x**2*y


def derivative_x3_explicit_poly1(x, y):
    return 60*x**2 + 72*x*y**3


def derivative_y2_explicit_poly1(x, y):
    return -12*x*y + 18*x**4*y


def explicit_poly2(x, y):
    return x*2*y*3 - x*y**4


def explicit_poly3(x, y):
    return x*2*y*3 - x*y**4 + 1e-5*x + 1e-5*y


RNG = np.random.default_rng(seed=0)


def test_Poly2D_knows_degree():
    Nx = 5
    Ny = 3

    A = np.zeros((Nx+1, Ny+1))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    Pa = pf.Poly2D(A)

    assert Pa.degree == (Pa.nx, Pa.ny)

    assert Pa.nx == 5
    assert Pa.ny == 3


def test_Poly2D_calculation():
    Nx = 5
    Ny = 3

    A = np.zeros((Nx+1, Ny+1))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    Pa = pf.Poly2D(A)

    x, y = RNG.random([2, 10])

    result = Pa(x, y)

    desired = explicit_poly1(x, y)

    nt.assert_allclose(result, desired)


def test_Poly2D_derivative_correct_degree():
    Nx = 5
    Ny = 3

    n_dx = 2
    n_dy = 2

    A = np.zeros((Nx+1, Ny+1))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    Pa = pf.Poly2D(A)

    dPa = Pa.der(n_dx, n_dy)

    assert dPa.degree == (Nx - n_dx, Ny - n_dy)


def test_Poly2D_multiderivative_calculation():
    Nx = 5
    Ny = 3

    A = np.zeros((Nx+1, Ny+1))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    x, y = RNG.random([2, 10])

    dPa = pf.Poly2D(A).der(2, 2)
    result = dPa(x, y)
    desired = derivative_x2_y2_explicit_poly1(x, y)

    nt.assert_allclose(result, desired)


def test_Poly2D_x_derivative_calculation():
    Nx = 5
    Ny = 3

    A = np.zeros((Nx+1, Ny+1))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    x, y = RNG.random([2, 10])

    dPa = pf.Poly2D(A).der_x(3)
    result = dPa(x, y)
    desired = derivative_x3_explicit_poly1(x, y)

    nt.assert_allclose(result, desired)


def test_Poly2D_y_derivative_calculation():
    Nx = 5
    Ny = 3

    A = np.zeros((Nx+1, Ny+1))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    x, y = RNG.random([2, 10])

    dPa = pf.Poly2D(A).der_y(2)
    result = dPa(x, y)
    desired = derivative_y2_explicit_poly1(x, y)

    nt.assert_allclose(result, desired)


def test_Poly2D_shifted():
    Nx = 5
    Ny = 3

    A = np.zeros((Nx+1, Ny+1))
    A[5, 0] = 1
    A[4, 3] = 3
    A[1, 3] = -2
    A[0, 0] = 7

    x0 = 3
    y0 = -2

    Pa = pf.Poly2D(A, center=(x0, y0))

    x, y = RNG.random([2, 10])

    result = Pa(x, y)

    desired = explicit_poly1(x - x0, y - y0)

    nt.assert_allclose(result, desired)


def test_Poly2D_fit():
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 50
    x_test, y_test = 100*RNG.random(size=(2, n_samples)) - 50

    z_s = explicit_poly2(x_sample, y_sample)
    z_t = explicit_poly2(x_test, y_test)

    poly = pf.Poly2D.fit(x_sample, y_sample, z_s, degree=(4, 5))

    z_f = poly(x_test, y_test)

    nt.assert_allclose(z_f, z_t)


def test_Poly2D_fit_shifted():
    x0, y0 = -4, 3
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 50
    x_test, y_test = 100*RNG.random(size=(2, n_samples)) - 50

    z_s = explicit_poly2(x_sample, y_sample)
    z_t = explicit_poly2(x_test, y_test)

    poly = pf.Poly2D.fit(x_sample,
                         y_sample,
                         z_s,
                         degree=(4, 5),
                         center=(x0, y0),
                         )

    z_f = poly(x_test, y_test)

    nt.assert_allclose(z_f, z_t)


def test_Poly2D_fit_zero_cc_constraint():
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 5
    x_test, y_test = 100*RNG.random(size=(2, n_samples)) - 5

    z_s = explicit_poly2(x_sample, y_sample)
    z_t = explicit_poly2(x_test, y_test)

    poly = pf.Poly2D.fit(x_sample,
                         y_sample,
                         z_s,
                         degree=(4, 5),
                         constraint="zero_cc")

    z_f = poly(x_test, y_test)

    nt.assert_allclose(z_f, z_t, rtol=1e-4, atol=1e-4)

    assert poly.c[0, 0] == 0
    assert poly(0, 0) == 0


def test_Poly2D_fit_zero_grad_constraint():
    n_samples = 100
    x_sample, y_sample = 100*RNG.random(size=(2, n_samples)) - 5
    x_test, y_test = 100*RNG.random(size=(2, n_samples)) - 5

    z_s = explicit_poly3(x_sample, y_sample)
    z_t = explicit_poly3(x_test, y_test)

    poly = pf.Poly2D.fit(x_sample,
                         y_sample,
                         z_s,
                         degree=(4, 5),
                         constraint="zero_grad")

    z_f = poly(x_test, y_test)

    nt.assert_allclose(z_f, z_t, rtol=1e-4, atol=1e-4)

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
                      degree=(4, 5),
                      constraint="unknown_constraint")
