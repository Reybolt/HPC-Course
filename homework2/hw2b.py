

"""
Demonstration module for quadratic interpolation.
Update this docstring to describe your code.
Modified by: Reynaldo Arteaga
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

def quad_interp(xi,yi):
    """
    Quadratic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2.

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==3 and len(yi)==3, error_message

    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(3), xi, xi**2]).T

    ### Fill in this part to compute c ###
    c = solve(A,yi).T
    return c


def test_quad1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2.])
    yi = np.array([ 1., -1.,  7.])
    c = quad_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)


def test_quad2():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  1.])
    yi = np.array([ 1., -1.,  7.])
    c = quad_interp(xi,yi)
    c_true = np.array([-1.,  3.,  5.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

def plot_quad(xi,yi):
    """
    Computes quadratic interpolation with helper function and plots/saves result    ing figure.
    """
    c = quad_interp(xi,yi)
    
    # Plot the resulting polynomial:
    x = np.linspace(xi.min() - 1,  xi.max() + 1, 1000)
    y = c[0] + c[1]*x + c[2]*x**2

    plt.figure(1)       # open plot figure window
    plt.clf()           # clear figure
    plt.plot(x,y,'b-')  # connect points with a blue line

    # Add data points  (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')   # plot as red circles
    plt.ylim(-2,8)         # set limits in y for plot

    plt.title("Data points and interpolating polynomial")

    plt.savefig('quadratic.png')   # save figure as .png file
        
def cubic_interp(xi,yi):
    """
    Cubic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,3.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3.

    """

   # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==4 and len(yi)==4, error_message

    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(4), xi, xi**2, xi**3]).T

    ### Fill in this part to compute c ###
    c = solve(A,yi)
    return c


def test_cubic1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 1.])
    yi = np.array([ 1., -1.,  7., 5.])
    c = cubic_interp(xi,yi)
    c_true = np.array([-1.,  4.,  4., -2. ])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)


def plot_cubic(xi,yi):
    """
    Computes cubic interpolation with helper function and plots/saves result        ing figure.
    """
    c = cubic_interp(xi,yi)
    
    # Plot the resulting polynomial:
    x = np.linspace(xi.min() - 1,  xi.max() + 1, 1000)
    y = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3

    plt.figure(1)       # open plot figure window
    plt.clf()           # clear figure
    plt.plot(x,y,'b-')  # connect points with a blue line

    # Add data points  (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')   # plot as red circles
    plt.ylim(-2,8)         # set limits in y for plot

    plt.title("Data points and interpolating polynomial")

    plt.savefig('cubic.png')   # save figure as .png file
        
        
def poly_interp(xi,yi):
    """
    (N-1)-st order interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,...,n-1.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + ... + c[n-1]*x**(n-1).

    """

   # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==len(yi), error_message

    # Set up linear system to interpolate through data points:
    A = np.vander(xi)

    ### Fill in this part to compute c ###
    c = solve(A,yi)
    c = c[::-1]
    return c


def test_poly1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 1.])
    yi = np.array([ 1., -1.,  7., 5.])
    c = poly_interp(xi,yi)
    c_true = np.array([-1.,  4.,  4., -2. ])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)


def test_poly2():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 1., -2.])
    yi = np.array([ 1., -1.,  7., 5., -7.])
    c = poly_interp(xi,yi)
    c_true = np.array([-1.,  1.5,  5.25, 0.5, -1.25 ])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)


def plot_poly(xi,yi):
    """
    Computes polynomial interpolation with helper function and plots/saves resul    ting figure.
    """
    c = poly_interp(xi,yi)
    
    # Plot the resulting polynomial:
    x = np.linspace(xi.min() - 1,  xi.max() + 1, 1000)
    n = c.size
    y = c[n -1]
    for j in range(n-1,0,-1):
        y = y*x + c[j-1]
        

    plt.figure(1)       # open plot figure window
    plt.clf()           # clear figure
    plt.plot(x,y,'b-')  # connect points with a blue line

    # Add data points  (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')   # plot as red circles
    plt.ylim(-2,8)         # set limits in y for plot

    plt.title("Data points and interpolating polynomial")

    plt.savefig('polynomial.png')   # save figure as .png file
        
