import scipy.interpolate
import scipy.special as sc
import scipy.integrate as ig
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EPSILON = 0.0000000000005
UPPER_LIMIT = 3 + EPSILON
LOWER_LIMIT = 3 - EPSILON


def pearson_function(x, beta, sigma):
    """ Calculation of the Pearson Function

    :param x:
    :param beta: kurtoses
    :param sigma: standard deviation

    :type x: float
    :type beta: float
    :type sigma: float

    :return: the probability value
    :rtype: float
    """
    if beta < 2:
        return 0
    elif beta < LOWER_LIMIT:
        m = (5 * beta - 9) / (6 - 2 * beta)
        a = np.sqrt(2 * m + 3) * sigma
        beta_func = sc.beta(0.5, m + 0.5)
        fract = (x / a) ** 2
        san_fract = np.where(np.greater(fract, 1), 1, fract)
        f = ((m + 0.5) / (np.pi * a)) * beta_func * ((1 - san_fract) ** m)
    elif np.abs(beta - 3) < EPSILON:
        # maybe use a constant for sqrt(2) and sqrt(pi)
        a = np.sqrt(2) * sigma
        f = (1 / (np.sqrt(np.pi) * a)) * np.exp(-(x / a) ** 2)
    else:
        m = (5 * beta - 9) / (6 - 2 * beta)
        a = np.sqrt(np.abs(2 * m + 3)) * sigma
        beta_func = sc.beta(0.5, np.abs(m))
        f = (np.abs(m + 0.5) / (np.pi * a)) * beta_func * (
                    (1 + (x / a) ** 2) ** m)
    return f


def pearson_function_fast(x, beta, sigma):
    """ Calculation of the Pearson Function with whole-array operations

    :param x:
    :param beta: kurtosis
    :param sigma: standard deviation

    :type x: ndarray
    :type beta: ndarray
    :type sigma: ndarray

    :return: the probability values
    :rtype: ndarray
    """
    beta_between_2_3 = np.logical_and(np.greater_equal(beta, 2),
                                      np.less(beta, LOWER_LIMIT))
    beta_eq_3 = np.less(np.abs(beta - 3), EPSILON)
    beta_gt_3 = np.greater(beta, UPPER_LIMIT)

    # not the most beautiful solution
    beta_san = np.where(np.equal(beta, 3), 0, beta)
    m = np.where(beta_eq_3, 0, (5 * beta_san - 9) / (6 - 2 * beta_san))

    a = np.where(beta_eq_3, np.sqrt(2) * sigma,
                 sigma * np.sqrt(np.where(
                     beta_gt_3,
                     np.abs(2 * m + 3),
                     2 * m + 3))
                 )

    beta_func = np.where(beta_between_2_3, sc.beta(0.5, m + 0.5),
                         np.where(beta_gt_3, sc.beta(0.5, np.abs(m)), 0))

    fract = (x / a) ** 2

    san_fract = np.where(beta_between_2_3,
                         np.where(np.greater(fract, 1), 1, fract), fract)

    with np.errstate(invalid='ignore'):
        f = np.where(beta_between_2_3,
                     ((m + 0.5) / (np.pi * a))
                     * beta_func * ((1 - san_fract) ** m),
                     np.where(beta_eq_3, (1 / ((np.pi ** 2) * a))
                              * np.exp(-san_fract),
                              np.where(beta_gt_3,
                                       (np.abs(m + 0.5) / (np.pi * a))
                                       * beta_func * ((1 + san_fract) ** m),
                                       np.NaN)))

    return f


def test_pearson_function():
    """Tests if the pearson function fulfills all criteria"""
    sigma_test = lambda x, b1, s1: x ** 2 * pearson_function(x, b1, s1)
    beta_test = lambda x, b2, s2: x ** 4 * pearson_function(x, b2, s2)
    for sigma in range(1, 10):
        for beta in range(2, 10):
            test1 = ig.quad(pearson_function, -np.inf, np.inf, (beta, sigma))
            test2 = ig.quad(sigma_test, -np.inf, np.inf, (beta, sigma))
            test3 = ig.quad(beta_test, -np.inf, np.inf, (beta, sigma))

            print("beta: " + str(beta) + "  sigma: " + str(sigma))
            print("should be 1: " + str(test1[0])
                  + " deviation: " + str(test1[1]))
            print("should be " + str(sigma ** 2) + ": " + str(test2[0])
                  + " deviation: " + str(test2[1]))
            print("should be " + str(beta) + ": " +
                  str(test3[0] / (sigma ** 4))
                  + " deviation: " + str(test3[1] / sigma ** 4))

            assert 1 >= (test1[0] - test1[1]) or (test1[0] + test1[1]) >= 1
            assert sigma ** 2 >= (test2[0] - test2[1]) \
                   or (test2[0] + test2[1]) >= sigma ** 2
            assert (sigma ** 4) * beta >= (test3[0] - test3[1]) \
                   or (test3[0] + test3[1]) >= (sigma ** 4) * beta
            print("Test passed\n")
    print("ALL TESTS PASSED")


def plot_pearson_3d(x, sigma):
    """ Plots all pearson function in a range for x and beta for a given sigma

    :param x: x value range
    :param sigma: standard deviation

    :type x: ndarray
    :type sigma: float
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    beta = np.arange(2, 3.5, 0.001)
    xx, bb = np.meshgrid(x, beta, sparse=True)

    f = pearson_function_fast(xx, bb, sigma)
    ax.plot_surface(xx, bb, f)
    plt.show()


def plot_pearson(x, beta, sigma):
    """ Plots the pearson function

    :param x:
    :param beta: kurtosis
    :param sigma: standard deviation

    :type x: ndarray
    :type beta: float
    :type sigma: float
    """
    f = pearson_function_fast(x, beta, sigma)
    plt.plot(x, f)
    plt.show()


def measurement_fun(x, d=0.5, sigma1=1, sigma2=1):
    """ Calculation of the measurement function

    :param x:
    :param d:
    :param sigma1:
    :param sigma2:

    :type x: float, ndarray
    :type d: float
    :type sigma1: float
    :type sigma1: float

    :return: the measurement value
    :rtype: float
    """
    with np.errstate(invalid='ignore'):
        f = d / (np.sqrt(2 * np.pi) * sigma1) \
            * np.exp(-0.5 * ((x / sigma1) ** 2)) \
            + (1 - d) * 4 \
            * np.where(x == 0,
                       (np.sqrt(np.pi) * sigma2) / 2,
                       x * sc.kn(1, 2 * x / (np.sqrt(np.pi) * sigma2))) \
            / ((np.pi * sigma2) ** 2)
    return f


def create_measurement(x, d=0.5, sigma1=1, sigma2=1):
    """ Calculation of the measurement data
    Extends the function by flipping it around the y-axis

        :param x:
        :param d:
        :param sigma1:
        :param sigma2:

        :type x: float, ndarray
        :type d: float
        :type sigma1: float
        :type sigma1: float

        :return: the measurement data
        :rtype: float
        """
    x_san = np.where(x > 0, x, x * -1)
    f = measurement_fun(x_san, d, sigma1, sigma2)
    return f


if __name__ == "__main__":
    x_values = np.arange(-3, 3, 0.01)
    test_pearson_function()
    plot_pearson_3d(x_values, 0.75)
    plot_pearson(x_values, 2.5, 1)
    f_meas = create_measurement(x_values)
    plt.plot(x_values, f_meas)
    cs = scipy.interpolate.CubicSpline(x_values, np.log(f_meas))
    plt.plot(x_values, cs(x_values))
    plt.show()
    print("done")
