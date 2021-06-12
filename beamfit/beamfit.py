import scipy.special as sc
import scipy.integrate as ig
import numpy as np
import matplotlib.pyplot as plt


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
    elif beta < 3:
        m = ((5 * beta) - 9) / (6 - (2 * beta))
        a = np.sqrt((2 * m) + 3) * sigma
        beta_func = sc.beta(0.5, m + 0.5)
        fract = np.square(x / a)
        san_fract = np.where(np.greater(fract, 1), 1, fract)
        # f = np.where(np.greater_equal(fract, 1), 0, ((m + 0.5) / (np.pi * a)) * beta_func * np.power((1 - fract), m))
        f = ((m + 0.5) / (np.pi * a)) * beta_func * np.power((1 - san_fract), m)
    elif beta == 3:
        # maybe use a constant for sqrt(2) and sqrt(pi)
        a = np.sqrt(2) * sigma
        f = (1 / (np.sqrt(np.pi) * a)) * np.exp(- np.square(x / a))
    else:
        m = (5 * beta - 9) / (6 - 2 * beta)
        a = np.sqrt(np.abs(2 * m + 3)) * sigma
        beta_func = sc.beta(0.5, np.abs(m))
        f = (np.abs(m + 0.5) / (np.pi * a)) * beta_func * np.power((1 + np.square(x / a)), m)
    return f


def pearson_function_fast(x, beta, sigma):
    """ Calculation of the Pearson Function with whole-array operations

    :param x:
    :param beta: kurtoses
    :param sigma: standard deviation

    :type x: ndarray
    :type beta: ndarray
    :type sigma: ndarray

    :return: the probability values
    :rtype: ndarray
    """
    beta_between_2_3 = np.logical_and(np.greater_equal(beta, 2), np.less(beta, 3))
    beta_eq_3 = np.equal(beta, 3)
    beta_gt_3 = np.greater(beta, 3)

    # not the most beautiful solution
    beta_san = np.where(beta_eq_3, 0, beta)
    m = np.where(beta_eq_3, 0, (5 * beta_san - 9) / (6 - 2 * beta_san))

    a = np.where(beta_eq_3, np.sqrt(2) * sigma,
                 sigma * np.sqrt(np.where(
                     beta_gt_3,
                     np.abs(2 * m + 3),
                     2 * m + 3))
                 )

    beta_func = np.where(beta_between_2_3, sc.beta(0.5, m + 0.5),
                         np.where(beta_gt_3, sc.beta(0.5, np.abs(m)), 0))

    fract = np.square(x / a)

    san_fract = np.where(beta_between_2_3, np.where(np.greater(fract, 1), 1, fract), fract)

    f = np.where(beta_between_2_3, ((m + 0.5) / (np.pi * a)) * beta_func * np.power((1 - san_fract), m),
                 np.where(beta_eq_3, (1 / (np.sqrt(np.pi) * a)) * np.exp(- san_fract),
                          np.where(beta_gt_3,
                                   (np.abs(m + 0.5) / (np.pi * a)) * beta_func * np.power((1 + san_fract), m),
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
            print("should be 1: " + str(test1[0]) + " deviation: " + str(test1[1]))
            print("should be " + str(sigma ** 2) + ": " + str(test2[0]) + " deviation: " + str(test2[1]))
            print("should be " + str(beta) + ": " + str(test3[0] / (sigma ** 4)) +
                  " deviation: " + str(test3[1] / sigma ** 4))

            assert 1 >= (test1[0] - test1[1]) or (test1[0] + test1[1]) >= 1
            assert sigma ** 2 >= (test2[0] - test2[1]) or (test2[0] + test2[1]) >= sigma ** 2
            assert (sigma ** 4) * beta >= (test3[0] - test3[1]) or (test3[0] + test3[1]) >= (sigma ** 4) * beta
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

    beta = np.arange(2, 20, 0.01)
    xx, bb = np.meshgrid(x, beta, sparse=True)

    f = pearson_function_fast(xx, bb, sigma)
    ax.plot_surface(xx, bb, f)
    plt.show()


def plot_pearson(x, beta, sigma):
    """ Plots the pearson function

    :param x:
    :param beta: kurtoses
    :param sigma: standard deviation

    :type x: ndarray
    :type beta: float
    :type sigma: float
    """
    f = pearson_function_fast(x, beta, sigma)
    plt.plot(x, f)
    plt.show()


if __name__ == "__main__":
    test_pearson_function()
    x_values = np.arange(-3, 3, 0.1)
    plot_pearson_3d(x_values, 0.5)
    print("done")
