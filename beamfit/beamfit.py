import scipy.interpolate
import scipy.special as sc
import scipy.integrate as ig
import scipy.optimize as op
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
    ax.set_title("3D Pearson Function")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel("$f(x)$")
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
    plt.title("Pearson function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
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
    :type sigma2: float

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
        :type sigma2: float

        :return: the measurement data
        :rtype: float
        """
    x_san = np.where(x > 0, x, x * -1)
    f = measurement_fun(x_san, d, sigma1, sigma2)
    return f


def sum_n_pearson(x, c, beta, sigma):
    """ Calculation of a sum of pearson functions

    :param x:
    :param c:
    :param beta:
    :param sigma:
    :type x: float, ndarray
    :type c: float, ndarray
    :type beta: float, ndarray
    :type sigma: float, ndarray

    :return: the function values
    :rtype: float, ndarray
    """
    return np.sum(c * pearson_function_fast(x[:, np.newaxis],
                                            beta,
                                            sigma), axis=1)


class SumNPearson:
    """ Class to fit a sum of pearson functions

    This class helps to set up the parameters to fit a sum of pearson functions
    to some measurement data. All parameters can be chosen to be a fix value
    or a value that gets fitted and that starts at a certain start point

        *fix_c_i - fixes a c value to a given value
        *unfix_c_i - unfixes a c value
        *set_startpoint_c_i - sets the start value of a c value
        *fix_beta_i - fixes a beta value to a given value
        *unfix_beta_i - unfixes a beta value
        *set_startpoint_beta_i - sets the start value of a beta value
        *fix_sigma_i - fixes a sigma value to a given value
        *unfix_sigma_i - unfixes a sigma value
        *set_startpoint_sigma_i - sets the start value of a sigma value
        *fit - starts the fitting process
        *get_cs - returns the c values
        *get_betas - returns the beta values
        *get_sigmas - returns the sigma values
    """

    def __init__(self, n=3):
        self.n = n
        self.cs = np.empty(n)
        self.c_flags = np.full(n, True)
        # last c should not be fitted
        self.c_flags[n - 1] = False
        self.betas = np.empty(n)
        self.beta_flags = np.full(n, True)
        self.sigmas = np.empty(n)
        self.sigma_flags = np.full(n, True)
        self.start_values = np.concatenate((np.full(n, 1 / n),
                                            np.full(n, 10),
                                            np.full(n, 1)))
        self.upper = np.concatenate((np.full(n, 1),
                                     np.full(n * 2, np.inf)))
        self.lower = np.concatenate((np.full(n, 0),
                                     np.full(n, 2),
                                     np.full(n, 0)))

    def fix_c_i(self, index, value):
        """Fixes the i-th c value to the given value

        :param index: index of the fixed c
        :param value: value which the c should be fixed to
        :type index: int
        :type value: float
        """
        # n - 1 because the last c should not be changed by the user
        if index >= self.n - 1:
            raise Exception("Index out of range")
        self.cs[index] = value
        self.c_flags[index] = False

    def unfix_c_i(self, index):
        """Unfixes the i-th c value

        :param index: index of the unfixed c
        :type index: int
        """
        # n - 1 because the last c should not be changed by the user
        if index >= self.n - 1:
            raise Exception("Index out of range")
        self.c_flags[index] = True

    def set_startpoint_c_i(self, index, value):
        """Sets the start point of the i-th c
        :param index: index of the c
        :param value: value where the fitting should be started
        :type index: int
        :type value: float
        """
        # n - 1 because the last c should not be changed by the user
        if index >= self.n - 1:
            raise Exception("Index out of range")
        self.start_values[index] = value

    def fix_beta_i(self, index, value):
        """Fixes the i-th beta value to the given value

        :param index: index of the fixed beta
        :param value: value which the beta should be fixed to
        :type index: int
        :type value: float
        """
        if index >= self.n:
            raise Exception("Index out of range")
        self.betas[index] = value
        self.beta_flags[index] = False

    def unfix_beta_i(self, index):
        """Unfixes the i-th beta value

        :param index: index of the unfixed beta
        :type index: int
        """
        if index >= self.n:
            raise Exception("Index out of range")
        self.beta_flags[index] = True

    def set_startpoint_beta_i(self, index, value):
        """Sets the start point of the i-th beta
        :param index: index of the beta
        :param value: value where the fitting should be started
        :type index: int
        :type value: float
        """
        if index >= self.n:
            raise Exception("Index out of range")
        self.start_values[index + self.n] = value

    def fix_sigma_i(self, index, value):
        """Fixes the i-th sigma value to the given value

        :param index: index of the fixed sigma
        :param value: value which the sigma should be fixed to
        :type index: int
        :type value: float
        """
        if index >= self.n:
            raise Exception("Index out of range")
        self.sigmas[index] = value
        self.sigma_flags[index] = False

    def unfix_sigma_i(self, index):
        """Unfixes the i-th sigma value

        :param index: index of the unfixed sigma
        :type index: int
        """
        if index >= self.n:
            raise Exception("Index out of range")
        self.sigma_flags[index] = True

    def set_startpoint_sigma_i(self, index, value):
        """Sets the start point of the i-th sigma
        :param index: index of the sigma
        :param value: value where the fitting should be started
        :type index: int
        :type value: float
        """
        if index >= self.n:
            raise Exception("Index out of range")
        self.start_values[index + 2 * self.n] = value

    def _func(self, x, *p):
        params = np.concatenate((self.cs, self.betas, self.sigmas))
        flags = np.concatenate((self.c_flags,
                                self.beta_flags,
                                self.sigma_flags))
        params[flags] = np.array(p)
        self.cs, self.betas, self.sigmas = np.split(params, 3)
        self.cs[self.n - 1] = 1 - np.sum(self.cs[:self.n - 1])
        return sum_n_pearson(x, self.cs, self.betas, self.sigmas)

    def fit(self, x_data, y_data):
        """Fits the sum of pearson functions to the given data

        This method starts the fitting process with the given data. The fitted
        parameters are stored inside the object and can be accessed with the
        corresponding method.

        :param x_data:
        :param y_data:

        :type x_data: ndarray
        :type y_data: ndarray
        """
        flags = np.concatenate((self.c_flags,
                                self.beta_flags,
                                self.sigma_flags))
        op.curve_fit(self._func, x_data, y_data,
                     bounds=(self.lower[flags], self.upper[flags]),
                     p0=self.start_values[flags])

    def get_cs(self):
        """Returns the c parameters
        :rtype: ndarray
        """
        return self.cs

    def get_betas(self):
        """Returns the beta parameters
        :rtype: ndarray
        """
        return self.betas

    def get_sigmas(self):
        """Returns the sigma parameters
        :rtype: ndarray
        """
        return self.sigmas


if __name__ == "__main__":
    x_values = np.arange(-4, 4, 0.5)
    x_fine = np.arange(-4, 4, 0.01)
    test_pearson_function()
    plot_pearson_3d(x_fine, 0.75)
    plot_pearson(x_fine, 2.5, 1)

    f_meas = create_measurement(x_values, d=0.5, sigma1=1, sigma2=1)
    cs = scipy.interpolate.CubicSpline(x_values, np.log(f_meas))
    _, ax_meas = plt.subplots()
    ax_meas.plot(x_values, f_meas,
                 linestyle="None", marker="x", label="$f^{exp}(x)$")
    ax_meas.plot(x_fine, np.exp(cs(x_fine)), label="$f(x)$")
    ax_meas.set_xlabel("x")
    ax_meas.set_ylabel("f(x)")
    ax_meas.set_title("Measurement data and the cubic spine of the given data")
    ax_meas.legend()
    plt.show()

    n_per = 3
    show_f_i = 1
    _, axs = plt.subplots(1, n_per, figsize=(26, 8))
    for j in range(1, n_per + 1):
        sumNP = SumNPearson(j)
        # sumNP.fix_beta_i(0, 4)
        # sumNP.set_startpoint_beta_i(0, 4)
        sumNP.fit(x_values, f_meas)
        cs = sumNP.get_cs()
        betas = sumNP.get_betas()
        sigmas = sumNP.get_sigmas()
        print("n = " + str(j) + ":")
        print("\tc: " + str(cs))
        print("\tbetas: " + str(betas))
        print("\tsigmas: " + str(sigmas))

        axs[j - 1].plot(x_values, f_meas, linestyle="None",
                        marker="x", label="Measurements")
        axs[j - 1].plot(x_fine,
                        sum_n_pearson(x_fine, cs, betas, sigmas),
                        label="$f(x)$")
        if show_f_i:
            for k in range(0, j):
                axs[j - 1].plot(x_fine, cs[k] *
                                pearson_function_fast(x_fine,
                                                      betas[k],
                                                      sigmas[k]),
                                label="$f_{}(x)$".format(str(k + 1)),
                                linestyle="--")

        axs[j - 1].set_title("n=" + str(j))
        axs[j - 1].legend()

    plt.tight_layout()
    plt.show()

    print("done")
