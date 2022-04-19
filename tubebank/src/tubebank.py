#  Developed by Ellie Litwack for Baltimore Aircoil Company
#  April 2022

import numpy as np
import scipy.interpolate
import warnings
import math

#  Data from Fundamentals of Heat and Mass Transfer,
#  by Incopera et. al, 6th edition pages 442-443
#  and https://www.thermopedia.com/content/1211/
#  Digitized using DataTief and WebPlotDigitizer
#  Based on original research by A. Zukauskas
#  at the Academy of Sciences of the Lithuanian SSR

# load all the data
folder = './data/'
square_k1 = [[float(x), np.loadtxt(open(folder + "inline_k1_" + x + ".csv", "rb"), delimiter=",")]
             for x in ['1e3', '1e4', '1e5', '1e6']]
inline_EU = [[float(x) / 100, np.loadtxt(open(folder + "inline_EU_PL" + x + ".csv", "rb"), delimiter=",")]
             for x in ['125', '150', '200', '250']]
staggered_k1 = [[float(x), np.loadtxt(open(folder + "staggered_k1_" + x + ".csv", "rb"), delimiter=",")]
                for x in ['1e2', '1e3', '1e4', '1e5']]
staggered_EU = [[float(x) / 100, np.loadtxt(open(folder + "staggered_EU_PT" + x + ".csv", "rb"), delimiter=",")]
                for x in ['125', '150', '200', '250']]
inline_k3 = [[float(x), np.loadtxt(open(folder + "thermopedia_k3_inline_re" + x + ".csv", "rb"), delimiter=",")]
             for x in ['1e2', '1e4', '1e6']]
staggered_k3 = [[float(x), np.loadtxt(open(folder + "thermopedia_k3_staggered_re" + x + ".csv", "rb"), delimiter=",")]
                for x in ['1e2', '1e4', '1e6']]
inline_k4 = np.loadtxt(open(folder + "inline_k4.csv", "rb"), delimiter=",")
staggered_k4 = np.loadtxt(open(folder + "staggered_k4.csv", "rb"), delimiter=",")


#  ys are known to be 1.25, 1.5, 2.0, 2.5
#  make a rectangular bivariate spline for each data set
#  by filling in all x values from univariate splines


# define helper functions for interpolation
def _init_2d_interpolator(data):
    """
    Initialize a 2D interpolator from a list of lists of points.

    Parameters:
    data ([[y, [[x, f],[x, f],... ]], [y, [[x, f],[x, f],...]],... ]): Where y is the discrete value (such as b)
    and x is the continuous value (such as Re)

    Returns:
    interpolator (scipy.interpolate.RectBivariateSpline): The interpolator
    """
    #  get all the xs
    data.sort(key=lambda line: line[0])  # sort by y
    all_xs = []  # for generating a rectangular mesh
    for line in data:
        all_xs = all_xs + [point[0] for point in line[1]]
    all_xs = list(set(all_xs))  # remove duplicates
    all_xs.sort()  # sort by x
    #  generate a univariate spline for each y
    splines = []
    for line in data:
        splines.append([line[0],  # b
                        scipy.interpolate.interp1d(
                            [j[0] for j in line[1]],  # x
                            [j[1] for j in line[1]],  # f
                            kind='cubic', fill_value='extrapolate', bounds_error=False)])

    data_postprocessed = [[spline[0], spline[1](all_xs)] for spline in splines]
    #  generate the rectangular mesh
    xs = np.array(all_xs)
    ys = np.array([j[0] for j in data_postprocessed])
    fs = np.array([np.array(j[1]) for j in data_postprocessed])
    #  generate the interpolator
    return scipy.interpolate.RectBivariateSpline(ys, xs, fs)


def _init_inline_k1_interpolator(data):
    """
    Handles a special case for the k1 data.

    Parameters:
    data ([[y, [[x, f],[x, f],... ]], [y, [[x, f],[x, f],...]],... ]): Where y is the discrete value (such as b)
    and x is the continuous value (such as Re)

    Returns:
    interpolator (scipy.interpolate.RectBivariateSpline): The interpolator
    """
    #  repeat for staggered
    #  get all the xs
    data.sort(key=lambda line: line[0])  # sort by y
    all_xs = []  # for generating a rectangular mesh
    for line in data:
        all_xs = all_xs + [point[0] for point in line[1]]
    #  generate a univariate spline for each b

    splines = []
    for line in data:
        splines.append([line[0],  # b
                        scipy.interpolate.interp1d(
                            [np.log10(j[0]) for j in line[1]],  # Re
                            [np.log10(j[1]) for j in line[1]],  # Eu
                            kind='quadratic', fill_value="extrapolate", bounds_error=False)])

    all_xs = [np.log10(x) for x in all_xs]
    all_xs = all_xs + [x for x in
                       [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
    all_xs = list(set(all_xs))  # remove duplicates
    all_xs.sort()  # sort by x
    data_postprocessed = [[spline[0], spline[1](all_xs)] for spline in splines]
    #  generate the rectangular mesh
    xs = np.array(all_xs)
    ys = np.array([j[0] for j in data_postprocessed])
    fs = np.array([np.array(j[1]) for j in data_postprocessed])
    #  generate the interpolator
    return scipy.interpolate.RectBivariateSpline(np.log10(ys), xs, fs, kx=3, ky=3)


def _init_staggered_k1_interpolator(data):
    """
    Initialize a 2D interpolator from a list of lists of points.
    Handles a special case for the k1 data.

    Parameters:
    data ([[y, [[x, f],[x, f],... ]], [y, [[x, f],[x, f],...]],... ]): Where y is the discrete value (such as b)
    and x is the continuous value (such as Re)

    Returns:
    interpolator (scipy.interpolate.RectBivariateSpline): The interpolator
    """
    #  repeat for staggered
    #  get all the xs
    data.sort(key=lambda line: line[0])  # sort by y
    all_xs = []  # for generating a rectangular mesh
    for line in data:
        all_xs = all_xs + [point[0] for point in line[1]]
    all_xs = list(set(all_xs))  # remove duplicates
    all_xs.sort()  # sort by x
    #  generate a univariate spline for each y
    splines = []
    for line in data:
        splines.append([line[0],  # b
                        scipy.interpolate.interp1d(
                            [j[0] for j in line[1]],  # x
                            [j[1] for j in line[1]],  # f
                            kind='cubic', fill_value='extrapolate', bounds_error=False)])

    data_postprocessed = [[spline[0], spline[1](all_xs)] for spline in splines]
    #  generate the rectangular mesh
    xs = np.array(all_xs)
    ys = np.array([j[0] for j in data_postprocessed])
    fs = np.array([np.array(j[1]) for j in data_postprocessed])
    #  generate the interpolator
    return scipy.interpolate.RectBivariateSpline(ys, xs, fs, kx=1, ky=3)


def _init_1d_interpolator(data):
    data.sort()
    return scipy.interpolate.interp1d([j[1] for j in data], [j[0] for j in data], kind='cubic', fill_value=0,
                                      bounds_error=False)


# get interpolators for pressure drop correlations
# b is pt
inline_k4_interpolator = _init_1d_interpolator(inline_k4)
staggered_k4_interpolator = _init_1d_interpolator(staggered_k4)
staggered_EU_interpolator = _init_2d_interpolator(staggered_EU)  # b, Re
inline_EU_interpolator = _init_2d_interpolator(inline_EU)  # b, Re
inline_k1_interpolator = _init_inline_k1_interpolator(square_k1)  # log10(Re), log10((pt-1)/(pl-1)), gives log10(k1)
staggered_k1_interpolator = _init_staggered_k1_interpolator(staggered_k1)  # C1 smooth in PT/PL axis, C0 smooth in Re


#  define masks for pressure drop correlations
def _inline_EU_validity_mask(b, re):
    """
    A conservative mask for the inline Eu correlation

    Parameters:
    b (float or numpy array): value of b parameter (PL)
    re (float or numpy array): must have the same shape as b

    Returns:
    numpy array of booleans or boolean: Same shape as b and Re.
    True means the result is valid at that point in Re/b space.
    Much of the data outside of the bounds has been extrapolated
    to ensure that the interpolator produces good derivatives
    (because filling with an error value biases the derivative).
    Failure to use the mask can lead to bad results.
    """
    re_max = 1700000  # conservative boundary, tested empirically to ensure good edges
    return \
        np.logical_or(
            np.logical_or(
                np.logical_and(
                    np.logical_and(np.greater_equal(re, 1544.952), np.less_equal(re, re_max)),
                    np.logical_and(np.greater_equal(b, 1.25), np.less_equal(b, 2.5))),
                np.logical_and(
                    np.logical_and(np.greater_equal(re, 29.84215), np.less_equal(re, re_max)),
                    np.logical_and(np.greater_equal(b, 1.25), np.less_equal(b, 1.5))
                )),
            np.logical_or(
                np.logical_or(
                    np.logical_and(
                        np.logical_and(np.greater_equal(re, 29.58769), np.less_equal(re, re_max)),
                        np.equal(b, 1.25)),
                    np.logical_and(
                        np.logical_and(np.greater_equal(re, 29.84215), np.less_equal(re, re_max)),
                        np.equal(b, 1.5))
                ),
                np.logical_or(
                    np.logical_and(
                        np.logical_and(np.greater_equal(re, 1544.952), np.less_equal(re, re_max)),
                        np.equal(b, 2.00)),
                    np.logical_and(
                        np.logical_and(np.greater_equal(re, 569.636), np.less_equal(re, re_max)),
                        np.equal(b, 2.50))
                )
            ))


def _staggered_EU_validity_mask(b, re):
    """
    A conservative mask for the inline Eu correlation

    Parameters:
    b (float or numpy array): value of b parameter (PT)
    re (float or numpy array): must have the same shape as b

    Returns:
    numpy array of booleans or boolean: Same shape as b and Re.
    True means the result is valid at that point in Re/b space.
    Much of the data outside of the bounds has been extrapolated
    to ensure that the interpolator produces good derivatives
    (because filling with an error value biases the derivative).
    Failure to use the mask can lead to bad results.
    """
    re_max = 2400000  # conservative boundary, tested empirically to ensure good edges
    return \
        np.logical_or(
            np.logical_or(
                np.logical_and(
                    np.logical_and(np.greater_equal(re, 119.5538), np.less_equal(re, re_max)),
                    np.logical_and(np.greater_equal(b, 1.25), np.less_equal(b, 2.5))),
                np.logical_and(
                    np.logical_and(np.greater_equal(re, 9.60957), np.less_equal(re, re_max)),
                    np.logical_and(np.greater_equal(b, 1.25), np.less_equal(b, 1.5))
                )),
            np.logical_or(
                np.logical_or(
                    np.logical_and(
                        np.logical_and(np.greater_equal(re, 9.60957), np.less_equal(re, re_max)),
                        np.equal(b, 1.25)),
                    np.logical_and(
                        np.logical_and(np.greater_equal(re, 9.60957), np.less_equal(re, re_max)),
                        np.equal(b, 1.5))
                ),
                np.logical_or(
                    np.logical_and(
                        # conservative and empirical boundary
                        np.logical_and(np.greater_equal(re, 119.55374), np.less_equal(re, re_max)),
                        np.equal(b, 2.00)),
                    np.logical_and(
                        np.logical_and(np.greater_equal(re, 119.55374), np.less_equal(re, re_max)),
                        np.equal(b, 2.50))
                )
            ))


def _inline_k1_validity_mask(pt, pl, re):
    ptplratio = (pt - 1) / (pl - 1)
    return \
        np.logical_and(
            np.logical_and(np.greater_equal(ptplratio, 0.1), np.less_equal(ptplratio, 5.9)),
            np.logical_and(np.greater_equal(re, 1e3), np.less_equal(re, 1e6)))


def _staggered_k1_validity_mask(pt, pl, re):
    ptplratio = pt / pl
    return \
        np.logical_or(
            np.logical_or(
                np.logical_and(
                    np.logical_and(np.greater_equal(ptplratio, .5013), np.less_equal(ptplratio, 3.725)),
                    np.logical_and(np.greater_equal(re, 1e3), np.less_equal(re, 1e4))),
                np.logical_and(
                    np.logical_and(np.greater_equal(ptplratio, .48), np.less_equal(ptplratio, 3.28)),
                    np.logical_and(np.greater_equal(re, 1e4), np.less_equal(re, 1e5)))
            ),
            np.logical_and(
                np.logical_and(np.greater_equal(ptplratio, 1.155), np.less_equal(ptplratio, 3.26)),
                np.logical_and(np.greater_equal(re, 1e2), np.less_equal(re, 1e3)))
        )


# define equations, special case handles, and wrapper functions for pressure drop correlations

def _get_k3(staggered, re, n):
    """
    Get the value of K3 for a given Re and n.

    Parameters:
    staggered (bool): True if the correlation is staggered.
    re (float): The Reynolds number.
    n (int): The number of tube bank rows.

    Returns:
    float: The value of K3.
    """
    if n > 7:
        return 1.0
    elif not (1e2 <= re <= 1e6):
        raise ValueError("Re must be between 1e2 and 1e6")
    elif type(n) != int and not n.is_integer():
        raise ValueError("n must be an integer")

    data = inline_k3
    if staggered:
        data = staggered_k3
    if 1e2 < re < 1e4:
        # how close are we to 1e4 on a log scale?
        log_re = np.log10(re)
        interpolation_distance = log_re - 2  # how much we are interpolating up from 1e2
        slope = data[1][1][n - 1] - data[0][1][n - 1]  # slope of the underlying data
        interpolation_amount = slope * interpolation_distance
        return data[0][1][n - 1] + interpolation_amount
    elif 1e4 < re < 1e6:
        # how close are we to 1e6 on a log scale?
        log_re = np.log10(re)
        interpolation_distance = log_re - 4  # how much we are interpolating up from 1e4
        slope = data[2][1][n - 1] - data[1][1][n - 1]  # slope of the underlying data
        interpolation_amount = slope * interpolation_distance
        return data[1][1][n - 1] + interpolation_amount
    elif re == 1e2:
        return data[0][1][n - 1]
    elif re == 1e4:
        return data[1][1][n - 1]
    else:
        return data[2][1][n - 1]


def _get_k2(viscosity_ratio, re):
    """
    Gives the bulk/boundary viscosity ratio correction factor. The difference in these viscosities comes from the
    difference in temperature between the bulk fluid and the boundary layer.

    :param viscosity_ratio (float): The boundary layer viscosity divided by the bulk viscosity.
    :param re (float): The Reynolds number.
    :return:
    """
    if viscosity_ratio > 1.0:
        p = 0.776 * math.exp(-0.545 * (math.pow(re, 0.256)))
        return math.pow(viscosity_ratio, p)
    elif viscosity_ratio < 1.0:
        if re > 1e3:
            raise ValueError("Reynolds number must be less than 1e3 if viscosity ratio is less than 1.0")
        else:
            return 0.968 * math.exp(-1.076 * (math.pow(re, 0.196)))
    else:
        return 1.0


def _inline_k1_interpolator_wrapper(re, pt, pl):
    """
    Handles underlying logarithmic interpolation for the inline K1 values.
    :param re: float
    :param pt: float
    :param pl: float
    :return: k1
    """
    return 10 ** inline_k1_interpolator(np.log10(re), np.log10((pt - 1) / (pl - 1)))


def _get_eu(re, n, staggered, pt, pl, viscosity_ratio=1.0, alpha=0.0, beta=90.0, bounds_warnings=False,
            k_fill_value=None):
    # Check bounds not dependent on staggered vs. inline
    if beta > 90.0:  # bound for k4
        if bounds_warnings:
            warnings.warn("beta > 90.0")
        return None
    elif not -90 <= alpha <= 90:  # bound for k4
        if bounds_warnings:
            warnings.warn("alpha not between -90 and 90")
        return None
    elif viscosity_ratio <= 0:  # bound for k2
        if bounds_warnings:
            warnings.warn("viscosity ratio <= 0")
        return None
    elif (type(n) != int and not n.is_integer()):  # integer condition for k3
        if bounds_warnings:
            warnings.warn("n not an integer")
        return None
    elif (not (1e2 <= re <= 1e6)) and n <= 7:  # reynolds number condition for k3
        if bounds_warnings:
            warnings.warn("Re not between 1e2 and 1e6 but n <= 7")
        return None

    if viscosity_ratio == 1.0:
        k2 = 1.0
    else:
        k2 = _get_k2(viscosity_ratio, re)

    # to avoid a value error, we need to bounds check before getting k3

    k3 = _get_k3(staggered, re, n)

    if staggered:
        EU = staggered_EU_interpolator(pt, re)
        k1 = staggered_k1_interpolator(re, pt / pl)
        k4 = np.cos(np.radians(alpha)) * staggered_k4_interpolator(beta)
        if not _staggered_k1_validity_mask(pt, pl, re):
            if k_fill_value is None:
                return None
            k1 = k_fill_value
            if bounds_warnings:
                warnings.warn("k1 out of bounds")
        if not beta < 27.85182898:
            if k_fill_value is None:
                return None
            k4 = k_fill_value
            if bounds_warnings:
                warnings.warn("k4 out of bounds")
        if not _staggered_EU_validity_mask(pt, re):
            if bounds_warnings:
                warnings.warn("Staggered EU and/or k1 and/or beta not valid")
            return None
    else:
        EU = inline_EU_interpolator(pt, re)
        k1 = _inline_k1_interpolator_wrapper(re, pt, pl)
        k4 = np.cos(np.radians(alpha)) * inline_k4_interpolator(beta)
        if not _inline_EU_validity_mask(pt, re) or not _inline_k1_validity_mask(pt, pl, re) or beta < 24.68562874:
            if bounds_warnings:
                warnings.warn("Inline EU and/or k1 and/or beta not valid")
            return None

    return EU * k1 * k2 * k3 * k4


# define the pressure drop function itself

def _get_max_velocity(staggered, st, sl, diameter, mean_superficial_velocity):
    # calculate maximum velocity
    sd = (sl ** 2 + (st / 2) ** 2) ** (1 / 2)
    if not staggered or sd < (st + diameter) / 2:
        maximum_velocity = (st / (st - diameter)) * mean_superficial_velocity
    else:
        maximum_velocity = st / (2 * (sd - diameter)) * mean_superficial_velocity
    return maximum_velocity


def get_pressure_drop(n, staggered, diameter, st, sl, mean_superficial_velocity, density, viscosity, alpha=0.0,
                      beta=90.0,
                      bounds_warnings=False, viscosity_ratio=1.0, k_fill_value=None):
    """
    Calculates the pressure drop across the bank.

    You must use consistent base units for all inputs (all masses must be the same unit,
    all lengths must be the same unit, all masses must be the same unit).

    For example, if you give lengths in meters, you must give density in mass/m^3. If you
    give velocity in feet/year, you must give viscosity in mass/ft/year.

    Using meters, meters/second, kilograms/meter^3, and Pascal-seconds yields a result in Pascals.

    You will get back a pressure drop in units mass/(length^2*time^2).

    :param n: The number of rows.
    :param staggered: Whether the pipe is staggered or inline.
    :param diameter: The diameter of the pipe (length).
    :param st: Transverse pitch (length).
    :param sl: Longitudinal pitch (length).
    :param mean_superficial_velocity: The mean superficial velocity of fluid (length/time).
    :param density: The density of the fluid (mass/(length^3)).
    :param viscosity: The dynamic viscosity of the fluid (mass/(length*time)).
    :param alpha: Rotation angle (degrees).
    :param beta: Incline angle (degrees).
    :param bounds_warnings: Whether to print warnings when out of bounds.
    :param viscosity_ratio: The viscosity ratio of the surrounding fluid.
    :return: The pressure drop across the pipe (mass/(length*time^2)).
    """

    maximum_velocity = _get_max_velocity(staggered, st, sl, diameter, mean_superficial_velocity)
    re = (maximum_velocity * diameter * density) / viscosity
    eu = _get_eu(re, n, staggered, (st / diameter), (sl / diameter), alpha=alpha, beta=beta,
                 bounds_warnings=bounds_warnings, viscosity_ratio=viscosity_ratio, k_fill_value=k_fill_value)
    if eu is None:
        return None
    else:
        delta_p_row_mean = eu * 0.5 * density * maximum_velocity ** 2
        return delta_p_row_mean * n


# define Zukauskas 1972 Nusselt number correlation


def get_coefficients_c1_and_m(re, st, sl, staggered):
    pitchratio = st / sl
    if not staggered and pitchratio < 0.7 and 1e3 <= re < 2e5:
        c1 = 0.27
        m = 0.63
    elif staggered and pitchratio < 2 and 1e3 <= re < 2e5:
        c1 = 0.35 * (pitchratio ** (1 / 5))
        m = 0.6
    elif staggered and pitchratio >= 2 and 1e3 <= re < 2e5:
        c1 = 0.4
        m = 0.6
    elif not staggered and 2e5 <= re <= 2e6:
        c1 = 0.021
        m = 0.84
    elif staggered and 2e5 <= re <= 2e6:
        c1 = 0.022
        m = 0.84
    elif 1e2 <= re <= 1e3:
        raise ValueError("The tubes should be approximated as single isolated cylinders.")
    else:
        raise ValueError("The correlation is not valid for the given inputs.")
    return c1, m


# define the c2 coefficients
c2s_aligned = [[1, 0.7],
               [2, 0.8],
               [3, 0.86],
               [4, 0.9],
               [5, 0.92],
               [7, 0.95],
               [10, 0.97],
               [13, 0.98],
               [16, 0.99],
               [20, 1.0]]
c2s_aligned_interpolator = scipy.interpolate.interp1d(np.array([x[0] for x in c2s_aligned]),
                                                      np.array([x[1] for x in c2s_aligned]))
c2s_staggered = [[1, 0.64],
                 [2, 0.76],
                 [3, 0.84],
                 [4, 0.89],
                 [5, 0.92],
                 [7, 0.95],
                 [10, 0.97],
                 [13, 0.98],
                 [16, 0.99],
                 [20, 1.0]]
c2s_staggered_interpolator = scipy.interpolate.interp1d(np.array([x[0] for x in c2s_staggered]),
                                                        np.array([x[1] for x in c2s_staggered]))


# Prs is the surface prandtl number
def get_nusselt_number(staggered, diameter, mean_superficial_velocity, density, viscosity, st, sl, prandlt_number, n,
                       bulk_surface_prandlt_ratio=1.0):
    """
        Calculates the bank's Nusselt number.

        You must use consistent base units for all inputs (all masses must be the same unit,
        all lengths must be the same unit, all masses must be the same unit).

        For example, if you give lengths in meters, you must give density in mass/m^3. If you
        give velocity in feet/year, you must give viscosity in mass/ft/year. The result will be
        dimensionless.

        Valid for n >= 20, 0.7 ~<= Prandlt number ~<= 500, 1000 ~<= Re ~<= 2e6.

        :param staggered: Whether the pipes are staggered or inline.
        :param diameter: The diameter of the pipe (length).
        :param mean_superficial_velocity: The mean superficial velocity of fluid (length/time).
        :param density: The density of the fluid (mass/(length^3)).
        :param viscosity: The dynamic viscosity of the fluid (mass/(length*time)).
        :param st: Transverse pitch (length).
        :param sl: Longitudinal pitch (length).
        :param prandlt_number: The Prandlt number of the fluid.
        :param n: The number of rows in the bank.
        :param bulk_surface_prandlt_ratio: The ratio of the bulk Prandlt number to the surface Prandlt number,
        for cases where there is a large difference in bulk and surface fluid properties.
        :return: The Nusselt number.
        """
    maximum_velocity = _get_max_velocity(staggered, st, sl, diameter, mean_superficial_velocity)
    re = (maximum_velocity * diameter * density) / viscosity
    re = re
    c1, m = get_coefficients_c1_and_m(re, st, sl, staggered)
    nusselt_number = c1 * (re ** m) * (prandlt_number ** 0.36) * (bulk_surface_prandlt_ratio ** 0.25)
    if n < 20 and staggered:
        c2 = c2s_staggered_interpolator(n)
    elif n < 20 and not staggered:
        c2 = c2s_aligned_interpolator(n)
    else:
        c2 = 1.0
    return nusselt_number * c2

#  If n <= 7, a correction factor for short tube banks is applied.
n = 5 #  The number of rows.
diameter = 0.02 #  tube diameters (length)
st = 0.03 #  transverse pitch (length)
sl = 0.03 #  longitudinal pitch (length)
mean_superficial_velocity = 2 #  (length/time)
density = 1.2 #  mass/(length*time)
viscosity = 1.8e-5 #  dynamic viscosity (mass/(length*time))
staggered = True #  False for inline banks, true for stagggered ones
#  alpha, beta, viscosity_ratio, and bounds_warnings are optional.
#  get the pressure drop across a tube bank
get_pressure_drop(n, staggered, diameter, st, sl,
     mean_superficial_velocity, density, viscosity)

prandlt_number = 0.7
#  get the Nusselt number of a tube bank
#  valid for n >= 20, 0.7 ~<= Prandlt number ~<= 500,
#  1000 ~<= Re ~<= 2e6
get_nusselt_number(staggered, diameter, mean_superficial_velocity,
     density, viscosity, st, sl, prandlt_number, n)
