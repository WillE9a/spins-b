"""
Defines parametrizations of the structure. A parametrization is a mapping from a
set of values to a structure (z-values).
"""
import abc
import numbers
from typing import Dict, List, Union, Tuple, Optional

import numpy as np
import math
from scipy import signal
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator
from spins.invdes.parametrization import cubic_utils

from autograd import numpy as npa
from autograd import tensor_jacobian_product
from scipy import special

########### Density Filter Functions Adapted From MEEP ########################
# original source: https://github.com/NanoComp/meep/blob/master/python/adjoint/filters.py
        
def mapping(x,eta,beta,radius,width,height,resolution):    
    if radius > 0:
        # filter
        x_filt = conic_filter(x,radius,width,height,resolution)
        #x_filt = cylindrical_filter(x,radius,width,height,resolution)
        
        # projection
        x_proj = tanh_projection(x_filt,beta,eta)
    else:
        x_proj = tanh_projection(x,beta,eta)
    # interpolate to actual materials
    return x_proj.flatten() 


def _proper_pad(x,n):
    '''
    Parameters
    ----------
    x : array_like (2D)
        Input array. Must be 2D.
    n : int
        Total size to be padded to.
    '''
    N = x.size
    k = n - (2*N-1)
    return np.concatenate((x,np.zeros((k,)),np.flipud(x[1:])))

def _centered(arr, newshape):
    '''Helper function that reformats the padded array of the fft filter operation.
    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _edge_pad(arr, pad):

    # fill sides
    left = npa.tile(arr[0, :], (pad[0][0], 1))  # left side
    right = npa.tile(arr[-1, :], (pad[0][1], 1))  # right side
    top = npa.tile(arr[:, 0], (pad[1][0], 1)).transpose()  # top side
    bottom = npa.tile(arr[:, -1], (pad[1][1], 1)).transpose()  # bottom side)

    # fill corners
    top_left = npa.tile(arr[0, 0], (pad[0][0], pad[1][0]))  # top left
    top_right = npa.tile(arr[-1, 0], (pad[0][1], pad[1][0]))  # top right
    bottom_left = npa.tile(arr[0, -1], (pad[0][0], pad[1][1]))  # bottom left
    bottom_right = npa.tile(arr[-1, -1],
                            (pad[0][1], pad[1][1]))  # bottom right

    out = npa.concatenate((npa.concatenate(
        (top_left, top, top_right)), npa.concatenate((left, arr, right)),
                           npa.concatenate(
                               (bottom_left, bottom, bottom_right))),
                          axis=1)

    return out

def simple_2d_filter(x, h):
    """A simple 2d filter algorithm that is differentiable with autograd.
    Uses a 2D fft approach since it is typically faster and preserves the shape
    of the input and output arrays.
    The ffts pad the operation to prevent any circular convolution garbage.
    Parameters
    ----------
    x : array_like (2D)
        Input array to be filtered. Must be 2D.
    h : array_like (2D)
        Filter kernel (before the DFT). Must be same size as `x`
    Returns
    -------
    array_like (2D)
        The output of the 2d convolution.
    """
    (kx, ky) = x.shape
    x = _edge_pad(x,((kx, kx), (ky, ky)))
    return _centered(npa.real(npa.fft.ifft2(npa.fft.fft2(x)*npa.fft.fft2(h))),(kx, ky))

def cylindrical_filter(x, radius, Lx, Ly, resolution):
    '''A uniform cylindrical filter [1]. Typically allows for sharper transitions.
    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    radius : float
        Filter radius (in "meep units")
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like (2D)
        Filtered design parameters.
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''
    Nx = int(Lx*resolution)
    Ny = int(Ly*resolution)
    x = x.reshape(Nx, Ny) # Ensure the input is 2D

    xv = np.arange(0,Lx/2,1/resolution)
    yv = np.arange(0,Ly/2,1/resolution)

    cylindrical = lambda a: np.where(a <= radius, 1, 0)
    hx = cylindrical(xv)
    hy = cylindrical(yv)

    h = np.outer(_proper_pad(hx,3*Nx),_proper_pad(hy,3*Ny))

    # Normalize kernel
    h = h / np.sum(h.flatten())  # Normalize the filter

    # Filter the response
    return simple_2d_filter(x, h)


def conic_filter(x, radius, Lx, Ly, resolution):
    '''A linear conic filter, also known as a "Hat" filter in the literature [1].
    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    radius : float
        Filter radius (in "meep units")
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like (2D)
        Filtered design parameters.
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''
    Nx = int(Lx*resolution)
    Ny = int(Ly*resolution)
    x = x.reshape(Nx, Ny) # Ensure the input is 2D

    xv = np.arange(0,Lx/2,1/resolution)
    yv = np.arange(0,Ly/2,1/resolution)

    conic = lambda a: np.where(np.abs(a**2) <= radius**2, (1 - a / radius), 0)
    hx = conic(xv)
    hy = conic(yv)

    h = np.outer(_proper_pad(hx,3*Nx),_proper_pad(hy,3*Ny))

    # Normalize kernel
    h = h / np.sum(h.flatten())  # Normalize the filter

    # Filter the response
    return simple_2d_filter(x, h)


def gaussian_filter(x, sigma, Lx, Ly, resolution):
    '''A simple gaussian filter of the form exp(-x **2 / sigma ** 2) [1].
    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    sigma : float
        Filter radius (in "meep units")
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like (2D)
        Filtered design parameters.
    References
    ----------
    [1] Wang, E. W., Sell, D., Phan, T., & Fan, J. A. (2019). Robust design of
    topology-optimized metasurfaces. Optical Materials Express, 9(2), 469-482.
    '''
    Nx = int(Lx*resolution)
    Ny = int(Ly*resolution)
    x = x.reshape(Nx, Ny) # Ensure the input is 2D

    xv = np.arange(0,Lx/2,1/resolution)
    yv = np.arange(0,Ly/2,1/resolution)

    gaussian = lambda a: np.exp(-a**2 / sigma**2)
    hx = gaussian(xv)
    hy = gaussian(yv)

    h = np.outer(_proper_pad(hx,3*Nx),_proper_pad(hy,3*Ny))

    # Normalize kernel
    h = h / np.sum(h.flatten())  # Normalize the filter

    # Filter the response
    return simple_2d_filter(x, h)

'''
# ------------------------------------------------------------------------------------ #
Erosion and dilation operators
'''


def exponential_erosion(x, radius, beta, Lx, Ly, resolution):
    ''' Performs and exponential erosion operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Eroded design parameters.
    References
    ----------
    [1] Sigmund, O. (2007). Morphology-based black and white filters for topology optimization.
    Structural and Multidisciplinary Optimization, 33(4-5), 401-424.
    [2] Schevenels, M., & Sigmund, O. (2016). On the implementation and effectiveness of
    morphological close-open and open-close filters for topology optimization. Structural
    and Multidisciplinary Optimization, 54(1), 15-21.
    '''

    x_hat = npa.exp(beta * (1 - x))
    return 1 - npa.log(
        cylindrical_filter(x_hat, radius, Lx, Ly, resolution).flatten()) / beta


def exponential_dilation(x, radius, beta, Lx, Ly, resolution):
    ''' Performs a exponential dilation operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Dilated design parameters.
    References
    ----------
    [1] Sigmund, O. (2007). Morphology-based black and white filters for topology optimization.
    Structural and Multidisciplinary Optimization, 33(4-5), 401-424.
    [2] Schevenels, M., & Sigmund, O. (2016). On the implementation and effectiveness of
    morphological close-open and open-close filters for topology optimization. Structural
    and Multidisciplinary Optimization, 54(1), 15-21.
    '''

    x_hat = npa.exp(beta * x)
    return npa.log(
        cylindrical_filter(x_hat, radius, Lx, Ly, resolution).flatten()) / beta


def heaviside_erosion(x, radius, beta, Lx, Ly, resolution):
    ''' Performs a heaviside erosion operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Eroded design parameters.
    References
    ----------
    [1] Guest, J. K., Prévost, J. H., & Belytschko, T. (2004). Achieving minimum length scale in topology
    optimization using nodal design variables and projection functions. International journal for
    numerical methods in engineering, 61(2), 238-254.
    '''

    x_hat = cylindrical_filter(x, radius, Lx, Ly, resolution).flatten()
    return npa.exp(-beta * (1 - x_hat)) + npa.exp(-beta) * (1 - x_hat)


def heaviside_dilation(x, radius, beta, Lx, Ly, resolution):
    ''' Performs a heaviside dilation operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Dilated design parameters.
    References
    ----------
    [1] Guest, J. K., Prévost, J. H., & Belytschko, T. (2004). Achieving minimum length scale in topology
    optimization using nodal design variables and projection functions. International journal for
    numerical methods in engineering, 61(2), 238-254.
    '''

    x_hat = cylindrical_filter(x, radius, Lx, Ly, resolution).flatten()
    return 1 - npa.exp(-beta * x_hat) + npa.exp(-beta) * x_hat


def geometric_erosion(x, radius, alpha, Lx, Ly, resolution):
    ''' Performs a geometric erosion operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Eroded design parameters.
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''
    x_hat = npa.log(x + alpha)
    return npa.exp(cylindrical_filter(x_hat, radius, Lx, Ly,
                                      resolution)).flatten() - alpha


def geometric_dilation(x, radius, alpha, Lx, Ly, resolution):
    ''' Performs a geometric dilation operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Dilated design parameters.
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''

    x_hat = npa.log(1 - x + alpha)
    return -npa.exp(cylindrical_filter(x_hat, radius, Lx, Ly,
                                       resolution)).flatten() + alpha + 1


def harmonic_erosion(x, radius, alpha, Lx, Ly, resolution):
    ''' Performs a harmonic erosion operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Eroded design parameters.
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''

    x_hat = 1 / (x + alpha)
    return 1 / cylindrical_filter(x_hat, radius, Lx, Ly,
                                  resolution).flatten() - alpha


def harmonic_dilation(x, radius, alpha, Lx, Ly, resolution):
    ''' Performs a harmonic dilation operation.
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like
        Dilated design parameters.
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''

    x_hat = 1 / (1 - x + alpha)
    return 1 - 1 / cylindrical_filter(x_hat, radius, Lx, Ly,
                                      resolution).flatten() + alpha


'''
# ------------------------------------------------------------------------------------ #
Projection filters
'''


def tanh_projection(x, beta, eta):
    '''Projection filter that thresholds the input parameters between 0 and 1. Typically
    the "strongest" projection.
    Parameters
    ----------
    x : array_like
        Design parameters
    beta : float
        Thresholding parameter (0 to infinity). Dictates how "binary" the output will be.
    eta: float
        Threshold point (0 to 1)
    Returns
    -------
    array_like
        Projected and flattened design parameters.
    References
    ----------
    [1] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and robust
    formulations in topology optimization. Structural and Multidisciplinary Optimization, 43(6), 767-784.
    '''

    return (npa.tanh(beta * eta) +
            npa.tanh(beta *
                     (x - eta))) / (npa.tanh(beta * eta) + npa.tanh(beta *
                                                                    (1 - eta)))


def heaviside_projection(x, beta, eta):
    '''Projection filter that thresholds the input parameters between 0 and 1.
    Parameters
    ----------
    x : array_like
        Design parameters
    beta : float
        Thresholding parameter (0 to infinity). Dictates how "binary" the output will be.
    eta: float
        Threshold point (0 to 1)
    Returns
    -------
    array_like
        Projected and flattened design parameters.
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''

    case1 = eta * npa.exp(-beta * (eta - x) / eta) - (eta - x) * npa.exp(-beta)
    case2 = 1 - (1 - eta) * npa.exp(-beta * (x - eta) /
                                    (1 - eta)) - (eta - x) * npa.exp(-beta)
    return npa.where(x < eta, case1, case2)


'''
# ------------------------------------------------------------------------------------ #
Length scale operations
'''


def get_threshold_wang(delta, sigma):
    '''Calculates the threshold point according to the gaussian filter radius (`sigma`) and
    the perturbation parameter (`sigma`) needed to ensure the proper length
    scale and morphological transformation according to Wang et. al. [2].
    Parameters
    ----------
    sigma : float
        Smoothing radius (in meep units)
    delta : float
        Perturbation parameter (in meep units)
    Returns
    -------
    float
        Threshold point (`eta`)
    References
    ----------
    [1] Wang, F., Jensen, J. S., & Sigmund, O. (2011). Robust topology optimization of
    photonic crystal waveguides with tailored dispersion properties. JOSA B, 28(3), 387-397.
    [2] Wang, E. W., Sell, D., Phan, T., & Fan, J. A. (2019). Robust design of
    topology-optimized metasurfaces. Optical Materials Express, 9(2), 469-482.
    '''

    return 0.5 - special.erf(delta / sigma)


def get_eta_from_conic(b, R):
    ''' Extracts the eroded threshold point (`eta_e`) for a conic filter given the desired
    minimum length (`b`) and the filter radius (`R`). This only works for conic filters.
    Note that the units for `b` and `R` can be arbitrary so long as they are consistent.
    Results in paper were thresholded using a "tanh" Heaviside projection.
    Parameters
    ----------
    b : float
        Desired minimum length scale.
    R : float
        Conic filter radius
    Returns
    -------
    float
        The eroded threshold point (1-eta)
    References
    ----------
    [1] Qian, X., & Sigmund, O. (2013). Topological design of electromechanical actuators with
    robustness toward over-and under-etching. Computer Methods in Applied
    Mechanics and Engineering, 253, 237-251.
    [2] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and
    robust formulations in topology optimization. Structural and Multidisciplinary
    Optimization, 43(6), 767-784.
    [3] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''

    norm_length = b / R
    if norm_length < 0:
        eta_e = 0
    elif norm_length < 1:
        eta_e = 0.25 * norm_length**2 + 0.5
    elif norm_length < 2:
        eta_e = -0.25 * norm_length**2 + norm_length
    else:
        eta_e = 1
    return eta_e


def get_conic_radius_from_eta_e(b, eta_e):
    """Calculates the corresponding filter radius given the minimum length scale (b)
    and the desired eroded threshold point (eta_e).
    Parameters
    ----------
    b : float
        Desired minimum length scale.
    eta_e : float
        Eroded threshold point (1-eta)
    Returns
    -------
    float
        Conic filter radius.
    References
    ----------
    [1] Qian, X., & Sigmund, O. (2013). Topological design of electromechanical actuators with
    robustness toward over-and under-etching. Computer Methods in Applied
    Mechanics and Engineering, 253, 237-251.
    [2] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and
    robust formulations in topology optimization. Structural and Multidisciplinary
    Optimization, 43(6), 767-784.
    [3] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    """
    if (eta_e >= 0.5) and (eta_e < 0.75):
        return b / (2 * np.sqrt(eta_e - 0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2 - 2 * np.sqrt(1 - eta_e))
    else:
        raise ValueError(
            "The erosion threshold point (eta_e) must be between 0.5 and 1.")


def indicator_solid(x, c, filter_f, threshold_f, resolution):
    '''Calculates the indicator function for the void phase needed for minimum length optimization [1].
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_e : float
        Erosion threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    Returns
    -------
    array_like
        Indicator value
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = npa.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0] *
                resolution)**2 + (gradient_filtered_field[1] * resolution)**2
    if grad_mag.ndim != 2:
        raise ValueError(
            "The gradient fields must be 2 dimensional. Check input array and filter functions."
        )
    I_s = design_field * npa.exp(-c * grad_mag)
    return I_s


def constraint_solid(x, c, eta_e, filter_f, threshold_f, resolution):
    '''Calculates the constraint function of the solid phase needed for minimum length optimization [1].
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_e : float
        Erosion threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    Returns
    -------
    float
        Constraint value
    Example
    -------
    >> g_s = constraint_solid(x,c,eta_e,filter_f,threshold_f) # constraint
    >> g_s_grad = grad(constraint_solid,0)(x,c,eta_e,filter_f,threshold_f) # gradient
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x)
    I_s = indicator_solid(x.reshape(filtered_field.shape), c, filter_f,
                          threshold_f, resolution).flatten()
    return npa.mean(I_s * npa.minimum(filtered_field.flatten() - eta_e, 0)**2)


def indicator_void(x, c, filter_f, threshold_f, resolution):
    '''Calculates the indicator function for the void phase needed for minimum length optimization [1].
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_d : float
        Dilation threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    Returns
    -------
    array_like
        Indicator value
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x).reshape(x.shape)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = npa.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0] *
                resolution)**2 + (gradient_filtered_field[1] * resolution)**2
    if grad_mag.ndim != 2:
        raise ValueError(
            "The gradient fields must be 2 dimensional. Check input array and filter functions."
        )
    return (1 - design_field) * npa.exp(-c * grad_mag)


def constraint_void(x, c, eta_d, filter_f, threshold_f, resolution):
    '''Calculates the constraint function of the void phase needed for minimum length optimization [1].
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_d : float
        Dilation threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    Returns
    -------
    float
        Constraint value
    Example
    -------
    >> g_v = constraint_void(p,c,eta_d,filter_f,threshold_f) # constraint
    >> g_v_grad = tensor_jacobian_product(constraint_void,0)(p,c,eta_d,filter_f,threshold_f,g_s) # gradient
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x)
    I_v = indicator_void(x.reshape(filtered_field.shape), c, filter_f,
                         threshold_f, resolution).flatten()
    return npa.mean(I_v * npa.minimum(eta_d - filtered_field.flatten(), 0)**2)


def gray_indicator(x):
    '''Calculates a measure of "grayness" according to [1].
    Lower numbers ( < 2%) indicate a good amount of binarization [1].
    Parameters
    ----------
    x : array_like
        Filtered and thresholded design parameters (between 0 and 1)
    Returns
    -------
    float
        Measure of "grayness" (in percent)
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''
    return npa.mean(4 * x.flatten() * (1 - x.flatten())) * 100







# def _centered(arr, newshape):
#     '''Helper function that reformats the padded array of the fft filter operation.
#     Borrowed from scipy:
#     https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
#     '''
#     # Return the center newshape portion of the array.
#     newshape = np.asarray(newshape)
#     currshape = np.array(arr.shape)
#     startind = (currshape - newshape) // 2
#     endind = startind + newshape
#     myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
#     return arr[tuple(myslice)]

# def _edge_pad(arr, pad):
    
#     # fill sides
#     left = npa.tile(arr[0,:],(pad[0][0],1)) # left side
#     right = npa.tile(arr[-1,:],(pad[0][1],1)) # right side
#     top = npa.tile(arr[:,0],(pad[1][0],1)).transpose() # top side
#     bottom = npa.tile(arr[:,-1],(pad[1][1],1)).transpose() # bottom side)
    
#     # fill corners
#     top_left = npa.tile(arr[0,0], (pad[0][0],pad[1][0])) # top left
#     top_right = npa.tile(arr[-1,0], (pad[0][1],pad[1][0])) # top right
#     bottom_left = npa.tile(arr[0,-1], (pad[0][0],pad[1][1])) # bottom left
#     bottom_right = npa.tile(arr[-1,-1], (pad[0][1],pad[1][1])) # bottom right
    
#     out = npa.concatenate((
#         npa.concatenate((top_left,top,top_right)),
#         npa.concatenate((left,arr,right)),
#         npa.concatenate((bottom_left,bottom,bottom_right))    
#     ),axis=1)
    
#     return out

# def _zero_pad(arr, pad):
    
#     # fill sides
#     left = npa.tile(0,(pad[0][0],arr.shape[1])) # left side
#     right = npa.tile(0,(pad[0][1],arr.shape[1])) # right side
#     top = npa.tile(0,(arr.shape[0],pad[1][0])) # top side
#     bottom = npa.tile(0,(arr.shape[0],pad[1][1])) # bottom side
    
#     # fill corners
#     top_left = npa.tile(0, (pad[0][0],pad[1][0])) # top left
#     top_right = npa.tile(0, (pad[0][1],pad[1][0])) # top right
#     bottom_left = npa.tile(0, (pad[0][0],pad[1][1])) # bottom left
#     bottom_right = npa.tile(0, (pad[0][1],pad[1][1])) # bottom right
    
#     out = npa.concatenate((
#         npa.concatenate((top_left,top,top_right)),
#         npa.concatenate((left,arr,right)),
#         npa.concatenate((bottom_left,bottom,bottom_right))    
#     ),axis=1)
    
#     return out

# def simple_2d_filter(x,kernel,Lx,Ly):
#     """A simple 2d filter algorithm that is differentiable with autograd.
#     Uses a 2D fft approach since it is typically faster and preserves the shape
#     of the input and output arrays.
    
#     The ffts pad the operation to prevent any circular convolution garbage.
#     Parameters
#     ----------
#     x : array_like (2D)
#         Design parameters
#     Lx : float
#         Length of design region in X direction (interpolated grid points)
#     Ly : float
#         Length of design region in Y direction (interpolated grid points)
#     Returns
#     -------
#     array_like (2D)
#         Filtered design parameters.
#     """
#     # Get 2d parameter space shape
#     # Dimensions ajusted to spins grid coordinantes.
#     Nx, = Ly.shape
#     Ny, = Lx.shape
#     (kx,ky) = kernel.shape
    
#     # Ensure the input is 2D
#     x = x.reshape(Nx,Ny)
    
#     # pad the kernel and input to avoid circular convolution and
#     # to ensure boundary conditions are met.
#     kernel = _zero_pad(kernel,((kx,kx),(ky,ky)))
#     x = _edge_pad(x,((kx,kx),(ky,ky)))
    
#     # Transform to frequency domain for fast convolution
#     H = npa.fft.fft2(kernel)
#     X = npa.fft.fft2(x)
    
#     # Convolution (multiplication in frequency domain)
#     Y = H * X
    
#     # We need to fftshift since we padded both sides if each dimension of our input and kernel.
#     y = npa.fft.fftshift(npa.real(npa.fft.ifft2(Y)))
    
#     # Remove all the extra padding
#     y = _centered(y,(kx,ky))
        
#     return y 

# def conic_filter(x,radius,Lx,Ly):
#     '''A linear conic filter, also known as a "Hat" filter in the literature [1].
    
#     Parameters
#     ----------
#     x : array_like (2D)
#         Design parameters
#     radius : float
#         Filter radius (in "meep units")
#     Lx : float
#         Length of design region in X direction (interpolated grid points)
#     Ly : float
#         Length of design region in Y direction (interpolated grid points)
#     Returns
#     -------
#     array_like (2D)
#         Filtered design parameters.
    
#     References
#     ----------
#     [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
#     density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
#     '''    
#     # Formulate grid over entire design region
#     xv, yv = np.meshgrid(Ly, Lx, sparse=True, indexing='ij')
    
#     # Calculate kernel
#     rad_grid = 2*radius
#     kernel = np.where(np.abs(xv ** 2 + yv ** 2) <= rad_grid**2,(1-np.sqrt(abs(xv ** 2 + yv ** 2))/rad_grid),0)
    
#     # Normalize kernel
#     kernel = kernel / np.sum(kernel.flatten()) # Normalize the filter
    
#     # Filter the response
#     y = simple_2d_filter(x,kernel,Lx,Ly)
    
#     return y

# def geo_const_filter(x,radius,Lx,Ly):
#     '''Defines a circular region of maximum dielectric constant within the design space.
    
#     Parameters
#     ----------
#     x : array_like (2D)
#         Design parameters
#     radius : float
#         Circle radius (in "grid units")
#     Lx : float
#         Length of design region in X direction (interpolated grid points)
#     Ly : float
#         Length of design region in Y direction (interpolated grid points)
#     Returns
#     -------
#     array_like (2D)
#         Filtered design parameters.
#     '''    
#     # Get 2d parameter space shape
#     Nx, = Ly.shape
#     Ny, = Lx.shape
    
#     # Ensure the input is 2D
#     x = x.reshape(Nx,Ny)    
    
#     # Formulate grid over entire design region
#     xv, yv = np.meshgrid(Ly, Lx, sparse=True, indexing='ij')
    
#     # Calculate kernel
#     rad_grid = 2*radius
#     mask = np.where(np.abs(xv ** 2 + yv ** 2) <= rad_grid**2, 1, 0)

#     x = npa.where(mask, 1.0, x).flatten()    
    
#     return x

# '''
# # ------------------------------------------------------------------------------------ #
# Projection filters
# '''

# def tanh_projection(x,beta,eta):
#     '''Projection filter that thresholds the input parameters between 0 and 1. Typically
#     the "strongest" projection.
#     Parameters
#     ----------
#     x : array_like
#         Design parameters
#     beta : float
#         Thresholding parameter (0 to infinity). Dictates how "binary" the output will be.
#     eta: float
#         Threshold point (0 to 1)  
#     Returns
#     -------
#     array_like
#         Projected and flattened design parameters.
#     References
#     ----------
#     [1] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and robust 
#     formulations in topology optimization. Structural and Multidisciplinary Optimization, 43(6), 767-784.
#     '''
    
#     return (npa.tanh(beta*eta) + npa.tanh(beta*(x-eta))) / (npa.tanh(beta*eta) + npa.tanh(beta*(1-eta)))

# def get_conic_radius_from_eta_e(b, eta_e):
#     """Calculates the corresponding filter radius given the minimum length scale (b)
#     and the desired eroded threshold point (eta_e).
    
#     Parameters
#     ----------
#     b : float
#         Desired minimum length scale.
#     eta_e : float
#         Eroded threshold point (1-eta)
    
#     Returns
#     -------
#     float
#         Conic filter radius.
    
#     References
#     ----------
#     [1] Qian, X., & Sigmund, O. (2013). Topological design of electromechanical actuators with 
#     robustness toward over-and under-etching. Computer Methods in Applied 
#     Mechanics and Engineering, 253, 237-251.
#     [2] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and 
#     robust formulations in topology optimization. Structural and Multidisciplinary 
#     Optimization, 43(6), 767-784.
#     [3] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
#     density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
#     """
    
#     # Modified to be adjusted to spins-b grid units
#     b_grid = 2*b
    
#     if (eta_e >= 0.5) and (eta_e < 0.75):
#         return b_grid / (2*np.sqrt(eta_e-0.5))
#     elif (eta_e >= 0.75) and (eta_e <= 1):
#         return b_grid / (2-2*np.sqrt(1-eta_e))
#     else:
#         raise ValueError("The erosion threshold point (eta_e) must be between 0.5 and 1.0")
      
# def mapping(x,eta,beta,filter_radius,fab_radius,design_region_width,design_region_height):
#     # geometric constraints
#     constrained_field = geo_const_filter(x,fab_radius,design_region_width,design_region_height)
    
#     # filter
#     filtered_field = conic_filter(constrained_field,filter_radius,design_region_width,design_region_height)

#     # projection
#     projected_field = tanh_projection(filtered_field,beta,eta)

#     # interpolate to actual materials
#     return projected_field.flatten()     
        
###############################################################################
###############################################################################
###############################################################################

class Parametrization(metaclass=abc.ABCMeta):
    """Represents an abstract parametrization.

    All parametrizations should inherit from this class.
    """
    @abc.abstractmethod
    def get_structure(self) -> np.ndarray:
        """Produces the corresponding structure.

        `get_structure` assumes that the parametrization values represent a
        feasible structure. Call `project` before calling `get_structure` if
        there is a possibility that the parametrization values may represent an
        infeasible structure.

        Returns:
            A vector corresponding to the z.
        """
        raise NotImplementedError('get_structure method not defined')

    @abc.abstractmethod
    def calculate_gradient(self) -> LinearOperator:
        """Calculates the gradient of the parametrization.

        Note that implementations should consider caching the gradient if the
        operation is expensive.

        Returns:
            A linear operator that represents the Jacobian of the
            parametrization.
        """
        raise NotImplementedError('calculate_gradient not defined')

    def project(self) -> None:
        """Projects the parametrization to a feasible structure.

        The parametrization may be modified to have values that do not
        correspond to a feasible structure. Calling this method before
        get_structure causes the parametrization values to be modified so that
        the structure is in the feasible set.
        """
        pass

    def get_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:  # pylint: disable=R0201
        """Gets box constraints for the parametrization.

        Return a list of lower and upper bounds for each entry of the
        parametrization vector. Return `None` for a bound if it is
        unbounded. Return `None` for bounds if there are no bounds at all.
        """
        return None

    @abc.abstractmethod
    def encode(self) -> np.ndarray:
        """Encode the parametrization into vector form. """
        raise NotImplementedError('encode not implemented')

    @abc.abstractmethod
    def decode(self, vector: np.ndarray) -> None:
        """Decode the parametrization from vector form. """
        raise NotImplementedError('decode not implemented')

    def to_vector(self) -> np.ndarray:
        """Converts parametrization to vector representation.

        to_vector/from_vector are intended to be called by external clients.
        encode/decode are meant to be called internally.

        The difference between encode/decode and to_vector/from_vector
        lies in the fact that encode/decode are guaranteed to be symmetric
        whereas to_vector/from_vector need not be, i.e. decode(encode())
        should be an effective no-op whereas from_vector(to_vector()) might
        not be. Specifically, encode/decode converts between the parametrization
        and the raw vector, which may contain invalid values (e.g. negative
        radius). On the other hand, to_vector/from_vector guarantee that the
        parametrization is valid.
        """
        return self.encode()

    def from_vector(self, vector: np.ndarray) -> None:
        """Converts vector representation into parametrization. """
        self.decode(vector)
        self.project()

    def serialize(self) -> dict:
        """Serializes parametrization information.

        Serialize returns a dictionary of all the information necessary to
        recover the parametrization (via deserialize). This includes
        the current parametrization vector as well as any other parametrization
        metadata (e.g. etch fraction).
        """
        return {"vector": self.to_vector()}

    def deserialize(self, data):
        """Deserializes parametrization dictionary.

        Deserializes parametrization information that was serialized using
        serialize().
        """
        self.from_vector(data["vector"])


class DirectParam(Parametrization):
    """ Represents a direct parametrization.

    A direct parametrization holds the z-value of each pixel.
    Projection is defined to keep the z-value between 0 and 1.
    """
    def __init__(
            self, initial_value: np.ndarray,
            bounds: List[float] = (0, 1)) -> None:
        self.vector = np.array(initial_value).astype(float)
        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        # Expand upper and lower bounds into arrays.
        if isinstance(self.lower_bound, numbers.Number):
            self.lower_bound = (self.lower_bound, ) * len(self.vector)
        if isinstance(self.upper_bound, numbers.Number):
            self.upper_bound = (self.upper_bound, ) * len(self.vector)
        if self.lower_bound is None:
            self.lower_bound = (None, ) * len(self.vector)
        if self.upper_bound is None:
            self.upper_bound = (None, ) * len(self.vector)

    def get_structure(self) -> np.ndarray:
        return self.vector

    def project(self) -> None:
        # np.clip does not except None as valid bound.
        # Therefore, we change Nones to +/- inf.
        lower_bound = [
            b if b is not None else -np.inf for b in self.lower_bound
        ]
        upper_bound = [
            b if b is not None else np.inf for b in self.upper_bound
        ]
        self.vector = np.clip(self.vector, lower_bound, upper_bound)

    def get_bounds(self):
        return (self.lower_bound, self.upper_bound)

    def calculate_gradient(self) -> None:
        return sparse.eye(len(self.vector))

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        self.vector = vector


class CubicParam(Parametrization):
    """ Parametrization that interpolates a coarse grid to a finer grid that is
        used as z.

    initial_value = initial parametrization (on coarse grid) taking into account the
                        the symmetry and periodicity
    coarse_x, coarse_y = coarse grid
    fine_x, fine_y = fine grid
    symmetry = 2 element array that indicates with 0 or 1 if there is symmetry in the x
                    and or y direction (this is imposed on the coarse grid)
    periodicity = 2 element array that indicates with 0 or 1 if the boundaries in the x
                    and/or y direction are the same
    periods = 2 element array that indicates how many periods there are in the x and/or y
                    direction
    lower_bound, upper_bound = lower and upper bound of the parametrization
    bounds = lower and upper bound of the parametrization
    """
    def __init__(self,
                 initial_value: np.ndarray,
                 coarse_x: np.ndarray,
                 coarse_y: np.ndarray,
                 fine_x: np.ndarray,
                 fine_y: np.ndarray,
                 symmetry: np.ndarray = np.array([0, 0]),
                 periodicity: np.ndarray = np.array([0, 0]),
                 periods: np.ndarray = np.array([0, 0]),
                 lower_bound: Union[float, List[float]] = 0,
                 upper_bound: Union[float, List[float]] = 1,
                 bounds: List[float] = None) -> None:
        self.x_z = fine_x
        self.y_z = fine_y
        self.x_p = coarse_x
        self.y_p = coarse_y
        self.beta = 1 / 3  # relaxation factor of the fabrication constraint
        self.k = 4  # factor in the exponential in the sigmoid function used to discretize
        
        self.geometry_matrix, self.reverse_geometry_matrix = cubic_utils.make_geometry_matrix_cubic(
            (len(coarse_x), len(coarse_y)), symmetry, periodicity, periods)
        
        # correct the initial value
        if isinstance(initial_value, (float, int, complex)):
            self.vector = initial_value * np.ones(
                self.geometry_matrix.shape[1])
        elif len(initial_value) == self.reverse_geometry_matrix.shape[0]:
            self.vector = initial_value
        elif len(initial_value) == self.reverse_geometry_matrix.shape[1]:
            self.vector = self.reverse_geometry_matrix @ initial_value
        else:
            raise ValueError('Invalid initial value')

        # Make the interpolation matrix.
        #periodicity_phi2f = np.logical_and(periodicity, np.logical_not(periods))
        phi2f, _, _ = cubic_utils.CubicMatrices(self.x_z, self.y_z, self.x_p,
                                                self.y_p, periodicity)

        self.vec2f = phi2f @ self.geometry_matrix

        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    def set_k(self, k):
        ''' set the slope of the sigmoid function. '''
        self.k = k

    def get_structure(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        if self.k:
            return 1 / (1 + np.exp(-self.k * (2 * z_cubic - 1)))
        else:
            return z_cubic

    def calculate_gradient(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        if self.k:                    
            return sparse.diags(2 * self.k * np.exp(-self.k * (2 * z_cubic - 1)) /
                                     (1 + np.exp(-self.k * (2 * z_cubic - 1)))**2) @ self.vec2f
        else:
            return self.vec2f

    def get_bounds(self):
        vec_len = len(self.vector)
        return ((self.lower_bound, ) * vec_len, (self.upper_bound, ) * vec_len)

    def project(self) -> None:
        self.vector = np.clip(self.vector, self.lower_bound, self.upper_bound)

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        self.vector = vector

    def serialize(self) -> Dict:
        return {
            "vector": self.to_vector(),
            "sigmoid_strength": self.k,
        }

    def deserialize(self, state: Dict) -> None:
        self.from_vector(state["vector"])
        self.k = state["sigmoid_strength"]

    # functions to fit the parametrization
    def fit2eps(self, eps_bg, S, eps):
        from spins.invdes.problem import Fit2Eps, OptimizationProblem
        import spins.invdes.optimization as optim

        # make objective
        obj = Fit2Eps(eps_bg, S, eps)
        obj = OptimizationProblem(obj)

        # optimize continuous
        opt_cont = optim.ScipyOptimizer(method='L-BFGS-B',
                                        options={
                                            'maxiter': 200,
                                            'maxcor': 10
                                        })
        iter_num = 0

        def callback(_):
            nonlocal iter_num
            iter_num += 1

        opt_cont(obj, self, callback=callback)

    # Functions to generate gds
    def generate_polygons(self, dx: float):
        '''
            Generate a list of polygons

            input:
                dx: grid spacing
            output:
                list of the polygons
        '''
        x_z = self.x_z * dx / 2
        y_z = self.y_z * dx / 2
        design_area_fine = np.array([len(x_z), len(y_z)])
        phi = self.vec2f @ self.vector
        phi_mat = phi.reshape(design_area_fine, order='F')

        # Pad the design region with zeros to ensure outer boundary is drawn.
        phi_extended = np.zeros(design_area_fine + 2)
        phi_extended[1:-1, 1:-1] = phi_mat

        x_extended = np.r_[x_z[0] - dx / 2, x_z, x_z[-1] + dx / 2]
        y_extended = np.r_[y_z[0] - dx / 2, y_z, y_z[-1] + dx / 2]

        import matplotlib.pyplot as plt
        """ The line above was rewrited in order to accomplish non squared 
        design regions, so avoiding the following error: 
        TypeError: Length of x (202) must match number of columns in z (102) ---."""
        #cs = plt.contour(x_extended, y_extended, phi_extended - 0.5, [0])        
        cs = plt.contour(x_extended, y_extended, np.rot90(phi_extended) - 0.5, [0])
        paths = cs.collections[0].get_paths()

        return [p.to_polygons()[0] for p in paths]


class HermiteParam(Parametrization):
    """ Parametrization that interpolates coarse grid value and derivatives to a finer
        grid that is used as z. The parametrization is defined at the coarse grid by
        f, df/dx, df/dy and d^2f/dxdy. The parametrization vector is thus 4 times len(
        coarse_x)*len(coarse_y)

    initial_value = initial parametrization (on coarse grid) taking into account the
                        the symmetry and periodicity
    coarse_x, coarse_y = coarse grid
    fine_x, fine_y = fine grid
    symmetry = 2 element array that indicates with 0 or 1 if there is symmetry in the x
                    and or y direction (this is imposed on the coarse grid)
    periodicity = 2 element array that indicates with 0 or 1 if the boundaries in the x
                    and/or y direction are the same
    periods = 2 element array that indicates how many periods there are in the x and/or y
                    direction
    lower_bound, upper_bound = lower and upper bound of the parametrization
    bounds = lower and upper bound of the parametrization
    scale = scaling factor for the derivatives
    """
    def __init__(self,
                 initial_value: np.ndarray,
                 coarse_x: np.ndarray,
                 coarse_y: np.ndarray,
                 fine_x: np.ndarray,
                 fine_y: np.ndarray,
                 symmetry: np.ndarray = np.array([0, 0]),
                 periodicity: np.ndarray = np.array([0, 0]),
                 periods: np.ndarray = np.array([0, 0]),
                 lower_bound: Union[float, List[float]] = -np.inf,
                 upper_bound: Union[float, List[float]] = np.inf,
                 bounds: List[float] = None,
                 scale: float = 1.75) -> None:
        self.x_z = fine_x
        self.y_z = fine_y
        self.x_p = coarse_x
        self.y_p = coarse_y
        self.beta = 1 / 3  # relaxation factor of the fabrication constraint
        self.k = 4  # factor in the exponential in the sigmoid function used to discretize
        self.scale_deriv = scale

        self.fine_x_grid, self.fine_y_grid = np.meshgrid(fine_x,
                                                         fine_y,
                                                         indexing='ij')

        self.geometry_matrix, self.reverse_geometry_matrix = cubic_utils.make_geometry_matrix_hermite(
            (len(coarse_x), len(coarse_y)), symmetry, periodicity, periods)

        # correct the initial value
        self.derivative_matrix = cubic_utils.idxdydxy_matrix(
            coarse_x,
            coarse_y,
            deriv_scaling=np.array([
                1, scale * np.diff(fine_x).mean(),
                scale**2 * np.diff(fine_x).mean()**2
            ]))

        # correct the initial value
        if isinstance(initial_value, (float, int, complex)):
            self.vector = initial_value * np.ones(
                self.geometry_matrix.shape[1])
        elif len(initial_value) == self.geometry_matrix.shape[1]:
            self.vector = initial_value
        elif len(initial_value) == self.geometry_matrix.shape[0]:
            self.vector = self.reverse_geometry_matrix @ initial_value
        elif len(initial_value) == self.derivative_matrix.shape[1]:
            self.vector = self.reverse_geometry_matrix @ \
                self.derivative_matrix @ initial_value
        #TODO vcruysse: account for the following cases
        #elif len(initial_value) == symmetry_matrix.shape[1]*4:
        #elif len(initial_value) == symmetry_matrix.shape[1]:
        #elif len(initial_value) == periodic_matrix_n.shape[0]*4:
        #elif len(initial_value) == periodic_matrix_n.shape[0]:
        else:
            raise ValueError('Invalid initial value')

        # Make the interpolation matrix.
        phi2f, _, _ = cubic_utils.CubicMatrices(
            fine_x,
            fine_y,
            coarse_x,
            coarse_y,
            periodicity,
            derivatives=True,
            deriv_scaling=np.array([
                1, scale * np.diff(fine_x).mean(),
                scale**2 * np.diff(fine_x).mean()**2
            ]))
        self.vec2f = phi2f @ self.geometry_matrix

        # Set bounds
        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    def get_structure(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        return 1 / (1 + np.exp(-self.k * (2 * z_cubic - 1)))

    def calculate_gradient(self) -> np.ndarray:
        z_cubic = self.vec2f @ self.vector
        return sparse.diags(2 * self.k * np.exp(-self.k * (2 * z_cubic - 1)) /
                            (1 + np.exp(-self.k *
                                        (2 * z_cubic - 1)))**2) @ self.vec2f

    def set_k(self, k):
        self.k = k

    def project(self) -> None:
        self.vector = np.clip(self.vector, self.lower_bound, self.upper_bound)

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        if isinstance(vector, (float, int, complex)):
            self.vector = vector * np.ones(self.geometry_matrix.shape[1])
        elif len(vector) == self.geometry_matrix.shape[1]:
            self.vector = vector
        elif len(vector) == self.geometry_matrix.shape[0]:
            self.vector = self.reverse_geometry_matrix @ vector
        #TODO vcruysse: account for the following cases
        #elif len(initial_value) == symmetry_matrix.shape[1]*4:
        #elif len(initial_value) == symmetry_matrix.shape[1]:
        #elif len(initial_value) == periodic_matrix_n.shape[0]*4:
        #elif len(initial_value) == periodic_matrix_n.shape[0]:
        else:
            raise ValueError('Invalid initial value')

    def serialize(self) -> Dict:
        return {
            "vector": self.to_vector(),
            "sigmoid_strength": self.k,
        }

    def deserialize(self, state: Dict) -> None:
        self.from_vector(state["vector"])
        self.k = state["sigmoid_strength"]

    # functions to fit the parametrization
    def fit2eps(self, eps_bg, S, eps):
        from spins.invdes.problem import Fit2Eps, OptimizationProblem
        import spins.invdes.optimization as optim

        # make objective
        obj = Fit2Eps(eps_bg, S, eps)
        obj = OptimizationProblem(obj)

        # optimize continuous
        opt_cont = optim.ScipyOptimizer(method='L-BFGS-B',
                                        options={
                                            'maxiter': 200,
                                            'maxcor': 10
                                        })
        iter_num = 0

        def callback(v):
            nonlocal iter_num
            iter_num += 1
            print('fit2eps-continous: ' + str(iter_num))

        opt_cont(obj, self, callback=callback)
        
        
# class CubicParamDensityFilter(Parametrization):
#     """ Parametrization that interpolates a coarse grid to a finer grid that is
#         used as z.

#     initial_value = initial parametrization (on coarse grid) taking into account the
#                         the symmetry and periodicity
#     coarse_x, coarse_y = coarse grid
#     fine_x, fine_y = fine grid
#     symmetry = 2 element array that indicates with 0 or 1 if there is symmetry in the x
#                     and or y direction (this is imposed on the coarse grid)
#     periodicity = 2 element array that indicates with 0 or 1 if the boundaries in the x
#                     and/or y direction are the same
#     periods = 2 element array that indicates how many periods there are in the x and/or y
#                     direction
#     lower_bound, upper_bound = lower and upper bound of the parametrization
#     bounds = lower and upper bound of the parametrization
#     """
#     def __init__(self,
#                  initial_value: np.ndarray,
#                  coarse_x: np.ndarray,
#                  coarse_y: np.ndarray,
#                  fine_x: np.ndarray,
#                  fine_y: np.ndarray,
#                  symmetry: np.ndarray = np.array([0, 0]),
#                  periodicity: np.ndarray = np.array([0, 0]),
#                  periods: np.ndarray = np.array([0, 0]),
#                  lower_bound: Union[float, List[float]] = 0,
#                  upper_bound: Union[float, List[float]] = 1,
#                  bounds: List[float] = None,
#                  k = None,
#                  center_rad = float,
#                  center_x = 0,
#                  center_y = 0,
#                  min_feature = float,
#                  eta_i = float,
#                  eta_e = float,
#                  eta_d = float) -> None:
#         self.x_z = fine_x
#         self.y_z = fine_y
#         self.x_p = coarse_x
#         self.y_p = coarse_y
#         self.symmetry = symmetry
#         self.periodicity = periodicity
#         self.periods = periods
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
#         self.bounds = bounds        

#         self.k = 1 if k is None else k      # beta factor in tanh projection function.
#         self.center_rad = center_rad        # Center radius of fabrication constraint.
#         self.m_feature = min_feature
#         self.radius = get_conic_radius_from_eta_e(self.m_feature, eta_e)
#         self.eta_i = eta_i
#         self.eta_e = eta_e
#         self.eta_d = eta_d        
        
#         self.geometry_matrix, self.reverse_geometry_matrix = cubic_utils.make_geometry_matrix_cubic(
#             (len(coarse_x), len(coarse_y)), symmetry, periodicity, periods)
        
#         # correct the initial value
#         if isinstance(initial_value, (float, int, complex)):
#             self.vector = initial_value * np.ones(
#                 self.geometry_matrix.shape[1])
#         elif len(initial_value) == self.reverse_geometry_matrix.shape[0]:
#             self.vector = initial_value
#         elif len(initial_value) == self.reverse_geometry_matrix.shape[1]:
#             self.vector = self.reverse_geometry_matrix @ initial_value
#         else:
#             raise ValueError('Invalid initial value')

#         # Make the interpolation matrix.
#         #periodicity_phi2f = np.logical_and(periodicity, np.logical_not(periods))
#         phi2f, _, _ = cubic_utils.CubicMatrices(self.x_z, self.y_z, self.x_p,
#                                                 self.y_p, periodicity)

#         self.vec2f = phi2f @ self.geometry_matrix

#         # Set bounds
#         if bounds:
#             self.lower_bound = bounds[0]
#             self.upper_bound = bounds[1]
#         else:
#             self.lower_bound = lower_bound
#             self.upper_bound = upper_bound
        
#         # Build the conic filter        
#         dx, dy = np.meshgrid(range(-math.ceil(self.radius)+1, math.ceil(self.radius)),
#                              range(-math.ceil(self.radius)+1, math.ceil(self.radius)), indexing='ij')

#         self.h = np.where(np.abs(dx**2 + dy**2) <= self.radius**2, (1 - np.sqrt(abs(dx**2 + dy**2))/self.radius), 0)
#         self.h = self.h / np.sum(self.h.flatten()) # Normalize the filter
#         self.Nx, = self.x_z.shape
#         self.Ny, = self.y_z.shape
#         self.hs = signal.convolve2d(np.ones((self.Ny, self.Nx)), self.h, boundary='fill', mode='same')
        
#         # Build the geometric mask
#         xv, yv = np.meshgrid(self.y_z, self.x_z, sparse=True, indexing='ij')        
#         self.geo_mask = np.where(np.abs((xv-center_y)**2 + (yv-center_x)**2) <= (2*self.center_rad)**2, 1, 0)
    

#     def set_k(self, k):
#         ''' set the slope of the sigmoid function. '''
#         self.k = k

#     def get_structure(self) -> np.ndarray:
# #        z_cubic = self.vec2f @ self.vector        
# #        if self.k:
# #            z = mapping(z_cubic, self.eta_i, self.k, self.radius, self.center_rad, self.x_z, self.y_z)
# ##            print("z")
# ##            print(z.shape)
# ##            print(z)
# #            return z
# #        else:
# #            return z_cubic  
        
#         z = self.vec2f @ self.vector               
#         if self.m_feature > 0:
#             z = mapping(z, self.eta_i, self.k, self.radius, self.center_rad, self.x_z, self.y_z)
#             return z 
#         else:
#             if self.k:
#                 # Ensure the input is 2D
#                 z_grid = z.reshape(self.Ny, self.Nx)
#                 z_proj = (np.tanh(self.k*self.eta_i) + np.tanh(self.k*(z_grid-self.eta_i))) / (np.tanh(self.k*self.eta_i) + np.tanh(self.k*(1-self.eta_i)))
#                 z_geo = np.where(self.geo_mask, 1.0, z_proj)
#                 #z_conic = np.divide(signal.fftconvolve(z_grid_proj, self.h, mode='same'), self.hs)                                  
#                 return z_geo.flatten()        
#             else:
#                 return z

#     def calculate_gradient(self) -> np.ndarray:
# #        z_cubic = self.vec2f @ self.vector
# #        if self.k:            
# #            grad_z = tensor_jacobian_product(mapping,0)(z_cubic, self.eta_i, self.k, self.radius, self.center_rad, self.x_z, self.y_z,
# #                                                        np.ones(z_cubic.shape))
# #            dz = sparse.diags(grad_z) @ self.vec2f
# ##            print("dz")
# ##            print(dz.shape)
# ##            print(dz)            
# #            return dz
# #        else:
# #            return self.vec2f
        
#         z = self.vec2f @ self.vector
#         if self.m_feature > 0:
#             grad_z = tensor_jacobian_product(mapping,0)(z, self.eta_i, self.k, self.radius, self.center_rad, self.x_z, self.y_z,
#                                                         np.ones(z.shape))
#             dz = sparse.diags(grad_z) @ self.vec2f           
#             return dz
#         else:
#             z_grid = z.reshape(self.Ny, self.Nx)
#             dz_proj = np.multiply((1-np.tanh(self.k*(z_grid - self.eta_i))**2), self.k/(np.tanh(self.k*self.eta_i)+np.tanh(self.k*(1-self.eta_i))))
#             dz_geo = np.multiply(np.where(self.geo_mask, 0.0, 1.0), dz_proj)
#             #dz_conic = signal.fftconvolve(np.divide(dz_grid_proj, self.hs), self.h, mode='same')
#             dz = sparse.diags(dz_geo.flatten()) @ self.vec2f            
#             return dz       
        
#     def get_bounds(self):
#         vec_len = len(self.vector)
#         return ((self.lower_bound, ) * vec_len, (self.upper_bound, ) * vec_len)

#     def project(self) -> None:
#         self.vector = np.clip(self.vector, self.lower_bound, self.upper_bound)

#     def encode(self) -> np.ndarray:
#         return self.vector

#     def decode(self, vector: np.ndarray) -> None:
#         self.vector = vector

#     def serialize(self) -> Dict:
#         return {
#             "vector": self.to_vector(),
#             "sigmoid_strength": self.k,
#         }

#     def deserialize(self, state: Dict) -> None:
#         self.from_vector(state["vector"])
#         self.k = state["sigmoid_strength"]

#     # functions to fit the parametrization
#     def fit2eps(self, eps_bg, S, eps):
#         from spins.invdes.problem import Fit2Eps, OptimizationProblem
#         import spins.invdes.optimization as optim

#         # make objective
#         obj = Fit2Eps(eps_bg, S, eps)
#         obj = OptimizationProblem(obj)

#         # optimize continuous
#         opt_cont = optim.ScipyOptimizer(method='L-BFGS-B',
#                                         options={
#                                             'maxiter': 200,
#                                             'maxcor': 10
#                                         })
#         iter_num = 0

#         def callback(_):
#             nonlocal iter_num
#             iter_num += 1

#         opt_cont(obj, self, callback=callback)

#     # Functions to generate gds
#     def generate_polygons(self, dx: float):
#         '''
#             Generate a list of polygons

#             input:
#                 dx: grid spacing
#             output:
#                 list of the polygons
#         '''
#         x_z = self.x_z * dx / 2
#         y_z = self.y_z * dx / 2
#         design_area_fine = np.array([len(x_z), len(y_z)])
#         phi = self.vec2f @ self.vector
#         phi_mat = phi.reshape(design_area_fine, order='F')

#         # Pad the design region with zeros to ensure outer boundary is drawn.
#         phi_extended = np.zeros(design_area_fine + 2)
#         phi_extended[1:-1, 1:-1] = phi_mat

#         x_extended = np.r_[x_z[0] - dx / 2, x_z, x_z[-1] + dx / 2]
#         y_extended = np.r_[y_z[0] - dx / 2, y_z, y_z[-1] + dx / 2]

#         import matplotlib.pyplot as plt
#         """ The line above was rewrited in order to accomplish non squared 
#         design regions, so avoiding the following error: 
#         TypeError: Length of x (202) must match number of columns in z (102) ---."""
#         #cs = plt.contour(x_extended, y_extended, phi_extended - 0.5, [0])        
#         cs = plt.contour(x_extended, y_extended, np.rot90(phi_extended) - 0.5, [0])
#         paths = cs.collections[0].get_paths()

#         return [p.to_polygons()[0] for p in paths]  


class CubicParamDensityFilter(Parametrization):
    """ Parametrization that interpolates a coarse grid to a finer grid that is
        used as z.

    initial_value = initial parametrization (on coarse grid) taking into account the
                        the symmetry and periodicity
    coarse_x, coarse_y = coarse grid
    fine_x, fine_y = fine grid
    symmetry = 2 element array that indicates with 0 or 1 if there is symmetry in the x
                    and or y direction (this is imposed on the coarse grid)
    periodicity = 2 element array that indicates with 0 or 1 if the boundaries in the x
                    and/or y direction are the same
    periods = 2 element array that indicates how many periods there are in the x and/or y
                    direction
    lower_bound, upper_bound = lower and upper bound of the parametrization
    bounds = lower and upper bound of the parametrization
    """
    def __init__(self,
                 initial_value: np.ndarray,
                 coarse_x: np.ndarray,
                 coarse_y: np.ndarray,
                 fine_x: np.ndarray,
                 fine_y: np.ndarray,
                 symmetry: np.ndarray = np.array([0, 0]),
                 periodicity: np.ndarray = np.array([0, 0]),
                 periods: np.ndarray = np.array([0, 0]),
                 lower_bound: Union[float, List[float]] = 0,
                 upper_bound: Union[float, List[float]] = 1,
                 bounds: List[float] = None,
                 k = None,
                 center_rad = None,
                 center_x = None,
                 center_y = None,
                 min_feature = None,
                 grid_res = None,
                 eta_i = float,
                 eta_e = float,
                 eta_d = float) -> None:
        self.x_z = fine_x
        self.y_z = fine_y
        self.x_p = coarse_x
        self.y_p = coarse_y
        self.symmetry = symmetry
        self.periodicity = periodicity
        self.periods = periods
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bounds = bounds        

        self.k = 1 if k is None else k              # beta factor in tanh projection function.
        self.center_rad = center_rad                # Radius of center circle on design region (grid units).
        self.center_x = center_x                    # x position of center circle (grid units).
        self.center_y = center_y                    # y position of center circle (grid units).
        self.m_feature = min_feature                # minimum fabrication feature size (um).
        self.grid_res = grid_res                    # Simulation resolution (um).
        self.fab_rad = get_conic_radius_from_eta_e(self.m_feature, eta_e)
        self.eta_i = eta_i
        self.eta_e = eta_e
        self.eta_d = eta_d    
        
        self.geometry_matrix, self.reverse_geometry_matrix = cubic_utils.make_geometry_matrix_cubic(
            (len(coarse_x), len(coarse_y)), symmetry, periodicity, periods)
        
        # correct the initial value
        if isinstance(initial_value, (float, int, complex)):
            self.vector = initial_value * np.ones(
                self.geometry_matrix.shape[1])
        elif len(initial_value) == self.reverse_geometry_matrix.shape[0]:
            self.vector = initial_value
        elif len(initial_value) == self.reverse_geometry_matrix.shape[1]:
            self.vector = self.reverse_geometry_matrix @ initial_value
        else:
            raise ValueError('Invalid initial value')

        # Make the interpolation matrix.
        #periodicity_phi2f = np.logical_and(periodicity, np.logical_not(periods))
        phi2f, _, _ = cubic_utils.CubicMatrices(self.x_z, self.y_z, self.x_p,
                                                self.y_p, periodicity)

        self.vec2f = phi2f @ self.geometry_matrix

        # Set bounds
        if bounds:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
        
        # Build the conic filter parameters        
        self.Nx, = self.x_z.shape
        self.Ny, = self.y_z.shape
        self.width = (self.Nx*self.grid_res)
        self.height = (self.Ny*self.grid_res)        
        
        # Build the geometric mask
        xv, yv = np.meshgrid(self.y_z, self.x_z, sparse=True, indexing='ij')        
        self.geo_mask = np.where(np.abs((xv-center_y)**2 + (yv-center_x)**2) <= (2*self.center_rad)**2, 1, 0).flatten() 
    

    def set_k(self, k):
        ''' set the slope of the sigmoid function. '''
        self.k = k

    def get_structure(self) -> np.ndarray:        
        z = self.vec2f @ self.vector              
        if self.k > 0:
            z_geo = np.where(self.geo_mask, 1.0, z)
            z_proj = mapping(z_geo, self.eta_i, self.k, self.fab_rad, self.height, self.width,1.0/self.grid_res)
            return z_proj 
        else:          
            return z

    def calculate_gradient(self) -> np.ndarray:
        z = self.vec2f @ self.vector
        if self.k > 0:
            z_geo = np.where(self.geo_mask, 1.0, z)
            grad_z = tensor_jacobian_product(mapping,0)(z_geo, self.eta_i, self.k, self.fab_rad, self.height, self.width,1.0/self.grid_res,
                                                        np.ones(z_geo.shape))
            dz_geo = np.multiply(np.where(self.geo_mask, 0.0, 1.0), grad_z)
            dz = sparse.diags(dz_geo) @ self.vec2f           
            return dz
        else:
            dz = sparse.diags(np.ones(z.shape)) @ self.vec2f            
            return dz       
        
    def get_bounds(self):
        vec_len = len(self.vector)
        return ((self.lower_bound, ) * vec_len, (self.upper_bound, ) * vec_len)

    def project(self) -> None:
        self.vector = np.clip(self.vector, self.lower_bound, self.upper_bound)

    def encode(self) -> np.ndarray:
        return self.vector

    def decode(self, vector: np.ndarray) -> None:
        self.vector = vector

    def serialize(self) -> Dict:
        return {
            "vector": self.to_vector(),
            "sigmoid_strength": self.k,
        }

    def deserialize(self, state: Dict) -> None:
        self.from_vector(state["vector"])
        self.k = state["sigmoid_strength"]

    # functions to fit the parametrization
    def fit2eps(self, eps_bg, S, eps):
        from spins.invdes.problem import Fit2Eps, OptimizationProblem
        import spins.invdes.optimization as optim

        # make objective
        obj = Fit2Eps(eps_bg, S, eps)
        obj = OptimizationProblem(obj)

        # optimize continuous
        opt_cont = optim.ScipyOptimizer(method='L-BFGS-B',
                                        options={
                                            'maxiter': 200,
                                            'maxcor': 10
                                        })
        iter_num = 0

        def callback(_):
            nonlocal iter_num
            iter_num += 1

        opt_cont(obj, self, callback=callback)

    # Functions to generate gds
    def generate_polygons(self, dx: float):
        '''
            Generate a list of polygons

            input:
                dx: grid spacing
            output:
                list of the polygons
        '''
        x_z = self.x_z * dx / 2
        y_z = self.y_z * dx / 2
        design_area_fine = np.array([len(x_z), len(y_z)])
        phi = self.vec2f @ self.vector
        phi_mat = phi.reshape(design_area_fine, order='F')

        # Pad the design region with zeros to ensure outer boundary is drawn.
        phi_extended = np.zeros(design_area_fine + 2)
        phi_extended[1:-1, 1:-1] = phi_mat

        x_extended = np.r_[x_z[0] - dx / 2, x_z, x_z[-1] + dx / 2]
        y_extended = np.r_[y_z[0] - dx / 2, y_z, y_z[-1] + dx / 2]

        import matplotlib.pyplot as plt
        """ The line above was rewrited in order to accomplish non squared 
        design regions, so avoiding the following error: 
        TypeError: Length of x (202) must match number of columns in z (102) ---."""
        #cs = plt.contour(x_extended, y_extended, phi_extended - 0.5, [0])        
        cs = plt.contour(x_extended, y_extended, np.rot90(phi_extended) - 0.5, [0])
        paths = cs.collections[0].get_paths()

        return [p.to_polygons()[0] for p in paths]        