# -*- coding: utf-8 -*-
# This code clone from https://github.com/manu-mannattil/nolitsa
# BSD-3 LICENSE : https://github.com/manu-mannattil/nolitsa/blob/master/LICENSE
# Change np.fft to scipt.fftpack

"""Functions to generate surrogate series.

This module provides a set of functions to generate surrogate series
from a given time series using multiple algorithms.

Surrogates Generation
---------------------

  * ft -- generates Fourier transform surrogates.
  * aaft -- generates amplitude adjusted Fourier transform surrogates.
  * iaaft -- generates iterative amplitude adjusted Fourier transform
    surrogates.

Utilities
---------

  * mismatch -- finds the segment of a time series with the least
    end-point mismatch.
"""

import numpy as np
import scipy


def rescale(x, interval=(0, 1)):
    """Rescale the given scalar time series into a desired interval.

    Rescales the given scalar time series into a desired interval using
    a simple linear transformation.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    interval: tuple, optional (default = (0, 1))
        Extent of the interval specified as a tuple.

    Returns
    -------
    y : array
        Rescaled scalar time series.
    """
    x = np.asarray(x)
    if interval[1] == interval[0]:
        raise ValueError('Interval must have a nonzero length.')

    return (interval[0] + (x - np.min(x)) * (interval[1] - interval[0]) /
            (np.max(x) - np.min(x)))


def ft(x):
    """Return simple Fourier transform surrogates.

    Returns phase randomized (FT) surrogates that preserve the power
    spectrum (or equivalently the linear correlations), but completely
    destroy the probability distribution.

    Parameters
    ----------
    x : array
        Real input array containg the time series.

    Returns
    -------
    y : array
        Surrogates with the same power spectrum as x.
    """
    y = scipy.fftpack.rfft(x)

    phi = 2 * np.pi * np.random.random(len(y))

    phi[0] = 0.0
    if len(x) % 2 == 0:
        phi[-1] = 0.0

    y = y * np.exp(1j * phi)
    return scipy.fftpack.irfft(np.real(y), n=len(x))


def aaft(x):
    """Return amplitude adjusted Fourier transform surrogates.

    Returns phase randomized, amplitude adjusted (AAFT) surrogates with
    crudely the same power spectrum and distribution as the original
    data (Theiler et al. 1992).  AAFT surrogates are used in testing
    the null hypothesis that the input series is correlated Gaussian
    noise transformed by a monotonic time-independent measuring
    function.

    Parameters
    ----------
    x : array
        1-D input array containg the time series.

    Returns
    -------
    y : array
        Surrogate series with (crudely) the same power spectrum and
        distribution.
    """
    # Generate uncorrelated Gaussian random numbers.
    y = np.random.normal(size=len(x))

    # Introduce correlations in the random numbers by rank ordering.
    y = np.sort(y)[np.argsort(np.argsort(x))]
    y = ft(y)

    return np.sort(x)[np.argsort(np.argsort(y))]


def iaaft(x, maxiter=1000, atol=1e-8, rtol=1e-10):
    """Return iterative amplitude adjusted Fourier transform surrogates.

    Returns phase randomized, amplitude adjusted (IAAFT) surrogates with
    the same power spectrum (to a very high accuracy) and distribution
    as the original data using an iterative scheme (Schreiber & Schmitz
    1996).

    Parameters
    ----------
    x : array
        1-D real input array of length N containing the time series.
    maxiter : int, optional (default = 1000)
        Maximum iterations to be performed while checking for
        convergence.  The scheme may converge before this number as
        well (see Notes).
    atol : float, optional (default = 1e-8)
        Absolute tolerance for checking convergence (see Notes).
    rtol : float, optional (default = 1e-10)
        Relative tolerance for checking convergence (see Notes).

    Returns
    -------
    y : array
        Surrogate series with (almost) the same power spectrum and
        distribution.
    i : int
        Number of iterations that have been performed.
    e : float
        Root-mean-square deviation (RMSD) between the absolute squares
        of the Fourier amplitudes of the surrogate series and that of
        the original series.

    Notes
    -----
    To check if the power spectrum has converged, we see if the absolute
    difference between the current (cerr) and previous (perr) RMSDs is
    within the limits set by the tolerance levels, i.e., if abs(cerr -
    perr) <= atol + rtol*perr.  This follows the convention used in
    the NumPy function numpy.allclose().

    Additionally, atol and rtol can be both set to zero in which
    case the iterations end only when the RMSD stops changing or when
    maxiter is reached.
    """
    # Calculate "true" Fourier amplitudes and sort the series.
    ampl = np.abs(scipy.fftpack.rfft(x))
    sort = np.sort(x)

    # Previous and current error.
    perr, cerr = (-1, 1)

    # Start with a random permutation.
    t = scipy.fftpack.rfft(np.random.permutation(x))

    for i in range(maxiter):
        # Match power spectrum.
        s = np.real(scipy.fftpack.irfft(ampl * t / np.abs(t), n=len(x)))

        # Match distribution by rank ordering.
        y = sort[np.argsort(np.argsort(s))]

        t = scipy.fftpack.rfft(y)
        cerr = np.sqrt(np.mean((ampl ** 2 - np.abs(t) ** 2) ** 2))

        # Check convergence.
        if abs(cerr - perr) <= atol + rtol * abs(perr):
            break
        else:
            perr = cerr

    # Normalize error w.r.t. mean of the "true" power spectrum.
    return y, i, cerr / np.mean(ampl ** 2)

def app(x,alpha=1.0):
    """amplitude and phase perturbations

    Parameters
    ----------
    x : array
        1-D real input array of length N containing the time series.
    alpha : float
        strength of app

     Returns
    -------
    y : array
        1-D output time series.
    """
    length = x.shape[0]
    x_fft = scipy.fftpack.fft(x)
    amp = np.abs(x_fft)
    phase = np.angle(x_fft)
    amp_std = np.std(amp)
    phase_std = np.std(phase)
    pos_a = np.sort(np.random.randint(0, length, 2))
    pos_p = np.sort(np.random.randint(0, length, 2))
    if pos_a[1]-pos_a[0]>10 and pos_p[1]-pos_p[0]>10:
        amp[pos_a[0]:pos_a[1]] = amp[pos_a[0]:pos_a[1]] + np.random.normal(0,alpha,pos_a[1]-pos_a[0])*amp_std
        phase[pos_p[0]:pos_p[1]] = phase[pos_p[0]:pos_p[1]] + np.random.normal(0,alpha,pos_p[1]-pos_p[0])*phase_std
    fft_re = amp*np.exp(1j*phase)
    y = scipy.fftpack.ifft(fft_re)
    return np.real(y)



def mismatch(x, length=None, weight=0.5, neigh=3):
    """Find the segment that minimizes end-point mismatch.

    Finds the segment in the time series that has minimum end-point
    mismatch.  To do this we calculate the mismatch between the end
    points of all segments of the given length and pick the segment with
    least mismatch (Ehlers et al. 1998).  We also enforce the
    condition that the difference between the first derivatives at the
    end points must be a minimum.

    Parameters
    ----------
    x : array
        Real input array containg the time series.
    length : int, optional
        Length of segment.  By default the largest possible length which
        is a power of one of the first five primes is selected.
    weight : float, optional (default = 0.5)
        Weight given to discontinuity in the first difference of the
        time series.  Must be between 0 and 1.
    neigh : int, optional (default = 3)
        Num of end points using which the discontinuity statistic should
        be computed.

    Returns
    -------
    ends : tuple
        Indices of the end points of the segment.
    d : float
        Discontinuity statistic for the segment.

    Notes
    -----
    Both the time series and its first difference are linearly rescaled
    to [0, 1].  Thus the discontinuity statistic varies between 0 and 1
    (0 means no discontinuity and 1 means maximum discontinuity).
    """
    # Calculate the first difference of the time series and rescale it
    # to [0, 1]
    dx = rescale(np.diff(x))
    x = rescale(x)[1:]
    n = len(x)

    if not length:
        primes = np.array([2, 3, 5, 7, 11])
        i = np.argmax(primes ** np.floor(np.log(n) / np.log(primes)) - n)
        length = int(primes[i] ** (np.floor(np.log(n) / np.log(primes[i]))))

    d = np.zeros(n - (length + neigh))

    for i in np.arange(n - (length + neigh)):
        d[i] = ((1 - weight) * (np.mean((x[i:i + neigh] -
                                x[i + length:i + length + neigh]) ** 2.0)) +
                weight * (np.mean((dx[i:i + neigh] -
                          dx[i + length:i + length + neigh]) ** 2.0)))

    return (1 + np.argmin(d), 1 + np.argmin(d) + length), np.min(d)


#####################################自己复现的代码，iaaft是错的#################################
"""
def rank_like(src,dst):
    src = np.sort(src)
    sort_index = np.argsort(dst)
    src_new = np.zeros_like(src)
    src_new[sort_index] = src[:]
    return src_new

def aaft(signal):
    # step 1 
    Xs = np.random.randn(len(signal))
    # step 2
    Xs = rank_like(Xs, signal)
    # step 3
    Xs_fft = fft(Xs)
    Xs_angle = np.angle(Xs_fft)
    np.random.shuffle(Xs_angle)
    Xs_fft_re = np.abs(Xs_fft)*np.exp(1j*Xs_angle)
    Xs_re = ifft(Xs_fft_re)
    # step 4
    signal_new = rank_like(signal, Xs_re)
    return signal_new

def iaaft(signal,iter=10):
    Ck = np.argsort(signal)
    Ak = fft(signal)
    Ak_abs = np.abs(Ak)
    #Pk = np.angle(X_fft)
    Sn = aaft(signal)

    for i in range(iter):
        Sk = fft(Sn)
        Sk_angle = np.angle(Sk)
        Sk_1 = Ak_abs*np.exp(1j*Sk_angle)

        Sn = ifft(Sk_1)

        Sn = rank_like(Sn, signal)
    return Sn
"""