"""

This module contains functions to reshape and manipulate
the correlation function and power spectra.

Functions:
 - unflatten_xipm(array)
 - flatten_xipm(corr, mask, settings)
 - mask_xipm(array, mask)
 - unmask_xipm(array, mask)

"""

import numpy as np
import settings as set



# ------------------- Flatten and unflatten correlation function --------------#

    #Given the position in the array, find the corresponding position in the unflattened array
def position_xipm(n, n_bins, n_theta):
    """ Given the position in the array, find the
        corresponding position in the unflattened array.

    Args:
        n: position in the flattened array.
        n_bins: number of bins.
        n_theta: number of theta_ell variables.

    Returns:
        p_pm, p_theta, p_bin_1, p_bin_2.

    """

    # Check that the input is consistent with these numbers
    p_max = 2*n_theta*n_bins*(n_bins+1)/2
    if n>=p_max:
        raise ValueError("The input number is larger than expected!")
    # div: gives position of bins. mod: gives pm and theta
    div, mod = np.divmod(n, 2*n_theta)
    # Calculate position of pm and theta
    if mod<n_theta:
        p_pm = 0
        p_theta = mod
    else:
        p_pm = 1
        p_theta = mod-n_theta
    # Calculate position of bin1 and bin2
    intervals = np.flip(np.array([np.arange(x,n_bins+1).sum() for x in np.arange(2,n_bins+2)]),0)
    p_bin_1 = np.where(intervals<=div)[0][-1]
    p_bin_2 = div - intervals[p_bin_1] + p_bin_1

    return p_pm, p_theta, p_bin_1, p_bin_2


def unflatten_xipm(array):
    """ Unflatten the correlation function.

    Args:
        array: correlation function.

    Returns:
        reshaped correlation function (pm, theta, bin1, bin2).

    """

    # Local variables
    n_bins = len(set.Z_BINS)-1
    n_theta = len(set.THETA_ARCMIN)

    # Initialize array with xipm
    xipm = np.zeros((2, n_theta, n_bins, n_bins))
    # Main loop: scroll all elements of the flattened array and reshape them
    for count in range(len(array)):
        # From position in flattened array, give position for each index
        p_pm, p_theta, p_bin_1, p_bin_2 = position_xipm(count, n_bins, n_theta)
        # Assign element to xipm (bin1 and bin2 are symmetric)
        xipm[p_pm,p_theta,p_bin_1,p_bin_2] = array[count]
        xipm[p_pm,p_theta,p_bin_2,p_bin_1] = array[count]

    return xipm


def flatten_xipm(corr, settings):
    """ Flatten the correlation function.

    Args:
        corr: correlation function.
        settings: dictionary with settings.

    Returns:
        flattened correlation function.

    """

    # Local variables
    n_theta_ell = settings['n_theta_ell']
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        n_bins = settings['n_kl']
    else:
        n_bins = settings['n_bins']

    # Flatten array
    if settings['method'] in ['full', 'kl_off_diag']:
        n_data = 2*n_theta_ell*n_bins*(n_bins+1)/2
        data_f = np.empty(n_data)
        for n in range(n_data):
            p_pm, p_tl, p_b1, p_b2 = position_xipm(n, n_bins, n_theta_ell)
            data_f[n] = corr[p_pm, p_tl, p_b1, p_b2]
    else:
        n_data = 2*n_theta_ell*n_bins
        data_f = np.empty(n_data)
        for n in range(n_data):
            div, mod = np.divmod(n, 2*n_theta_ell)
            if mod<n_theta_ell:
                p_pm = 0
                p_theta = mod
            else:
                p_pm = 1
                p_theta = mod-n_theta_ell
            p_bin = div
            data_f[n] = corr[p_pm, p_theta, p_bin]

    return data_f


# ------------------- Mask and Unmask correlation function --------------------#

def mask_xipm(array, mask, settings):
    """ Convert a unmasked array into a masked one.

    Args:
        array: array with the unmasked xipm.
        mask: mask that has been used.
        settings: dictionary with settings.

    Returns:
        array with masked xipm.

    """

    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        n_bins = settings['n_kl']
    else:
        n_bins = settings['n_bins']

    if settings['method'] in ['full', 'kl_off_diag']:
        mask_tot = np.tile(mask.flatten(),n_bins*(n_bins+1)/2)
    else:
        mask_tot = np.tile(mask.flatten(),n_bins)

    return array[mask_tot]


def unmask_xipm(array, mask):
    """ Convert a flatten masked array into
        an unmasked one (still flatten).

    Args:
        array: array with the masked xipm.
        mask: mask that has been used.

    Returns:
        array with unmasked xipm.

    """

    # Flatten mask and tile mask
    mask_f = mask.flatten()
    # Get number of times that theta_pm should be repeated
    div, mod = np.divmod(len(array), len(mask_f[mask_f]))
    if mod == 0:
        raise IOError('The length of the input array is not correct!')
    mask_f = np.tile(mask_f, div)

    # Find positions where to write values
    pos = np.where(mask_f)[0]

    # Define unmasked array
    xipm = np.zeros(len(mask_f))

    # Assign components
    for n1, n2 in enumerate(pos):
        xipm[n2] = array[n1]

    return xipm
