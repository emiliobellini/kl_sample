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

    #Given the position in the array, find the corresponding position in the unflattened array
    def position_xipm(n, n_bins, n_theta):
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


def flatten_xipm(corr, mask, settings):
    """ Reshape the correlation function.

    Args:
        corr: correlation function.
        mask: mask for the angle theta.
        settings: dictionary with settings.

    Returns:
        reshaped correlation function.

    """

    # Local variables
    n_x_var = settings['n_x_var']
    n_bins = settings['n_bins']

    # Join together pm and theta indices
    data_r = corr.reshape((2*n_x_var, n_bins, n_bins))
    # Mask unused theta
    data_r = data_r[mask.flatten()]
    # Remove symmetric elements in bin1 and bin2
    data_r = np.triu(data_r)

    # If KL
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        n_kl = settings['n_kl']
        # keep only the first n_kl bins
        data_r = data_r[:,:n_kl,:n_kl]
        if settings['method'] == 'kl_diag':
            # If diagonal, diagonalize in bin1 and bin2
            data_r = np.diagonal(data_r,  axis1=1, axis2=2)

    # Transpose array (bin1, bin2, pm_theta)
    data_r = np.transpose(data_r, axes=[1,2,0])
    # Flatten
    data_r = data_r.flatten()
    # Remove 0 elements (related to np.triu above)
    data_r = data_r[data_r != 0]

    return data_r


# ------------------- Mask and Unmask correlation function --------------------#

def mask_xipm(array, mask):
    """ Convert a flatten unmasked array
        into a masked one (still flatten).

    Args:
        array: array with the unmasked xipm.
        mask: mask that has been used.

    Returns:
        array with masked xipm.

    """
    return


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
    if mod != 0:
        raise IOError('The number of files in ' + fname + ' is not correct!')
    mask_f = np.tile(mask_f, div)

    # Find positions where to write values
    pos = np.where(mask_f)[0]

    # Define unmasked array
    xipm = np.zeros(len(mask_f))

    # Assign components
    for n1, n2 in enumerate(pos):
        xipm[n2] = array[n1]

    return xipm
