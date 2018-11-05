"""

This module contains functions to reshape and manipulate
the correlation function and power spectra.

Functions:
 - mask_cl(cl)
 - unify_fields_cl(cl, sims)
 - position_xipm(n, n_bins, n_theta)
 - unflatten_xipm(array)
 - flatten_xipm(corr, settings)
 - mask_xipm(array, mask, settings)
 - unmask_xipm(array, mask)

"""

import numpy as np
import settings as set



# ------------------- Manipulate Cl's -----------------------------------------#

def mask_cl(cl):
    if cl.ndim==4:
        return cl[:,1:-1]
    elif cl.ndim==5:
        return cl[:,:,1:-1]
    else:
        raise ValueError('Expected Cl\'s array with dimensions 4 or 5. Found {}'.format(cl.ndim))


def clean_cl(cl, noise):
    if cl.ndim==4:
        return cl - noise
    elif cl.ndim==5:
        clean = np.array([cl[:,x]-noise for x in range(len(cl[0]))])
        clean = np.transpose(clean,axes=(1,0,2,3,4))
        return clean
    else:
        raise ValueError('Expected Cl\'s array with dimensions 4 or 5. Found {}'.format(cl.ndim))


def flatten_cl(cl):
    tr_idx = np.triu_indices(cl.shape[-1])
    flat_cl = np.moveaxis(cl,[-2,-1],[0,1])
    flat_cl = flat_cl[tr_idx]
    flat_cl = np.moveaxis(flat_cl,[0],[-1])
    flat_cl = flat_cl.reshape(flat_cl.shape[:-2]+(flat_cl.shape[-2]*flat_cl.shape[-1],))
    return flat_cl


def unflatten_cl(cl, shape):
    tr_idx = np.triu_indices(shape[-1])
    unflat_cl = np.zeros(shape)
    tmp_cl = cl.reshape(shape[:-2]+(-1,))
    tmp_cl = np.moveaxis(tmp_cl,[-1],[0])
    unflat_cl = np.moveaxis(unflat_cl,[-2,-1],[0,1])
    unflat_cl[tr_idx] = tmp_cl
    unflat_cl = np.moveaxis(unflat_cl,[1],[0])
    unflat_cl[tr_idx] = tmp_cl
    unflat_cl = np.moveaxis(unflat_cl,[0,1],[-2,-1])
    return unflat_cl


def unify_fields_cl(cl, sims):
    cl_flat = flatten_cl(cl)
    sims_flat = flatten_cl(sims)
    tot_cl = np.zeros(cl_flat.shape[1:])
    tot_inv_cov = np.zeros(cl_flat.shape[1:]*2)
    for nf in range(len(cl)):
        cov = np.cov(sims_flat[nf].T)
        inv_cov = np.linalg.inv(cov)
        tot_inv_cov += inv_cov
        tot_cl += np.dot(inv_cov, cl_flat[nf])
    final_cl = np.dot(np.linalg.inv(tot_inv_cov), tot_cl)
    final_cl = unflatten_cl(final_cl, cl.shape[1:])
    return final_cl


def deband_cl(cl, bp):
    if cl.shape[-3] != bp.shape[0]:
        raise ValueError('Bandpowers and Cl shape mismatch!')
    new_shape = list(cl.shape)
    new_shape[-3] = bp[-1,-1]
    new_shape = tuple(new_shape)
    cl_dbp = np.zeros(new_shape)
    cl_dbp = np.moveaxis(cl_dbp,[-3],[0])
    for count, range in enumerate(bp):
        n_rep = range[1]-range[0]
        cl_flat = cl[count].flatten()
        cl_ext = np.repeat(cl[count],n_rep)
        cl_ext = cl_ext.reshape(cl.shape[1:]+(n_rep,))
        cl_ext = np.moveaxis(cl_ext,[-1],[0])
        cl_dbp[range[0]:range[1]] = cl_ext
    cl_dbp = np.moveaxis(cl_dbp,[0],[-3])
    return cl_dbp

def band_cl(cl, bp):
    if cl.shape[-3] < bp[-1,-1]+1:
        raise ValueError('Bandpowers and Cl shape mismatch!')
    new_shape = list(cl.shape)
    new_shape[-3] = bp.shape[0]
    new_shape = tuple(new_shape)
    cl_bp = np.zeros(new_shape)
    cl_bp = np.moveaxis(cl_bp,[-3],[0])
    for count, range in enumerate(bp):
        cl_re = np.moveaxis(cl,[-3],[0])
        cl_bp[count] = np.average(cl_re[range[0]:range[1]],axis=0)
    cl_bp = np.moveaxis(cl_bp,[0],[-3])
    return cl_bp

# ------------------- Flatten and unflatten correlation function --------------#

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
