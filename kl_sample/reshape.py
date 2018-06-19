
import numpy as np
import settings as set
#Given the position in the file, find the corresponding position in the array
def position_xipm(n):
    n_bins = len(set.Z_BINS)-1
    n_theta = len(set.THETA_ARCMIN)
    n_theta_xip = np.array(set.MASK_THETA[0]).astype(int).sum()
    n_theta_xim = np.array(set.MASK_THETA[1]).astype(int).sum()
    p_max = (n_theta_xip+n_theta_xim)*n_bins*(n_bins+1)/2
    if n>=p_max:
        raise ValueError("The input number is larger than expected!")
    div, mod = np.divmod(n, n_theta_xip+n_theta_xim)
    if mod<n_theta_xip:
        p_pm = 0
        p_theta = mod
    else:
        p_pm = 1
        p_theta = 3+mod-n_theta_xip
    intervals = np.flip(np.array([np.arange(x,n_bins+1).sum() for x in np.arange(2,n_bins+2)]),0)
    p_bin_1 = np.where(intervals<=div)[0][-1]
    p_bin_2 = div - intervals[p_bin_1] + p_bin_1
    return p_pm, p_theta, p_bin_1, p_bin_2

def unflatten_xipm(array):
    n_bins = len(set.Z_BINS)-1
    n_theta = len(set.THETA_ARCMIN)
    xipm = np.zeros((2, n_theta, n_bins, n_bins))
    for count in range(len(array)):
        p_pm, p_theta, p_bin_1, p_bin_2 = position_xipm(count)
        xipm[p_pm,p_theta,p_bin_1,p_bin_2] = array[count]
        xipm[p_pm,p_theta,p_bin_2,p_bin_1] = array[count]
    return xipm


def reshape_corr(corr, settings, mask):
    """ Reshape the correlation function.

    Args:
        corr: correlation function
        settings: dictionary with settings

    Returns:
        reshaped correlation function.

    """

    data_r = corr.reshape(
        (2*settings['n_x_var'],settings['n_bins'],settings['n_bins'])
        )
    data_r = np.triu(data_r[mask.flatten()])
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        data_r = data_r[:,:settings['n_kl'],:settings['n_kl']]
        if settings['method'] == 'kl_diag':
            data_r = np.diagonal(data_r,  axis1=1, axis2=2)
    data_r = np.transpose(data_r, axes=[1,2,0])
    data_r = data_r.flatten()
    data_r = data_r[data_r != 0]

    return data_r
