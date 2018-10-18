"""

This module contains the tools to prepare
data in fourier space.

Functions:
 - get_map(w, mask, cat)

"""

import sys
import numpy as np


def get_map(w, mask, cat, pos_in=None):
    """ Generate a map from a catalogue, a mask
        and a WCS object.

    Args:
        w: WCS object.
        mask: array with mask.
        cat: catalogue of objects.
        pos_in: to save cpu time it is possible to provide pixel positions.

    Returns:
        map_1, map_2: array with maps for each polarization.
        pos: pixel positions.

    """

    # Create arrays for the two shears
    map_1 = np.zeros(mask.shape)
    map_2 = np.zeros(mask.shape)

    # Get World position of each galaxy
    if pos_in is None:
        pos = zip(cat['ALPHA_J2000'],cat['DELTA_J2000'])
        # Calculate Pixel position of each galaxy
        pos = w.wcs_world2pix(pos, 0).astype(int)
        pos = np.flip(pos,axis=1) #Need to invert the columns
    else:
        pos = pos_in.copy()

    # Perform lex sort and get the sorted indices
    sorted_idx = np.lexsort(pos.T)
    sorted_pos =  pos[sorted_idx,:]
    # Differentiation along rows for sorted array
    diff_pos = np.diff(sorted_pos,axis=0)
    diff_pos = np.append([True],np.any(diff_pos!=0,1),0)
    # Get unique sorted labels
    sorted_labels = diff_pos.cumsum(0)-1
    # Get labels
    labels = np.zeros_like(sorted_idx)
    labels[sorted_idx] = sorted_labels
    # Get unique indices
    unq_idx  = sorted_idx[diff_pos]
    # Get unique pos's and ellipticities
    pos_unique = pos[unq_idx,:]
    w_at_pos = np.bincount(labels, weights=cat['weight'])
    g1_at_pos = np.bincount(labels, weights=cat['e1']*cat['weight'])/w_at_pos
    g2_at_pos = np.bincount(labels, weights=cat['e2']*cat['weight'])/w_at_pos
    # Create the maps
    map_1[pos_unique[:,0],pos_unique[:,1]] = g1_at_pos
    map_2[pos_unique[:,0],pos_unique[:,1]] = g2_at_pos

    empty = 1.-np.array([mask[tuple(x)] for x in pos_unique]).sum()/mask.flatten().sum()
    print '----> Empty pixels: {0:5.2%}'.format(empty)
    sys.stdout.flush()

    return map_1, map_2, pos


def get_cl(bp, mask, map, keep_mcm=False):
    """ Generate cl's from a mask and a map.

    Args:
        bp: bandpowers for ell.
        mask: array with mask.
        map: maps for each bin and polarization.
        keep_mcm: to save cpu time it is possible to save to file the mode coupling matrix.

    Returns:
        cl: array with cl (E/B, bins, ell).
        mcm_path: path to the mode coupling matrix.

    """

    print bp
    print mask
    print map



    cl = np.zeros(5)
    mcm_path = 0.

    return cl, mcm_path
