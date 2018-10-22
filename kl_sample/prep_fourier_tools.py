"""

This module contains the tools to prepare
data in fourier space.

Functions:
 - get_map(w, mask, cat)

"""

import sys
import os
import numpy as np
import pymaster as nmt


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

    return np.array([map_1, map_2]), pos


def get_cl(field, bp, hd, mask, map):
    """ Generate cl's from a mask and a map.

    Args:
        field: field.
        bp: bandpowers for ell.
        hd: header with infos about the mask and maps.
        mask: array with mask.
        map: maps for each bin and polarization.

    Returns:
        cl: array with cl (E/B, bins, ell).
        mcm_path: path to the mode coupling matrix.

    """

    # Initialize Cls
    n_bins = map.shape[1]
    n_ells = len(bp)
    cl = np.zeros((2,2,n_bins,n_bins,n_ells))

    # Dimensions
    Nx = hd['NAXIS1']
    Ny = hd['NAXIS2']
    Lx = Nx*abs(hd['CDELT1'])*np.pi/180 # Mask dimension in radians
    Ly = Ny*abs(hd['CDELT2'])*np.pi/180 # Mask dimension in radians

    # Fields definition
    fd = np.array([nmt.NmtFieldFlat(Lx,Ly,mask[x],[map[0,x],-map[1,x]]) for x in range(n_bins)])
    # Bins for flat sky fields
    b = nmt.NmtBinFlat(bp[:,0],bp[:,1])
    # Effective ells
    ell = b.get_effective_ells()

    # Iterate over redshift bins to compute Cl's
    mcm_paths = []
    for nb1 in range(n_bins):
        for nb2 in range(nb1,n_bins):
            # Temporary path for mode coupling matrix
            mcm_p = os.path.expanduser('~')+'/tmp_mcm_{}_Z{}{}.dat'.format(field,nb1+1,nb2+1)
            mcm_paths.append(mcm_p)
            # Define workspace for mode coupling matrix
            wf = nmt.NmtWorkspaceFlat()
            try:
                wf.read_from(mcm_p)
            except:
                wf.compute_coupling_matrix(fd[nb1],fd[nb2],b)
                wf.write_to(mcm_p)
                print 'Calculated mode coupling matrix for bins {}{}'.format(nb1+1,nb2+1)
                sys.stdout.flush()
            # Calculate Cl's
            cl_c = nmt.compute_coupled_cell_flat(fd[nb1],fd[nb2],b)
            cl_d = wf.decouple_cell(cl_c)
            cl_d = np.reshape(cl_d,(2,2,n_ells))
            cl[:,:,nb1,nb2,:] = cl_d
            cl[:,:,nb2,nb1,:] = cl_d

    return ell, cl, mcm_paths
