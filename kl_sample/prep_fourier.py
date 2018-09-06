"""

This module contains the pipeline to prepare data in
fourier space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import os
import sys
import numpy as np
# from astropy.io import fits
# from astropy import wcs
import settings as set
import io


def prep_fourier(args):
    """ Prepare data in fourier space.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to fits files file the output.

    """



# ------------------- Useful parameters ---------------------------------------#

    warning = False
    fields = ['W'+str(x+1) for x in range(4)]
    z_bins = np.array([[set.Z_BINS[n], set.Z_BINS[n+1]] for n in np.arange(len(set.Z_BINS)-1)])



# ------------------- Initialize paths ----------------------------------------#

    # Define absolute paths
    path = {}
    path['base'], path['fname'] = os.path.split(os.path.abspath(args.input_path))
    io.path_exists_or_error(path['base'])
    path['cat_full'] = path['base']+'/'+path['fname']+'full_cat.fits'
    for f in fields:
        path['cat_'+f] = path['base']+'/'+path['fname']+'cat_'+f+'.fits'
        path['m_'+f] = path['base']+'/'+path['fname']+'mult_corr_'+f+'.fits'
        path['map_'+f] = path['base']+'/'+path['fname']+'map_'+f+'.fits'
        path['mask_sec_'+f] = path['base']+'/mask_arcsec_'+f+'.fits.gz'
        path['mask_min_'+f] = path['base']+'/mask_'+f+'.fits'

    # Determine which modules have to be run, by checking the existence of the
    # output files and arguments passed by the user
    is_run_mask = np.array([not(os.path.exists(path['mask_min_'+f])) for f in fields]).any()
    if args.run_mask:
        is_run_mask = True
    is_run_cat = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
    if args.run_cat:
        is_run_cat = True
    is_run_mult = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
    if args.run_mult:
        is_run_mult = True
    is_run_map = np.array([not(os.path.exists(path['map_'+f])) for f in fields]).any()
    if args.run_map:
        is_run_map = True

    # Check the existence of the required input files
    if is_run_mask:
        if np.array([not(os.path.exists(path['mask_sec_'+f])) for f in fields]).any():
            print 'WARNING: Skipping module to calculate the masks. Input files not found!'
            sys.stdout.flush()
            is_run_mask = False
            warning = True
    else:
        print 'Skipping module to calculate the masks. Output files already there!'
        sys.stdout.flush()
    if is_run_cat:
        if not(os.path.exists(path['cat_full'])):
            print 'WARNING: Skipping module to calculate the clean catalogues. Input file not found!'
            sys.stdout.flush()
            is_run_cat = False
            warning = True
    else:
        print 'Skipping module to calculate the clean catalogues. Output files already there!'
        sys.stdout.flush()
    if is_run_mult:
        if not(is_run_cat) and np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any():
            print 'WARNING: Skipping module to calculate the multiplicative correction. Input files not found!'
            sys.stdout.flush()
            is_run_mult = False
            warning = True
    else:
        print 'Skipping module to calculate the multiplicative correction. Output files already there!'
        sys.stdout.flush()
    if is_run_map:
        if not(is_run_cat) and np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any():
            print 'WARNING: Skipping module to calculate the map. Input files not found!'
            sys.stdout.flush()
            is_run_map = False
            warning = True
    else:
        print 'Skipping module to calculate the map. Output files already there!'
        sys.stdout.flush()



# ------------------- Function to calculate the mask --------------------------#

    def run_mask(path=path, fields=fields, z_bins=z_bins):
        return



# ------------------- Function to calculate the clean catalogue ---------------#

    def run_cat(path=path, fields=fields, z_bins=z_bins):
        return



# ------------------- Function to calculate the multiplicative correction -----#

    def run_mult(path=path, fields=fields, z_bins=z_bins):
        return



# ------------------- Function to calculate the map ---------------------------#

    def run_map(path=path, fields=fields, z_bins=z_bins):
        return



# ------------------- Pipeline ------------------------------------------------#

    if is_run_mask:
        run_mask()
    if is_run_cat:
        run_cat()
    if is_run_mult:
        run_mult()
    if is_run_map:
        run_map()

    if warning:
        print 'Done! However something went unexpectedly!! Check your warnings!'
    else:
        print 'Success!!'




    #     io.path_exists_or_error(path['mask_sec_'+f])
    # path['base'] = io.path_exists_or_error(args.input_path)


    # # Local variables
    # dim_orig = 1 # Original pixel dimension of the mask in arcsec
    # dim_new = 120 # New pixel dimension in arcsec
    # n_avg_m = 2 # Radius of new pixels used to average the multiplicative correction to the shear (n_pixels = (2*n_avg_m+1)**2)
    #
    #
    # # Infer dereived quantities
    # fields = ['W'+str(x+1) for x in range(4)]
    # z_bins = np.array([[set.Z_BINS[n], set.Z_BINS[n+1]] for n in np.arange(len(set.Z_BINS)-1)])
    # dim_ratio = int(np.round(float(dim_new)/float(dim_orig)))
    #
    #
    # # Define absolute paths and check the existence of each required file
    # path = {}
    # path['input'] = io.path_exists_or_error(args.input_folder)
    # for f in fields:
    #     path['mask_'+f] = io.path_exists_or_error(path['input']+'/'+f+'.16bit.small.reg2.fits.gz')
    # path['output'] = io.path_exists_or_error(args.input_folder)
    #
    # # Read galaxy catalogue
    # gals = io.read_from_fits(path['input']+'/data.fits', 'data')
    #
    #
    # # Plotting function
    # # def plot_and_save(table, fname):
    # #     plt.imshow(table,interpolation='nearest')
    # #     plt.colorbar()
    # #     plt.savefig(fname)
    # #     plt.close()
    # #     return
    #
    #
    # # Function that filters the galaxies
    # def filter_gals(gals, field, z_bin):
    #     sel = set.get_mask(gals, z_bin[0], z_bin[1])
    #     sel = np.array([x[:2] in field for x in gals['id']])*sel
    #     return sel
    #
    #
    # # Main loop to scan over the fields
    # for f in fields:
    #
    #     # Calculating the mask
    #     print 'Calculating mask for field ' + f + ':'
    #     sys.stdout.flush()
    #
    #
    #     # Read original mask
    #     mask_orig = io.read_from_fits(path['mask_'+f], 'primary').astype(int)
    #     print '----> Original shape: ', mask_orig.shape
    #     sys.stdout.flush()
    #
    #     # Convert the mask to boolean
    #     mask_orig = (1-np.array(mask_orig, dtype=bool)).astype(bool)
    #
    #     # Calculate new dimensions
    #     div1, mod1 = np.divmod(mask_orig.shape[0], dim_ratio)
    #     div2, mod2 = np.divmod(mask_orig.shape[1], dim_ratio)
    #     if mod1 == 0:
    #         x1 = div1
    #     else:
    #         x1 = div1+1
    #     if mod2 == 0:
    #         x2 = div2
    #     else:
    #         x2 = div2+1
    #     start1 = int(np.round((x1*dim_ratio-mask_orig.shape[0])/2.))
    #     start2 = int(np.round((x2*dim_ratio-mask_orig.shape[1])/2.))
    #     end1 = start1+mask_orig.shape[0]
    #     end2 = start2+mask_orig.shape[1]
    #
    #     # Add borders to the mask
    #     mask_orig_ext = np.zeros((x1*dim_ratio, x2*dim_ratio), dtype=bool)
    #     mask_orig_ext[start1:end1,start2:end2] = mask_orig
    #     # plot_and_save(mask_orig_ext, path['output']+'/'+f+'_orig.pdf')
    #     print '----> Extended shape: ', mask_orig_ext.shape
    #     sys.stdout.flush()
    #
    #     # Calculate new mask
    #     mask = np.zeros((x1, x2))
    #     for count1 in range(x1):
    #         for count2 in range(x2):
    #             s1 = count1*dim_ratio
    #             s2 = count2*dim_ratio
    #             new_pix = mask_orig_ext[s1:s1+dim_ratio,s2:s2+dim_ratio].astype(float)
    #             mask[count1,count2] = np.average(new_pix)
    #     # plot_and_save(mask, path['output']+'/'+f+'_new.pdf')
    #     print '----> New shape: ', (x1, x2)
    #     sys.stdout.flush()
    #     add1 = float(x1)*dim_ratio/mask_orig.shape[0]-1.
    #     add2 = float(x2)*dim_ratio/mask_orig.shape[1]-1.
    #     print '----> Pixels added: '+'({0:5.2%}, '.format(add1)+'{0:5.2%})'.format(add2)
    #     sys.stdout.flush()
    #
    #     # Create a WCS object and save it to file
    #     hd = io.read_header_from_fits(path['mask_'+f], 'primary')
    #     # Create a new WCS object
    #     w = wcs.WCS(naxis=2)
    #     # Write header
    #     w.wcs.crpix = np.array([start1+hd['CRPIX1'], start2+hd['CRPIX2']])/dim_ratio
    #     w.wcs.cdelt = np.array([hd['CD1_1'], hd['CD2_2']])*dim_ratio
    #     w.wcs.crval = np.array([hd['CRVAL1'], hd['CRVAL2']])
    #     w.wcs.ctype = [hd['CTYPE1'], hd['CTYPE2']]
    #     hd = w.to_header()
    #     # Write to file
    #     fname = path['input']+'/'+f+'_mask.fits'
    #     io.write_to_fits(fname, mask, f+'_mask', header=hd)
    #
    #
    #
    #     # Second loop to divide galaxies in redshift bins
    #     for n_z_bin, z_bin in enumerate(z_bins):
    #
    #         # Calculating the map
    #         print 'Calculating map for field ' + f + ' and bin {}:'.format(n_z_bin+1)
    #         sys.stdout.flush()
    #
    #         # Create arrays for the two shears
    #         map_1 = np.zeros(mask.shape)
    #         map_2 = np.zeros(mask.shape)
    #
    #         # Filter galaxies
    #         gals_f = gals[filter_gals(gals, f, z_bin)]
    #
    #         # World position
    #         pos_w = np.array([[gals_f['ALPHA_J2000'][x],gals_f['DELTA_J2000'][x]] for x in range(len(gals_f))])
    #         # Pixel position
    #         pos_pix = w.wcs_world2pix(pos_w, 1).astype(int)
    #         pos_pix = np.flip(pos_pix,axis=1) #Need to invert the columns
    #         # Pixels where at least one galaxy has been found
    #         pix_gals = np.unique(pos_pix, axis=0)
    #
    #         print '----> Empty pixels: {0:5.2%}'.format(1.-np.array([mask[tuple(x)] for x in pix_gals]).sum()/mask.flatten().sum())
    #         sys.stdout.flush()
    #
    #         # Scan over the populated pixels and calculate the shear
    #         for count, pix in enumerate(pix_gals):
    #             # Calculate averaged multiplicative correction
    #             if pix[0]-n_avg_m<0:
    #                 s1 = 0
    #             elif pix[0]+n_avg_m>=mask.shape[0]:
    #                 s1 = mask.shape[0]-(2*n_avg_m+1)
    #             else:
    #                 s1 = pix[0]-n_avg_m
    #             if pix[1]-n_avg_m<0:
    #                 s2 = 0
    #             elif pix[1]+n_avg_m>=mask.shape[1]:
    #                 s2 = mask.shape[1]-(2*n_avg_m+1)
    #             else:
    #                 s2 = pix[1]-n_avg_m
    #             sel = pos_pix[:,0] >= s1
    #             sel = (pos_pix[:,0] < s1+2*n_avg_m+1)*sel
    #             sel = (pos_pix[:,1] >= s2)*sel
    #             sel = (pos_pix[:,1] < s2+2*n_avg_m+1)*sel
    #             m = gals_f[sel]['m']
    #             weight = gals_f[sel]['weight']
    #             m = np.average(m, weights=weight)
    #             # Select galaxies in a pixel
    #             sel = (pos_pix[:,0] == pix[0])*(pos_pix[:,1] == pix[1])
    #             # Define quantities
    #             e1 = gals_f[sel]['e1']
    #             e2 = gals_f[sel]['e2']
    #             c2 = gals_f[sel]['c2']
    #             weight = gals_f[sel]['weight']
    #             # Calculate shear
    #             map_1[tuple(pix)] = np.average(e1, weights=weight)/(1+m)
    #             map_2[tuple(pix)] = np.average(e2-c2, weights=weight)/(1+m)
    #
    #             # Print every some step
    #             if (count+1) % 1000 == 0:
    #                 print '----> Done {0:5.1%} of the pixels ({})'.format(float(count+1) /len(pix_gals), len(pix_gals))
    #                 sys.stdout.flush()
    #
    #         print '----> Shear calculation finished!'
    #         sys.stdout.flush()
    #
    #         # Write to file
    #         fname = path['input']+'/'+f+'_map.fits'
    #         name_1 = f+'_g1_z'+str(n_z_bin+1)+'_map'
    #         name_2 = f+'_g2_z'+str(n_z_bin+1)+'_map'
    #         io.write_to_fits(fname, map_1, name_1, header=hd)
    #         io.write_to_fits(fname, map_2, name_2, header=hd)
