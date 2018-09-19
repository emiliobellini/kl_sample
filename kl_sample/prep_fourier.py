"""

This module contains the pipeline to prepare data in
fourier space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import os
import sys
import re
import numpy as np
import time
from astropy.io import fits
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

    # Global variable collecting warnings
    warning = False
    # List of CFHTlens fields
    fields = set.FIELDS_CFHTLENS
    # List of redshift bins
    z_bins = np.array([[set.Z_BINS[n], set.Z_BINS[n+1]] for n in np.arange(len(set.Z_BINS)-1)])



# ------------------- Useful functions ----------------------------------------#

    def timing(fun, name):
        warning = False
        start = time.clock()
        warning = fun
        end = time.clock()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print 'Run {} module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(name,int(hours),int(minutes),seconds)
        return warning



# ------------------- Initialize paths ----------------------------------------#

    # Define absolute paths
    path = {}
    path['base'], path['fname'] = os.path.split(os.path.abspath(args.input_path))
    io.path_exists_or_error(path['base'])
    path['cat_full'] = path['base']+'/'+path['fname']+'full_cat.fits'
    path['mask_url'] = path['base']+'/mask_url.txt'
    path['photo_z'] = path['base']+'/photo_z.fits'
    for f in fields:
        path['cat_'+f] = path['base']+'/'+path['fname']+'cat_'+f+'.fits'
        path['map_'+f] = path['base']+'/'+path['fname']+'map_'+f+'.fits'
        path['mask_'+f] = path['base']+'/mask_'+f+'.fits'
        path['m_'+f] = path['base']+'/mult_corr_'+f+'.fits'

    # Determine which modules have to be run, by checking the existence of the
    # output files and arguments passed by the user
    is_run_mask = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
    if args.run_mask:
        is_run_mask = True
    is_run_cat = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
    if args.run_cat:
        is_run_cat = True
    is_run_mult = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
    if args.run_mult:
        is_run_mult = True
    is_run_pz = not(os.path.exists(path['photo_z']))
    if args.run_pz:
        is_run_pz = True
    is_run_map = np.array([not(os.path.exists(path['map_'+f])) for f in fields]).any()
    if args.run_map:
        is_run_map = True

    # Check the existence of the required input files
    if is_run_mask:
        nofile1 = not(os.path.exists(path['mask_url']))
        if nofile1:
            print 'WARNING: I will skip the MASK module. Input files not found!'
            sys.stdout.flush()
            is_run_mask = False
            warning = True
    else:
        print 'I will skip the MASK module. Output files already there!'
        sys.stdout.flush()
    if is_run_cat:
        nofile1 = not(os.path.exists(path['cat_full']))
        if nofile1:
            print 'WARNING: I will skip the CATALOGUE module. Input file not found!'
            sys.stdout.flush()
            is_run_cat = False
            warning = True
    else:
        print 'I will skip the CATALOGUE module. Output files already there!'
        sys.stdout.flush()
    if is_run_mult:
        nofile1 = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
        if not(is_run_cat) and nofile1:
            print 'WARNING: I will skip the MULT_CORR module. Input file not found!'
            sys.stdout.flush()
            is_run_mult = False
            warning = True
    else:
        print 'I will skip the MULT_CORR module. Output files already there!'
        sys.stdout.flush()
    if is_run_pz:
        nofile1 = not(os.path.exists(path['cat_full']))
        nofile2 = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
        if nofile1 or (not(is_run_mult) and nofile2):
            print 'WARNING: I will skip the PHOTO_Z module. Input files not found!'
            sys.stdout.flush()
            is_run_pz = False
            warning = True
    else:
        print 'I will skip the PHOTO_Z module. Output file already there!'
        sys.stdout.flush()
    if is_run_map:
        nofile1 = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
        nofile2 = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
        if (not(is_run_cat) and nofile1) or (not(is_run_mult) and nofile2):
            print 'WARNING: I will skip the MAP module. Input files not found!'
            sys.stdout.flush()
            is_run_map = False
            warning = True
    else:
        print 'I will skip the MAP module. Output files already there!'
        sys.stdout.flush()



# ------------------- Function to calculate the mask --------------------------#

    def run_mask(path=path, fields=fields, z_bins=z_bins):

        print 'Running MASK module'
        sys.stdout.flush()
        warning = False


        # Main loop: scan over the fields and generate new maps
        for f in fields:
            print 'Calculating mask for field ' + f + ':'
            sys.stdout.flush()

            # Remove old output file to avoid confusion
            try:
                os.remove(path['mask_'+f])
            except:
                pass

            # Get urls for the sub-masks
            urls = []
            with open(path['mask_url'], 'r') as fn:
                for line in fn:
                    if re.match('.+'+f+'.+finalmask_mosaic.fits', line):
                        urls.append(line.rstrip())



    #     # Function that read necessary data and check their internal structure
    #     def read_mask(fname, imname='PRIMARY',
    #             keys=['CRPIX1','CRPIX2','CD1_1','CD2_2','CRVAL1','CRVAL2','CTYPE1','CTYPE2']):
    #
    #         # Read the image
    #         try:
    #             mask = io.read_from_fits(fname, imname).astype(int)
    #         except KeyError:
    #             print 'WARNING: No image '+imname+' in '+fname+'. Skipping calculation!'
    #             return None, None, True
    #         # Read the header
    #         try:
    #             hd = io.read_header_from_fits(fname, imname)
    #             hd = dict([(x,hd[x]) for x in keys])
    #         except KeyError:
    #             print 'WARNING: Header ill defined in '+fname+', missing parameters. Skipping calculation!'
    #             return None, None, True
    #
    #         return mask, hd, warning
    #
    #
    #     # Function to degrade boolean masks
    #     def degrade_mask(mask, hd, size_pix=size_pix):
    #
    #         # Calculate new dimensions
    #         ratio_pix1 = abs(size_pix/hd['CD1_1']/60.**2)
    #         ratio_pix2 = abs(size_pix/hd['CD2_2']/60.**2)
    #
    #         # print mask.shape
    #         # print size_pix
    #         # print hd
    #
    #
    #
    # #     div1, mod1 = np.divmod(mask_orig.shape[0], dim_ratio)
    # #     div2, mod2 = np.divmod(mask_orig.shape[1], dim_ratio)
    # #     if mod1 == 0:
    # #         x1 = div1
    # #     else:
    # #         x1 = div1+1
    # #     if mod2 == 0:
    # #         x2 = div2
    # #     else:
    # #         x2 = div2+1
    # #     start1 = int(np.round((x1*dim_ratio-mask_orig.shape[0])/2.))
    # #     start2 = int(np.round((x2*dim_ratio-mask_orig.shape[1])/2.))
    # #     end1 = start1+mask_orig.shape[0]
    # #     end2 = start2+mask_orig.shape[1]
    # #
    # #     # Add borders to the mask
    # #     mask_orig_ext = np.zeros((x1*dim_ratio, x2*dim_ratio), dtype=bool)
    # #     mask_orig_ext[start1:end1,start2:end2] = mask_orig
    # #     print '----> Extended shape: ', mask_orig_ext.shape
    # #     sys.stdout.flush()
    # #
    # #     # Calculate new mask
    # #     mask = np.zeros((x1, x2))
    # #     for count1 in range(x1):
    # #         for count2 in range(x2):
    # #             s1 = count1*dim_ratio
    # #             s2 = count2*dim_ratio
    # #             new_pix = mask_orig_ext[s1:s1+dim_ratio,s2:s2+dim_ratio].astype(float)
    # #             mask[count1,count2] = np.average(new_pix)
    #
    # #     add1 = float(x1)*dim_ratio/mask_orig.shape[0]-1.
    # #     add2 = float(x2)*dim_ratio/mask_orig.shape[1]-1.
    # #     print '----> Pixels added: '+'({0:5.2%}, '.format(add1)+'{0:5.2%})'.format(add2)
    # #     sys.stdout.flush()
    # #
    # #     # Create a WCS object and save it to file
    # #     hd = io.read_header_from_fits(path['mask_'+f], 'primary')
    # #     # Create a new WCS object
    # #     w = wcs.WCS(naxis=2)
    # #     # Write header
    # #     w.wcs.crpix = np.array([start1+hd['CRPIX1'], start2+hd['CRPIX2']])/dim_ratio
    # #     w.wcs.cdelt = np.array([hd['CD1_1'], hd['CD2_2']])*dim_ratio
    # #     w.wcs.crval = np.array([hd['CRVAL1'], hd['CRVAL2']])
    # #     w.wcs.ctype = [hd['CTYPE1'], hd['CTYPE2']]
    # #     hd = w.to_header()
    #
    #         return mask, hd
    #
    #
    #     # Main loop: scan over the fields and generate new maps
    #     for f in fields:
    #         print 'Calculating mask for field ' + f + ':'
    #         sys.stdout.flush()
    #
    #         # Remove old output file to avoid confusion
    #         try:
    #             os.remove(path['mask_'+f])
    #         except:
    #             pass
    #
    #         # Read masks
    #         mask_sec, hd_sec, warning = read_mask(path['mask_sec_'+f])
    #         mask_gb, hd_gb, warning = read_mask(path['good_bad_'+f])
    #         if abs(hd_sec['CRVAL1']/hd_gb['CRVAL1']-1.)>1.e-5 or abs(hd_sec['CRVAL2']/hd_gb['CRVAL2']-1.)>1.e-5:
    #             print 'WARNING: Central position of the two masks is different. Skipping calculation!'
    #             warning = True
    #         if warning:
    #             continue
    #
    #         # Convert masks to boolean
    #         mask_sec = (1-np.array(mask_sec, dtype=bool)).astype(bool)
    #         print '----> Shape arcsec mask: ', mask_sec.shape
    #         sys.stdout.flush()
    #         mask_gb = np.array(mask_gb, dtype=bool)
    #         print '----> Shape good_bad mask: ', mask_gb.shape
    #         sys.stdout.flush()
    #
    #         # Degrade masks
    #         mask_min, hd_min = degrade_mask(mask_sec, hd_sec)
    #         print '----> Shape arcmin mask: ', mask_min.shape
    #         sys.stdout.flush()
    #         mask_gb, hd_gb = degrade_mask(mask_gb, hd_gb)
    #         print '----> New shape good_bad mask: ', mask_min.shape
    #         sys.stdout.flush()





        return warning



    #     # Write to file
    #     fname = path['input']+'/'+f+'_mask.fits'
    #     io.write_to_fits(fname, mask, f+'_mask', header=hd)



# ------------------- Function to calculate the clean catalogue ---------------#

    def run_cat(path=path, fields=fields, z_bins=z_bins):

        print 'Running CATALOGUE module'
        sys.stdout.flush()
        warning = False


        # Read galaxy catalogue
        table_name = 'data'
        fname = path['cat_full']
        try:
            cat = io.read_from_fits(fname, table_name)
        except KeyError:
            print 'WARNING: No key '+table_name+' in '+fname+'. Skipping calculation!'
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'c2', 'weight', 'id', 'Z_B', 'MASK', 'star_flag']
        for key in table_keys:
            if key not in cat.columns.names:
                print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                return True


        # Main loop: scan over the fields and generate new maps
        for f in fields:

            # Remove old output file to avoid confusion
            try:
                os.remove(path['cat_'+f])
            except:
                pass

            # Second loop to divide galaxies in redshift bins
            for n_z_bin, z_bin in enumerate(z_bins):

                print 'Calculating catalogue for field ' + f + ' and bin {}:'.format(n_z_bin+1)
                sys.stdout.flush()

                # Filter galaxies
                filter = set.filter_galaxies(cat, z_bin[0], z_bin[1], field=f)
                gals = cat[filter]

                # Create Table and save it
                table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'c1', 'c2', 'weight']
                columns = []
                for key in table_keys:
                    if key=='c1':
                        columns.append(fits.Column(name=key,array=np.zeros(len(gals)),format='E'))
                    else:
                        columns.append(fits.Column(name=key,array=gals[key],format='E'))
                name = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
                gals = fits.BinTableHDU.from_columns(columns, name=name)
                warning = io.write_to_fits(path['cat_'+f], gals, name, type='table')

            io.print_info_fits(path['cat_'+f])

        return warning



# ------------------- Function to calculate the multiplicative correction -----#

    def run_mult(path=path, fields=fields, z_bins=z_bins):
        warning = False
        return warning



# ------------------- Function to calculate the multiplicative correction -----#

    def run_pz(path=path, z_bins=z_bins):
        warning = False
        return warning



# ------------------- Function to calculate the map ---------------------------#

    def run_map(path=path, fields=fields, z_bins=z_bins):
        warning = False
        return warning



# ------------------- Pipeline ------------------------------------------------#

    if is_run_mask:
        warning = timing(run_mask(), 'MASK') or warning
    if is_run_cat:
        warning = timing(run_cat(), 'CATALOGUE') or warning
    if is_run_mult:
        warning = timing(run_mult(), 'MULT_CORR') or warning
    if is_run_pz:
        warning = timing(run_pz(), 'PHOTO_Z') or warning
    if is_run_map:
        warning = timing(run_map(), 'MAP') or warning

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
    # # Read galaxy catalogue
    # gals = io.read_from_fits(path['input']+'/data.fits', 'data')
    #
    #
    #
    #
    # # Main loop to scan over the fields
    # for f in fields:
    #
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
    #         gals_f = gals[set.filter_galaxies(gals, z_bin[0], z_bin[1], field=f)]
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
    #                 print '----> Done {0:5.1%} of the pixels ({1:d})'.format(float(count+1) /len(pix_gals), len(pix_gals))
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
