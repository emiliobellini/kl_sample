"""

This module contains the pipeline to prepare data in
fourier space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import os
import sys
import re
import numpy as np
import urllib
import time
import gzip
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
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
    # Size pixels masks in arcsecs (it has to be an integer number)
    size_pix = 120
    # Range of pixels used to average the multiplicative correction
    n_avg_m = 5



# ------------------- Initialize paths ----------------------------------------#

    # Define absolute paths
    path = {}
    path['base'], path['fname'] = os.path.split(os.path.abspath(args.input_path))
    io.path_exists_or_error(path['base'])
    path['cat_full'] = path['base']+'/'+path['fname']+'full_cat.fits.gz'
    path['mask_url'] = path['base']+'/mask_url.txt'
    path['photo_z'] = path['base']+'/photo_z.fits'
    for f in fields:
        path['cat_'+f] = path['base']+'/'+path['fname']+'cat_'+f+'.fits'
        path['map_'+f] = path['base']+'/'+path['fname']+'map_'+f+'.fits'
        path['mask_'+f] = path['base']+'/mask_'+f+'.fits'
        path['m_'+f] = path['base']+'/mult_corr_'+f+'.fits'
        path['mask_sec_'+f] = path['base']+'/mask_arcsec_'+f+'.fits.gz'

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
        nofile2 = np.array([not(os.path.exists(path['mask_sec_'+f])) for f in fields]).any()
        if nofile1 or nofile2:
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
        nofile1 = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
        if (not(is_run_mask) and nofile1) or (not(is_run_cat) and nofile2):
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
        nofile1 = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
        nofile3 = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
        if (not(is_run_mask) and nofile1) or (not(is_run_cat) and nofile2) or (not(is_run_mult) and nofile3):
            print 'WARNING: I will skip the MAP module. Input files not found!'
            sys.stdout.flush()
            is_run_map = False
            warning = True
    else:
        print 'I will skip the MAP module. Output files already there!'
        sys.stdout.flush()



# ------------------- Function to calculate the mask --------------------------#

    def run_mask(path=path, fields=fields, z_bins=z_bins, size_pix=size_pix):

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


            # Read urls where to find coordinates of the bad fields
            urls = []
            with open(path['mask_url'], 'r') as fn:
                for line in fn:
                    cond1 = re.match('.+'+f+'.+finalmask_mosaic.fits', line)
                    cond2 = not(np.array([re.match('.+'+x+'.+', line) for x in set.good_fit_patterns])).any()
                    if cond1 and cond2:
                        urls.append(line.rstrip())


            # Read mask data
            imname = 'primary'
            fname = path['mask_sec_'+f]
            try:
                mask_sec = io.read_from_fits(fname, imname).astype(np.uint16)
            except KeyError:
                print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True
            # Read mask header and check necessary keys
            try:
                hd_sec = io.read_header_from_fits(fname, imname)
            except KeyError:
                print 'WARNING: No header in '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True
            for key in ['CRPIX1','CRPIX2','CD1_1','CD2_2','CRVAL1','CRVAL2','CTYPE1','CTYPE2']:
                if key not in list(hd_sec.keys()):
                    print 'WARNING: No key '+key+' in '+fname+'. Skipping calculation!'
                    sys.stdout.flush()
                    return True

            # Convert mask to boolean
            mask_sec = (1-np.array(mask_sec, dtype=bool)).astype(np.int8)


            # Determine how many pixels should be grouped together in the degraded mask
            dim_ratio =int(np.round(abs(size_pix/(hd_sec['CD1_1']*60.**2))))
            cond1 = abs(dim_ratio/abs(size_pix/(hd_sec['CD1_1']*60.**2))-1)>1e-6
            cond2 = abs(dim_ratio/abs(size_pix/(hd_sec['CD2_2']*60.**2))-1)>1e-6
            if cond1 or cond2:
                print 'WARNING: Invalid pixel dimensions. Skipping calculation!'
                sys.stdout.flush()
                return True

            # Calculate how many pixels should be added to the original mask
            div1, mod1 = np.divmod(mask_sec.shape[0], dim_ratio)
            div2, mod2 = np.divmod(mask_sec.shape[1], dim_ratio)
            if mod1 == 0:
                x1 = div1
            else:
                x1 = div1+1
            if mod2 == 0:
                x2 = div2
            else:
                x2 = div2+1
            start1 = int(np.round((x1*dim_ratio-mask_sec.shape[0])/2.))
            start2 = int(np.round((x2*dim_ratio-mask_sec.shape[1])/2.))
            end1 = start1+mask_sec.shape[0]
            end2 = start2+mask_sec.shape[1]

            # Add borders to the mask
            mask_ext = np.zeros((x1*dim_ratio, x2*dim_ratio), dtype=np.int8)
            mask_ext[start1:end1,start2:end2] = mask_sec

            # Calculate new mask
            mask = np.zeros((x1, x2))
            for count1 in range(x1):
                for count2 in range(x2):
                    s1 = count1*dim_ratio
                    s2 = count2*dim_ratio
                    new_pix = mask_ext[s1:s1+dim_ratio,s2:s2+dim_ratio].astype(float)
                    mask[count1,count2] = np.average(new_pix)

            # Create header
            w = wcs.WCS(naxis=2)
            w.wcs.crpix = np.array([start1+hd_sec['CRPIX1'], start2+hd_sec['CRPIX2']])/dim_ratio
            w.wcs.cdelt = np.array([hd_sec['CD1_1'], hd_sec['CD2_2']])*dim_ratio
            w.wcs.crval = np.array([hd_sec['CRVAL1'], hd_sec['CRVAL2']])
            w.wcs.ctype = [hd_sec['CTYPE1'], hd_sec['CTYPE2']]
            hd = w.to_header()

            # Print message
            print '----> Degraded mask for '+f+'. Now I will remove the bad fields!'
            sys.stdout.flush()


            # Remove bad fields from mask
            imname = 'primary'
            for url in urls:
                badname = path['base']+'/'+os.path.split(url)[1]
                # Get the file if it is not there
                if not(os.path.exists(badname) or os.path.exists(badname+'.gz')):
                    urllib.urlretrieve(url, badname)
                # Compress file
                if os.path.exists(badname):
                    with open(badname, 'rb') as f_in:
                        with gzip.open(badname+'.gz', 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                if os.path.exists(badname+'.gz'):
                    try:
                        os.remove(badname)
                    except:
                        pass
                    badname = badname + '.gz'
                # Read the mask
                mask_bad = io.read_from_fits(badname, imname).astype(np.int16)
                hd_bad = io.read_header_from_fits(badname, imname)
                w_bad = wcs.WCS(hd_bad)
                # Find pixels inside the field
                n_arrs = 7
                div = np.divide(mask_bad.shape[0],n_arrs)
                starts = np.array([x*div for x in range(n_arrs) if x*div<mask_bad.shape[0]],dtype=np.int32)
                ends = [s+div for s in starts]
                ends[-1] = mask_bad.shape[0]
                for start, end in zip(starts, ends):
                    pos_bad = (start,0)+np.stack(np.where(mask_bad[start:end]<8192), axis=-1).astype(np.int32)
                    pos_bad = w_bad.wcs_pix2world(pos_bad, 1).astype(np.float32)
                    pos_bad = np.around(w.wcs_world2pix(pos_bad, 1)).astype(np.int32)
                    pos_bad = np.flip(pos_bad,axis=1) #Need to invert the columns
                    pos_bad = np.unique(pos_bad, axis=0)
                    mask[pos_bad[:,0], pos_bad[:,1]] = 0
                # Print message
                print '----> Removed bad field '+os.path.split(url)[1]+' from '+f+' mask!'
                sys.stdout.flush()
                # Remove file to save space
                if args.remove_files:
                    os.remove(badname)


            # Save to file the map
            name = 'MASK_{}'.format(f)
            warning = io.write_to_fits(path['mask_'+f], mask, name, header=hd, type='image') or warning

            io.print_info_fits(path['mask_'+f])

            # Generate plots
            if args.want_plots:
                plt.imshow(mask,interpolation='nearest')
                plt.colorbar()
                plt.savefig(path['base']+'/mask_'+f+'.pdf')
                plt.close()


        return warning



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
            sys.stdout.flush()
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'c2', 'weight', 'id', 'Z_B', 'MASK', 'star_flag']
        for key in table_keys:
            if key not in cat.columns.names:
                print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True


        # Main loop: scan over the fields
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
                warning = io.write_to_fits(path['cat_'+f], gals, name, type='table') or warning

            io.print_info_fits(path['cat_'+f])

        return warning



# ------------------- Function to calculate the multiplicative correction -----#

    def run_mult(path=path, fields=fields, z_bins=z_bins, n_avg_m=n_avg_m):

        print 'Running MULT_CORR module'
        sys.stdout.flush()
        warning = False


        # Read galaxy catalogue
        table_name = 'data'
        fname = path['cat_full']
        try:
            cat = io.read_from_fits(fname, table_name)
        except KeyError:
            print 'WARNING: No key '+table_name+' in '+fname+'. Skipping calculation!'
            sys.stdout.flush()
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'm', 'weight', 'id', 'Z_B', 'MASK', 'star_flag']
        for key in table_keys:
            if key not in cat.columns.names:
                print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True


        # Main loop: scan over the fields
        for f in fields:

            # Remove old output file to avoid confusion
            try:
                os.remove(path['m_'+f])
            except:
                pass

            # Read mask and create WCS object
            imname = 'MASK_{}'.format(f)
            fname = path['mask_'+f]
            try:
                mask = io.read_from_fits(fname, imname)
                hd = io.read_header_from_fits(fname, imname)
            except KeyError:
                print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True

            # Create a new WCS object
            w = wcs.WCS(hd)

            # Second loop to divide galaxies in redshift bins
            for n_z_bin, z_bin in enumerate(z_bins):

                print 'Calculating multiplicative correction for field ' + f + ' and bin {}:'.format(n_z_bin+1)
                sys.stdout.flush()


                # Create an empty array for the multiplicative correction
                mult_corr = np.zeros(mask.shape)

                # Filter galaxies
                filter = set.filter_galaxies(cat, z_bin[0], z_bin[1], field=f)
                gals = cat[filter]

                # Get World position of each galaxy
                pos = zip(gals['ALPHA_J2000'],gals['DELTA_J2000'])
                # Calculate Pixel position of each galaxy
                pos = w.wcs_world2pix(pos, 1).astype(int)
                pos = np.flip(pos,axis=1) #Need to invert the columns
                # Pixels where at least one galaxy has been found
                pos_unique = np.unique(pos, axis=0)
                # Scan over the populated pixels
                for count, pix in enumerate(pos_unique):
                    # Calculate range of pixels to average
                    if pix[0]-n_avg_m<0:
                        s1 = 0
                    elif pix[0]+n_avg_m>=mult_corr.shape[0]:
                        s1 = mult_corr.shape[0]-(2*n_avg_m+1)
                    else:
                        s1 = pix[0]-n_avg_m
                    if pix[1]-n_avg_m<0:
                        s2 = 0
                    elif pix[1]+n_avg_m>=mult_corr.shape[1]:
                        s2 = mult_corr.shape[1]-(2*n_avg_m+1)
                    else:
                        s2 = pix[1]-n_avg_m
                    # Select galaxies in range of pixels
                    sel = pos[:,0] >= s1
                    sel = (pos[:,0] < s1+2*n_avg_m+1)*sel
                    sel = (pos[:,1] >= s2)*sel
                    sel = (pos[:,1] < s2+2*n_avg_m+1)*sel
                    m = gals[sel]['m']
                    weight = gals[sel]['weight']
                    mult_corr[tuple(pix)] = np.average(m, weights=weight)

                    # Print message every some step
                    if (count+1) % 1e3 == 0:
                        print '----> Done {0:5.1%} of the pixels ({1:d})'.format(float(count+1) /len(pos_unique), len(pos_unique))
                        sys.stdout.flush()

                # Save to file the map
                name = 'MULT_CORR_{}_Z{}'.format(f, n_z_bin+1)
                warning = io.write_to_fits(path['m_'+f], mult_corr, name, header=hd, type='image') or warning

                # Generate plots
                if args.want_plots:
                    plt.imshow(mult_corr,interpolation='nearest')
                    plt.colorbar()
                    plt.savefig(path['base']+'/mult_corr_{}_z{}.pdf'.format(f, n_z_bin+1))
                    plt.close()

            io.print_info_fits(path['m_'+f])


        return warning



# ------------------- Function to calculate the multiplicative correction -----#

    def run_pz(path=path, z_bins=z_bins):

        print 'Running PHOTO_Z module'
        sys.stdout.flush()
        warning = False

        return warning



# ------------------- Function to calculate the map ---------------------------#

    def run_map(path=path, fields=fields, z_bins=z_bins):

        print 'Running MAP module'
        sys.stdout.flush()
        warning = False

        return warning



# ------------------- Pipeline ------------------------------------------------#

    if is_run_mask:
        start = time.clock()
        warning = run_mask() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run MASK module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    if is_run_cat:
        start = time.clock()
        warning = run_cat() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run CATALOGUE module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    if is_run_mult:
        start = time.clock()
        warning = run_mult() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run MULT_CORR module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    # if is_run_pz:
    #     start = time.clock()
    #     warning = run_pz() or warning
    #     end = time.clock()
    #     hrs, rem = divmod(end-start, 3600)
    #     mins, secs = divmod(rem, 60)
    #     print 'Run PHOTO_Z module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
    #     sys.stdout.flush()
    # if is_run_map:
    #     start = time.clock()
    #     warning = run_map() or warning
    #     end = time.clock()
    #     hrs, rem = divmod(end-start, 3600)
    #     mins, secs = divmod(rem, 60)
    #     print 'Run MAP module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
    #     sys.stdout.flush()

    if warning:
        print 'Done! However something went unexpectedly!! Check your warnings!'
        sys.stdout.flush()
    else:
        print 'Success!!'
        sys.stdout.flush()




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
