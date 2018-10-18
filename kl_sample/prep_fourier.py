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
import prep_fourier_tools as tools


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
    n_avg_m = 2
    # Bandpowers to calculate Cl's
    bandpowers = np.array([[  30,   80],
                           [  80,  260],
                           [ 260,  450],
                           [ 450,  670],
                           [ 670, 1310],
                           [1310, 2300],
                           [2300, 5100]])



# ------------------- Initialize paths ----------------------------------------#

    # Define absolute paths
    path = {}
    path['base'], path['fname'] = os.path.split(os.path.abspath(args.input_path))
    io.path_exists_or_error(path['base'])
    path['cat_full'] = '{}/{}cat_full.fits'.format(path['base'],path['fname'])
    path['mask_url'] = '{}/{}mask_url.txt'.format(path['base'],path['fname'])
    path['photo_z'] = '{}/photo_z.fits'.format(path['base'])
    for f in fields:
        path['mask_sec_'+f] = '{}/{}mask_arcsec_{}.fits'.format(path['base'],path['fname'],f)
        path['mask_'+f] = '{}/mask_{}.fits'.format(path['base'],f)
        path['m_'+f] = '{}/{}mult_corr_{}.fits'.format(path['base'],path['fname'],f)
        path['cat_'+f] = '{}/{}cat_{}.fits'.format(path['base'],path['fname'],f)
        path['map_'+f] = '{}/{}map_{}.fits'.format(path['base'],path['fname'],f)
        path['cl_'+f] = '{}/cl_{}.fits'.format(path['base'],f)
        path['cl_noise_'+f] = '{}/cl_noise_{}.fits'.format(path['base'],f)

    # Determine which modules have to be run, by checking the existence of the
    # output files and arguments passed by the user
    is_run_mask = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
    if args.run_mask or args.run_all:
        is_run_mask = True
    is_run_mult = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
    if args.run_mult or args.run_all:
        is_run_mult = True
    is_run_pz = not(os.path.exists(path['photo_z']))
    if args.run_pz or args.run_all:
        is_run_pz = True
    is_run_cat = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
    if args.run_cat or args.run_all:
        is_run_cat = True
    is_run_map = np.array([not(os.path.exists(path['map_'+f])) for f in fields]).any()
    if args.run_map or args.run_all:
        is_run_map = True
    is_run_cl = np.array([not(os.path.exists(path['cl_'+f])) for f in fields]).any()
    if args.run_cl or args.run_all:
        is_run_cl = True
    is_run_cl_noise = np.array([not(os.path.exists(path['cl_noise_'+f])) for f in fields]).any()
    if args.run_cl_noise or args.run_all:
        is_run_cl_noise = True

    # Check the existence of the required input files
    if is_run_mask:
        nofile1 = not(os.path.exists(path['mask_url']))
        nofile2 = np.array([not(os.path.exists(path['mask_sec_'+f])) for f in fields]).any()
        nofile3 = not(os.path.exists(path['cat_full']))
        if nofile1 or nofile2 or nofile3:
            print 'WARNING: I will skip the MASK module. Input files not found!'
            sys.stdout.flush()
            is_run_mask = False
            warning = True
    else:
        print 'I will skip the MASK module. Output files already there!'
        sys.stdout.flush()
    if is_run_mult:
        nofile1 = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
        nofile2 = not(os.path.exists(path['cat_full']))
        if (not(is_run_mask) and nofile1) or nofile2:
            print 'WARNING: I will skip the MULT_CORR module. Input file not found!'
            sys.stdout.flush()
            is_run_mult = False
            warning = True
    else:
        print 'I will skip the MULT_CORR module. Output files already there!'
        sys.stdout.flush()
    if is_run_pz:
        nofile1 = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
        nofile2 = not(os.path.exists(path['cat_full']))
        nofile3 = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
        if (not(is_run_mask) and nofile1) or nofile2 or (not(is_run_mult) and nofile3):
            print 'WARNING: I will skip the PHOTO_Z module. Input files not found!'
            sys.stdout.flush()
            is_run_pz = False
            warning = True
    else:
        print 'I will skip the PHOTO_Z module. Output file already there!'
        sys.stdout.flush()
    if is_run_cat:
        nofile1 = not(os.path.exists(path['cat_full']))
        nofile2 = np.array([not(os.path.exists(path['m_'+f])) for f in fields]).any()
        if nofile1 or (not(is_run_mult) and nofile2):
            print 'WARNING: I will skip the CATALOGUE module. Input file not found!'
            sys.stdout.flush()
            is_run_cat = False
            warning = True
    else:
        print 'I will skip the CATALOGUE module. Output files already there!'
        sys.stdout.flush()
    if is_run_map:
        nofile1 = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
        if (not(is_run_mask) and nofile1) or (not(is_run_cat) and nofile2):
            print 'WARNING: I will skip the MAP module. Input files not found!'
            sys.stdout.flush()
            is_run_map = False
            warning = True
    else:
        print 'I will skip the MAP module. Output files already there!'
        sys.stdout.flush()
    if is_run_cl:
        nofile1 = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(os.path.exists(path['map_'+f])) for f in fields]).any()
        if (not(is_run_mask) and nofile1) or (not(is_run_map) and nofile2):
            print 'WARNING: I will skip the CL module. Input files not found!'
            sys.stdout.flush()
            is_run_cl = False
            warning = True
    else:
        print 'I will skip the CL module. Output files already there!'
        sys.stdout.flush()
    if is_run_cl_noise:
        nofile1 = np.array([not(os.path.exists(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(os.path.exists(path['cat_'+f])) for f in fields]).any()
        if (not(is_run_mask) and nofile1) or (not(is_run_cat) and nofile2):
            print 'WARNING: I will skip the CL_NOISE module. Input files not found!'
            sys.stdout.flush()
            is_run_cl_noise = False
            warning = True
    else:
        print 'I will skip the CL_NOISE module. Output files already there!'
        sys.stdout.flush()



# ------------------- Function to calculate the mask --------------------------#

    def run_mask(path=path, fields=fields, z_bins=z_bins, size_pix=size_pix):

        print 'Running MASK module'
        sys.stdout.flush()
        warning = False

        # Read galaxy catalogue
        tabname = 'data'
        fname = path['cat_full']
        try:
            cat = io.read_from_fits(fname, tabname)
        except KeyError:
            print 'WARNING: No key '+tabname+' in '+fname+'. Skipping calculation!'
            sys.stdout.flush()
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'id']
        for key in table_keys:
            if key not in cat.columns.names:
                print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True


        # First loop: scan over the fields and generate new maps
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
                    pos_bad = np.flip(pos_bad,axis=1) #Need to invert the columns
                    pos_bad = w_bad.wcs_pix2world(pos_bad, 0).astype(np.float32)
                    pos_bad = np.around(w.wcs_world2pix(pos_bad, 0)).astype(np.int32)
                    pos_bad = np.flip(pos_bad,axis=1) #Need to invert the columns
                    pos_bad = np.unique(pos_bad, axis=0)
                    mask[pos_bad[:,0], pos_bad[:,1]] = 0
                # Print message
                print '----> Removed bad field '+os.path.split(url)[1]+' from '+f+' mask!'
                sys.stdout.flush()
                # Remove file to save space
                if args.remove_files:
                    os.remove(badname)

            # Remove bad galaxies from mask
            bad_fields = [x.split('_')[0].split('/')[-1] for x in urls]
            filter = np.array([x[:6] in bad_fields for x in cat['id']])
            pos = zip(cat['ALPHA_J2000'][filter],cat['DELTA_J2000'][filter])
            # Calculate Pixel position of each galaxy
            pos = w.wcs_world2pix(pos, 0).astype(int)
            pos = np.flip(pos,axis=1) #Need to invert the columns
            # Pixels where at least one galaxy has been found
            pos = np.unique(pos, axis=0)
            mask[pos[:,0], pos[:,1]] = 0
            print '----> Removed galaxies in bad fields for '+f+'!'
            sys.stdout.flush()


            # Save to file the mask
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



# ------------------- Function to calculate the multiplicative correction -----#

    def run_mult(path=path, fields=fields, z_bins=z_bins, n_avg_m=n_avg_m):

        print 'Running MULT_CORR module'
        sys.stdout.flush()
        warning = False


        # Read galaxy catalogue
        tabname = 'data'
        fname = path['cat_full']
        try:
            cat = io.read_from_fits(fname, tabname)
        except KeyError:
            print 'WARNING: No key '+tabname+' in '+fname+'. Skipping calculation!'
            sys.stdout.flush()
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'm', 'weight', 'id', 'Z_B', 'MASK', 'star_flag']
        for key in table_keys:
            if key not in cat.columns.names:
                print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True


        # First loop: scan over the fields
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

            # Second loop: divide galaxies in redshift bins
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
                pos = w.wcs_world2pix(pos, 0).astype(int)
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

                # # Generate plots
                # if args.want_plots:
                #     plt.imshow(mult_corr,interpolation='nearest')
                #     plt.colorbar()
                #     plt.savefig(path['base']+'/'+path['fname']+'mult_corr_{}_z{}.pdf'.format(f, n_z_bin+1))
                #     plt.close()

            io.print_info_fits(path['m_'+f])


        return warning



# ------------------- Function to calculate the photo_z -----------------------#

    def run_pz(path=path, z_bins=z_bins, fields=fields):

        print 'Running PHOTO_Z module'
        sys.stdout.flush()
        warning = False


        # Remove old output file to avoid confusion
        try:
            os.remove(path['photo_z'])
        except:
            pass

        # Read mask
        mask = {}
        w = {}
        for f in fields:
            # Read mask and create WCS object
            imname = 'MASK_{}'.format(f)
            fname = path['mask_'+f]
            try:
                mask[f] = io.read_from_fits(fname, imname)
                hd = io.read_header_from_fits(fname, imname)
            except KeyError:
                print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True
            # Create a new WCS object
            w[f] = wcs.WCS(hd)

        # Read galaxy catalogue
        tabname = 'data'
        imname = 'pz_full'
        fname = path['cat_full']
        try:
            cat = io.read_from_fits(fname, tabname)
        except KeyError:
            print 'WARNING: No key '+tabname+' in '+fname+'. Skipping calculation!'
            sys.stdout.flush()
            return True
        try:
            pz_full = io.read_from_fits(fname, imname)
        except KeyError:
            print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
            sys.stdout.flush()
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'm', 'weight', 'id', 'Z_B', 'MASK', 'star_flag']
        for key in table_keys:
            if key not in cat.columns.names:
                print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True

        # Read multiplicative corrections
        m = {}
        for f in fields:
            m[f] = {}
            fname = path['m_'+f]
            for n_z_bin, z_bin in enumerate(z_bins):
                imname = 'MULT_CORR_{}_Z{}'.format(f, n_z_bin+1)
                try:
                    m[f][n_z_bin] = io.read_from_fits(fname, imname)
                except KeyError:
                    print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
                    sys.stdout.flush()
                    return True


        # Create filters for each bin and field
        filter = {}
        for f in fields:
            filter[f] = {}
            for n_z_bin, z_bin in enumerate(z_bins):
                filt = set.filter_galaxies(cat, z_bin[0], z_bin[1], field=f)
                pix = np.transpose([cat[filt]['ALPHA_J2000'],cat[filt]['DELTA_J2000']])
                pix = w[f].wcs_world2pix(pix, 0).astype(int)
                masked = np.where(np.array([mask[f][iy,ix] for ix,iy in pix])<=0)[0]
                filt[filt][masked] = False
                filter[f][n_z_bin] = filt
        # Print progress message
        print '----> Created filters!'
        sys.stdout.flush()

        # Correct ellipticities
        m_corr = np.zeros(len(cat))
        for f in fields:
            for n_z_bin, z_bin in enumerate(z_bins):
                filt = filter[f][n_z_bin]
                pix = np.transpose([cat[filt]['ALPHA_J2000'],cat[filt]['DELTA_J2000']])
                pix = w[f].wcs_world2pix(pix, 0).astype(int)
                m_corr[filt] = np.array([m[f][n_z_bin][iy,ix] for ix,iy in pix])
        cat['e1'] = cat['e1']/(1+m_corr)
        cat['e2'] = (cat['e2']-cat['c2'])/(1+m_corr)
        # Print progress message
        print '----> Corrected ellipticities!'
        sys.stdout.flush()


        # Useful functions (area, n_eff, sigma_g)
        def get_area(fields, mask=mask, size_pix=size_pix):
            area = 0.
            for f in fields:
                area += mask[f].sum()*(size_pix/60.)**2.
            return area
        def get_n_eff(cat, area):
            wsum2 = (cat['weight'].sum())**2
            w2sum = (cat['weight']**2).sum()
            return wsum2/w2sum/area
        def get_sigma_g(cat):
            w2 = cat['weight']**2
            sg = np.dot(w2, (cat['e1']**2. + cat['e2']**2.)/2.)/w2.sum()
            return sg**0.5


        # Initialize quantities
        n_eff = np.zeros(len(z_bins))
        sigma_g = np.zeros(len(z_bins))
        photo_z = np.zeros((len(z_bins)+1,len(pz_full[0])))
        photo_z[0] = (np.arange(len(pz_full[0]))+1./2.)*set.dZ_CFHTlens
        n_eff_f = np.zeros((len(fields),len(z_bins)))
        sigma_g_f = np.zeros((len(fields),len(z_bins)))
        photo_z_f = np.zeros((len(fields),len(z_bins)+1,len(pz_full[0])))
        for count in range(len(fields)):
            photo_z_f[count,0] = (np.arange(len(pz_full[0]))+1./2.)*set.dZ_CFHTlens

        # First loop: scan over redshift bins
        for n_z_bin, z_bin in enumerate(z_bins):

            # Calculate quantities for each redshift bin

            # Merge filters
            sel = np.zeros(len(cat),dtype=bool)
            for f in fields:
                sel += filter[f][n_z_bin]
            # Filter galaxies
            gals = cat[sel]
            pz_z = pz_full[sel]
            # Get n_eff
            area = get_area(fields)
            n_eff[n_z_bin] = get_n_eff(gals, area)
            # Get sigma_g
            sigma_g[n_z_bin] = get_sigma_g(gals)
            # Get photo_z
            photo_z[n_z_bin+1] = np.average(pz_z, weights=gals['weight'], axis=0)


            # Calculate quantities for each redshift bin and field
            for count, f in enumerate(fields):
                # Filter galaxies
                gals = cat[filter[f][n_z_bin]]
                pz_z = pz_full[filter[f][n_z_bin]]
                # Get n_eff
                area = get_area([f])
                n_eff_f[count,n_z_bin] = get_n_eff(gals, area)
                # Get sigma_g
                sigma_g_f[count,n_z_bin] = get_sigma_g(gals)
                # Get photo_z
                photo_z_f[count,n_z_bin+1] = np.average(pz_z, weights=gals['weight'], axis=0)


            # Print progress message
            print '----> Completed bin {}'.format(n_z_bin+1)
            sys.stdout.flush()

        # Save to file the results
        warning = io.write_to_fits(path['photo_z'], photo_z, 'PHOTO_Z', type='image') or warning
        warning = io.write_to_fits(path['photo_z'], n_eff, 'N_EFF', type='image') or warning
        warning = io.write_to_fits(path['photo_z'], sigma_g, 'SIGMA_G', type='image') or warning
        warning = io.write_to_fits(path['photo_z'], photo_z_f, 'PHOTO_Z_PF', type='image') or warning
        warning = io.write_to_fits(path['photo_z'], n_eff_f, 'N_EFF_PF', type='image') or warning
        warning = io.write_to_fits(path['photo_z'], sigma_g_f, 'SIGMA_G_PF', type='image') or warning

        # Generate plots
        if args.want_plots:
            x = photo_z[0]
            for count in range(1,len(photo_z)):
                y = photo_z[count]
                plt.plot(x, y, label = 'Bin ' + str(count))
            plt.xlim(0.,2.)
            plt.xlabel('$z$', fontsize=14)
            plt.ylabel('Probability distribution', fontsize=14)
            plt.title('Photo-z')
            plt.legend(loc="upper right", frameon = False, fontsize=9, labelspacing=0.01)
            plt.savefig(path['base']+'/photo_z.pdf')
            plt.close()
            for n_f, f in enumerate(fields):
                x = photo_z_f[n_f,0]
                for count in range(1,photo_z_f.shape[1]):
                    y = photo_z_f[n_f,count]
                    plt.plot(x, y, label = 'Bin ' + str(count))
                plt.xlim(0.,2.)
                plt.xlabel('$z$', fontsize=14)
                plt.ylabel('Probability distribution', fontsize=14)
                plt.title('Photo-z {}'.format(f))
                plt.legend(loc="upper right", frameon = False, fontsize=9, labelspacing=0.01)
                plt.savefig(path['base']+'/photo_z_{}.pdf'.format(f))
                plt.close()

        io.print_info_fits(path['photo_z'])

        return warning



# ------------------- Function to calculate the clean catalogue ---------------#

    def run_cat(path=path, fields=fields, z_bins=z_bins):

        print 'Running CATALOGUE module'
        sys.stdout.flush()
        warning = False


        # Read galaxy catalogue
        tabname = 'data'
        fname = path['cat_full']
        try:
            cat = io.read_from_fits(fname, tabname)
        except KeyError:
            print 'WARNING: No key '+tabname+' in '+fname+'. Skipping calculation!'
            sys.stdout.flush()
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'c2', 'weight', 'id', 'Z_B', 'MASK', 'star_flag']
        for key in table_keys:
            if key not in cat.columns.names:
                print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True


        # First loop: scan over the fields
        for f in fields:

            # Remove old output file to avoid confusion
            try:
                os.remove(path['cat_'+f])
            except:
                pass

            # Second loop: divide galaxies in redshift bins
            for n_z_bin, z_bin in enumerate(z_bins):

                print 'Calculating catalogue for field ' + f + ' and bin {}:'.format(n_z_bin+1)
                sys.stdout.flush()


                # Read multiplicative corrections
                fname = path['m_'+f]
                imname = 'MULT_CORR_{}_Z{}'.format(f, n_z_bin+1)
                try:
                    m = io.read_from_fits(fname, imname)
                    hd = io.read_header_from_fits(fname, imname)
                except KeyError:
                    print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
                    sys.stdout.flush()
                    return True
                # Create a new WCS object
                w = wcs.WCS(hd)

                # Filter galaxies
                filter = set.filter_galaxies(cat, z_bin[0], z_bin[1], field=f)
                gals = cat[filter]

                # Calculate corrected ellipticities
                def find_m_correction(gal):
                    pix = w.wcs_world2pix([[gal['ALPHA_J2000'],gal['DELTA_J2000']]],0)
                    pix = tuple(np.flip(pix.astype(int),axis=1)[0])
                    return m[pix]
                m_corr = np.array([find_m_correction(gal) for gal in gals])
                # Ellipticities
                gals['e1'] = gals['e1']/(1+m_corr)
                gals['e2'] = (gals['e2']-gals['c2'])/(1+m_corr)

                # Create Table and save it
                table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']
                columns = []
                for key in table_keys:
                    columns.append(fits.Column(name=key,array=gals[key],format='E'))
                name = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
                gals = fits.BinTableHDU.from_columns(columns, name=name)
                warning = io.write_to_fits(path['cat_'+f], gals, name, type='table') or warning

            io.print_info_fits(path['cat_'+f])

        return warning



# ------------------- Function to calculate the map ---------------------------#

    def run_map(path=path, fields=fields, z_bins=z_bins):

        print 'Running MAP module'
        sys.stdout.flush()
        warning = False


        # First loop: scan over the fields
        for f in fields:

            # Remove old output file to avoid confusion
            try:
                os.remove(path['map_'+f])
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

            # Second loop: divide galaxies in redshift bins
            for n_z_bin, z_bin in enumerate(z_bins):

                print 'Calculating map for field ' + f + ' and bin {}:'.format(n_z_bin+1)
                sys.stdout.flush()


                # Read galaxy catalogue
                tabname = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
                fname = path['cat_'+f]
                try:
                    cat = io.read_from_fits(fname, tabname)
                except KeyError:
                    print 'WARNING: No key '+tabname+' in '+fname+'. Skipping calculation!'
                    sys.stdout.flush()
                    return True

                # Check that the table has the correct columns
                table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']
                for key in table_keys:
                    if key not in cat.columns.names:
                        print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
                        sys.stdout.flush()
                        return True

                # Get map
                map_1, map_2, _ = tools.get_map(w, mask, cat)


                # Save to file the map
                name = 'MAP_{}_Z{}_G1'.format(f, n_z_bin+1)
                warning = io.write_to_fits(path['map_'+f], map_1, name, header=hd, type='image') or warning
                name = 'MAP_{}_Z{}_G2'.format(f, n_z_bin+1)
                warning = io.write_to_fits(path['map_'+f], map_2, name, header=hd, type='image') or warning

                # Generate plots
                if args.want_plots:
                    plt.imshow(map_1,interpolation='nearest')
                    plt.colorbar()
                    plt.savefig(path['base']+'/'+path['fname']+'map_{}_z{}_g1.pdf'.format(f, n_z_bin+1))
                    plt.close()
                    plt.imshow(map_2,interpolation='nearest')
                    plt.colorbar()
                    plt.savefig(path['base']+'/'+path['fname']+'map_{}_z{}_g2.pdf'.format(f, n_z_bin+1))
                    plt.close()

            io.print_info_fits(path['map_'+f])

        return warning



# ------------------- Function to calculate the cl ----------------------------#

    def run_cl(path=path, fields=fields, z_bins=z_bins, bp=bandpowers):

        print 'Running CL module'
        sys.stdout.flush()
        warning = False


        # First loop: scan over the fields
        for f in fields:

            print 'Calculating cl for field {}:'.format(f,)
            sys.stdout.flush()

            # Remove old output file to avoid confusion
            try:
                os.remove(path['cl_'+f])
            except:
                pass

            # Read mask
            imname = 'MASK_{}'.format(f)
            fname = path['mask_'+f]
            try:
                mask = io.read_from_fits(fname, imname)
                hd = io.read_header_from_fits(fname, imname)
            except KeyError:
                print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
                sys.stdout.flush()
                return True

            # Read maps
            fname = path['map_'+f]
            n_pol = 2 # Number of shear polarizations (always 2)
            map = np.zeros((len(z_bins),n_pol)+(mask.shape))
            # Get maps for each bin and polarization
            for n_z_bin, z_bin in enumerate(z_bins):
                for count in range(n_pol):
                    t = 'MAP_{}_Z{}_G{}'.format(f, n_z_bin+1,count+1)
                    try:
                        map[n_z_bin,count] = io.read_from_fits(fname, t)
                    except KeyError:
                        print 'WARNING: No key '+t+' in '+fname+'. Skipping calculation!'
                        sys.stdout.flush()
                        return True

            # Get map
            cl, _ = tools.get_cl(bp, mask, map)

            # Save to file the map
            name = 'CL_{}'.format(f)
            warning = io.write_to_fits(path['cl_'+f], cl, name, type='image') or warning

        #         # Generate plots #TODO
        #         if args.want_plots:
        #             plt.imshow(map_1,interpolation='nearest')
        #             plt.colorbar()
        #             plt.savefig(path['base']+'/'+path['fname']+'map_{}_z{}_g1.pdf'.format(f, n_z_bin+1))
        #             plt.close()
        #             plt.imshow(map_2,interpolation='nearest')
        #             plt.colorbar()
        #             plt.savefig(path['base']+'/'+path['fname']+'map_{}_z{}_g2.pdf'.format(f, n_z_bin+1))
        #             plt.close()
        #
            io.print_info_fits(path['cl_'+f])

        return warning



# ------------------- Function to calculate the cl ----------------------------#

    def run_cl_noise(path=path, fields=fields, z_bins=z_bins):

        print 'Running CL_NOISE module'
        sys.stdout.flush()
        warning = False


        # # First loop: scan over the fields
        # for f in fields:
        #
        #     # Remove old output file to avoid confusion
        #     try:
        #         os.remove(path['map_'+f])
        #     except:
        #         pass
        #
        #     # Read mask and create WCS object
        #     imname = 'MASK_{}'.format(f)
        #     fname = path['mask_'+f]
        #     try:
        #         mask = io.read_from_fits(fname, imname)
        #         hd = io.read_header_from_fits(fname, imname)
        #     except KeyError:
        #         print 'WARNING: No key '+imname+' in '+fname+'. Skipping calculation!'
        #         sys.stdout.flush()
        #         return True
        #     # Create a new WCS object
        #     w = wcs.WCS(hd)
        #
        #     # Second loop: divide galaxies in redshift bins
        #     for n_z_bin, z_bin in enumerate(z_bins):
        #
        #         print 'Calculating map for field ' + f + ' and bin {}:'.format(n_z_bin+1)
        #         sys.stdout.flush()
        #
        #
        #         # Read galaxy catalogue
        #         tabname = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
        #         fname = path['cat_'+f]
        #         try:
        #             cat = io.read_from_fits(fname, tabname)
        #         except KeyError:
        #             print 'WARNING: No key '+tabname+' in '+fname+'. Skipping calculation!'
        #             sys.stdout.flush()
        #             return True
        #
        #         # Check that the table has the correct columns
        #         table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']
        #         for key in table_keys:
        #             if key not in cat.columns.names:
        #                 print 'WARNING: No key '+key+' in table of '+fname+'. Skipping calculation!'
        #                 sys.stdout.flush()
        #                 return True
        #
        #         # Get map
        #         map_1, map_2, _ = tools.get_map(w, mask, cat)
        #
        #
        #         # Save to file the map
        #         name = 'MAP_{}_Z{}_G1'.format(f, n_z_bin+1)
        #         warning = io.write_to_fits(path['map_'+f], map_1, name, header=hd, type='image') or warning
        #         name = 'MAP_{}_Z{}_G2'.format(f, n_z_bin+1)
        #         warning = io.write_to_fits(path['map_'+f], map_2, name, header=hd, type='image') or warning
        #
        #         # Generate plots
        #         if args.want_plots:
        #             plt.imshow(map_1,interpolation='nearest')
        #             plt.colorbar()
        #             plt.savefig(path['base']+'/'+path['fname']+'map_{}_z{}_g1.pdf'.format(f, n_z_bin+1))
        #             plt.close()
        #             plt.imshow(map_2,interpolation='nearest')
        #             plt.colorbar()
        #             plt.savefig(path['base']+'/'+path['fname']+'map_{}_z{}_g2.pdf'.format(f, n_z_bin+1))
        #             plt.close()
        #
        #     io.print_info_fits(path['map_'+f])

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
    if is_run_mult:
        start = time.clock()
        warning = run_mult() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run MULT_CORR module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    if is_run_pz:
        start = time.clock()
        warning = run_pz() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run PHOTO_Z module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    if is_run_cat:
        start = time.clock()
        warning = run_cat() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run CATALOGUE module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    if is_run_map:
        start = time.clock()
        warning = run_map() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run MAP module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    if is_run_cl:
        start = time.clock()
        warning = run_cl() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run CL module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()
    if is_run_cl_noise:
        start = time.clock()
        warning = run_cl_noise() or warning
        end = time.clock()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print 'Run CL_NOISE module in {:0>2} Hours {:0>2} Minutes {:05.2f} Seconds!'.format(int(hrs),int(mins),secs)
        sys.stdout.flush()

    if warning:
        print 'Done! However something went unexpectedly!! Check your warnings!'
        sys.stdout.flush()
    else:
        print 'Success!!'
        sys.stdout.flush()
