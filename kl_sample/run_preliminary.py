"""

This module contains the pipeline to calculate the observed Cls
and the covariance matrix later used by the run module.

In the .ini file, under the section 'paths' key 'raw_data', it
should be specified the path of an input folder containing:
- cat_full.fits: full catalogue in fits format
- mask_arcsec_N.fits.gz: (N=1,..,4) 1 arcsec resolution masks for each field in
  compressed fits format
- mask_url.txt: list of urls from which to download the public masks. They
  will be used to remove the bad fields from mask_arcsec_N.fits.gz

"""

import gzip
import os
import re
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import shutil
import urllib
import kl_sample.io as io
import kl_sample.settings as setts
from astropy import wcs
from astropy.io import fits


def run_preliminary(args):
    """ Calculate observed Cls and covariance matrix.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to fits files file the output.

    """

    # ----------- Read ini file ----------------------------------------------#
    ini = io.IniFile(path=args.params_file, exists=True)
    ini.read()

    # ----------- Define local variables -------------------------------------#

    warning = False  # Global variable collecting warnings
    # List of CFHTlens fields
    fields = ini.read_param('fields_cfhtlens', 'cfhtlens',
                            type='list_of_strings')
    # Number of simulations used to calculate the covariance matrix
    n_sims_cov = ini.read_param('n_sims_cov', 'settings', type='int')
    # Number of simulations used to calculate the noise
    n_sims_noise = ini.read_param('n_sims_noise', 'settings', type='int')
    # List of redshift bins
    z_bins = ini.read_param('z_bins', 'settings', type='list_of_floats')
    z_bins = np.vstack((z_bins[:-1], z_bins[1:])).T
    # Size pixels masks in arcsecs (it has to be an integer number)
    size_pix = ini.read_param('size_pix', 'settings', type='int')
    # Range of pixels used to average the multiplicative correction
    n_avg_m = ini.read_param('n_avg_m', 'settings', type='int')
    # Bandpowers to calculate Cl's
    bandpowers = ini.read_param('bandpowers', 'settings',
                                type='list_of_floats')
    bandpowers = np.vstack((bandpowers[:-1], bandpowers[1:])).T
    # Use mode coupling matrix to couple theoretical Cells
    # (otherwise decouple observations)
    couple_cells_theory = ini.read_param('couple_cells_theory', 'settings',
                                         type='bool')

    # ----------- Initialize -------------------------------------------------#

    # Get paths
    paths = io.get_io_paths(ini, fields, want_plots=args.want_plots)
    # Determine which modules have to be run
    is_run, warning = is_run_and_check(args, fields, paths, n_sims_cov)

    # Compare ini files
    if paths['processed_ini'].exists:
        paths['processed_ini'].read()
        diffs = ini.get_diffs(paths['processed_ini'])
        if diffs:
            raise IOError('Ini files are different! Quitting the calculation!')
    elif not all(is_run.values()):
        raise IOError('Some processed data were found in the folder, but '
                      'without an ini file. It is safer to quit the '
                      'calculation!')
    else:
        msg = '# This is an automatically generated file. It can be\n'\
              '# used for a new run, but also to check the consistency\n'\
              '# between different runs. To be conservative, when using\n'\
              '# processed data to complete the run_preliminary pipeline\n'\
              '# or use them with run, this ini file and the one used as\n'\
              '# input should be identical.\n\n'
        paths['processed_ini'].write(content=ini.content, header=msg)

    # ----------- Run modules ------------------------------------------------#

    if is_run['mask']:
        start = time.time()
        warning = run_mask(paths, fields, z_bins, size_pix,
                           args.remove_files, args.want_plots) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run MASK module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    if is_run['mult']:
        start = time.time()
        warning = run_mult(paths, fields, z_bins, n_avg_m,
                           args.want_plots) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run MULT_CORR module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    if is_run['photo_z']:
        start = time.time()
        warning = run_photo_z(paths, fields, z_bins, size_pix,
                              args.want_plots) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run PHOTO_Z module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    if is_run['cat']:
        start = time.time()
        warning = run_cat(paths, fields, z_bins) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run CATALOGUE module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    if is_run['map']:
        start = time.time()
        warning = run_map(paths, fields, z_bins, args.want_plots) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run MAP module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    if is_run['cl']:
        start = time.time()
        warning = run_cl(paths, fields, z_bins,
                         bandpowers, n_sims_noise, couple_cells_theory,
                         args.want_plots) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run CL module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    if is_run['cat_sims']:
        start = time.time()
        warning = run_cat_sims(paths, fields, z_bins, n_sims_cov) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run CAT_SIMS module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    if is_run['cl_sims']:
        start = time.time()
        warning = run_cl_sims(paths, fields, z_bins, bandpowers,
                              couple_cells_theory) or warning
        end = time.time()
        hrs, rem = divmod(end-start, 3600)
        mins, secs = divmod(rem, 60)
        print('Run CL_SIMS module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
              ' Seconds!'.format(int(hrs), int(mins), secs))
        sys.stdout.flush()

    # Collect everything in a single file
    start = time.time()
    warning = run_final(paths, fields, z_bins, bandpowers) or warning
    end = time.time()
    hrs, rem = divmod(end-start, 3600)
    mins, secs = divmod(rem, 60)
    print('Run FINAL module in {:0>2} Hours {:0>2} Minutes {:05.2f}'
          ' Seconds!'.format(int(hrs), int(mins), secs))
    sys.stdout.flush()

    # ----------- Final messages ---------------------------------------------#

    if warning:
        print('Done! However something went unexpectedly! '
              'Check your warnings!')
        sys.stdout.flush()
    else:
        print('Success!!')
        sys.stdout.flush()

    return


# ------------------- Preliminary checks -------------------------------------#

def is_run_and_check(args, fields, paths, n_sims_cov):
    """ Determine which modules to run and do some preliminary check.

    Args:
        args: the arguments read by the parser.
        fields: list of the observed fields.
        paths: dictionary with all the necessary paths.
        n_sims_cov: number of simulations for the covariance matrix

    Returns:
        is_run: dictionary with the modules to be run.
        warning: true if it generated some warning.

    """
    # Local variables
    warning = False
    is_run = {}

    # Determine which modules have to be run, by checking the existence
    # of the output files and arguments passed by the user.
    # - Mask
    is_run['mask'] = any([not(paths['mask_'+f].exists) for f in fields])
    if args.run_mask or args.run_all:
        is_run['mask'] = True
    # - Multiplicative correction
    is_run['mult'] = any([not(paths['mult_'+f].exists) for f in fields])
    if args.run_mult or args.run_all:
        is_run['mult'] = True
    # - Photo z
    is_run['photo_z'] = not(paths['photo_z'].exists)
    if args.run_pz or args.run_all:
        is_run['photo_z'] = True
    # - Catalogue
    is_run['cat'] = any([not(paths['cat_'+f].exists) for f in fields])
    if args.run_cat or args.run_all:
        is_run['cat'] = True
    # - Map
    is_run['map'] = any([not(paths['map_'+f].exists) for f in fields])
    if args.run_map or args.run_all:
        is_run['map'] = True
    # - Cl
    is_run['cl'] = any([not(paths['cl_'+f].exists) for f in fields])
    if args.run_cl or args.run_all:
        is_run['cl'] = True
    # - Catalogue simulations
    is_run['cat_sims'] = not(paths['cat_sims'].exists)
    if paths['cat_sims'].exists:
        is_files = True
        for f in fields:
            sims = [x for x in os.listdir(paths['cat_sims'].path)
                    if re.match('.+{}\.fits'.format(f), x)]  # noqa:W605
            if len(sims) != n_sims_cov:
                is_files = False
        is_run['cat_sims'] = not(is_files)
    if args.run_cat_sims or args.run_all:
        is_run['cat_sims'] = True
    # - Cl simulations
    is_run['cl_sims'] = any([not(paths['cl_sims_'+f].exists) for f in fields])
    if args.run_cl_sims or args.run_all:
        is_run['cl_sims'] = True

    # Check the existence of the required input files
    # - Mask
    if is_run['mask']:
        nofile1 = not(paths['mask_url'].exists)
        nofile2 = any([not(paths['mask_sec_'+f].exists) for f in fields])
        nofile3 = not(paths['cat_full'].exists)
        if nofile1 or nofile2 or nofile3:
            print(
                'WARNING: I will skip the MASK module. Input files not found!')
            is_run['mask'] = False
            warning = True
    else:
        print('I will skip the MASK module. Output files already there!')
        sys.stdout.flush()
    # - Multiplicative correction
    if is_run['mult']:
        nofile1 = any([not(paths['mask_'+f].exists) for f in fields])
        nofile2 = not(paths['cat_full'].exists)
        if (not(is_run['mask']) and nofile1) or nofile2:
            print('WARNING: I will skip the MULT_CORR module. Input file not '
                  'found!')
            sys.stdout.flush()
            is_run['mult'] = False
            warning = True
    else:
        print('I will skip the MULT_CORR module. Output files already there!')
        sys.stdout.flush()
    # - Photo z
    if is_run['photo_z']:
        nofile1 = any([not(paths['mask_'+f].exists) for f in fields])
        nofile2 = not(paths['cat_full'].exists)
        nofile3 = any([not(paths['mult_'+f].exists) for f in fields])
        test1 = not(is_run['mask']) and nofile1
        test3 = not(is_run['mult']) and nofile3
        if test1 or nofile2 or test3:
            print('WARNING: I will skip the PHOTO_Z module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['photo_z'] = False
            warning = True
    else:
        print('I will skip the PHOTO_Z module. Output file already there!')
        sys.stdout.flush()
    # - Catalogue
    if is_run['cat']:
        nofile1 = not(paths['cat_full'].exists)
        nofile2 = any([not(paths['mult_'+f].exists) for f in fields])
        if nofile1 or (not(is_run['mult']) and nofile2):
            print('WARNING: I will skip the CATALOGUE module. Input file not '
                  'found!')
            sys.stdout.flush()
            is_run['cat'] = False
            warning = True
    else:
        print('I will skip the CATALOGUE module. Output files already there!')
        sys.stdout.flush()
    # - Map
    if is_run['map']:
        nofile1 = any([not(paths['mask_'+f].exists) for f in fields])
        nofile2 = any([not(paths['cat_'+f].exists) for f in fields])
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cat']) and nofile2
        if test1 or test2:
            print('WARNING: I will skip the MAP module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['map'] = False
            warning = True
    else:
        print('I will skip the MAP module. Output files already there!')
        sys.stdout.flush()
    # - Cl
    if is_run['cl']:
        nofile1 = any([not(paths['mask_'+f].exists) for f in fields])
        nofile2 = any([not(paths['cat_'+f].exists) for f in fields])
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cat']) and nofile2
        if test1 or test2:
            print('WARNING: I will skip the CL module. Input files not found!')
            sys.stdout.flush()
            is_run['cl'] = False
            warning = True
    else:
        print('I will skip the CL module. Output files already there!')
        sys.stdout.flush()
    # - Catalogue simulations
    if is_run['cat_sims']:
        nofile1 = any([not(paths['mask_'+f].exists) for f in fields])
        nofile2 = any([not(paths['cat_'+f].exists) for f in fields])
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cat']) and nofile2
        if test1 or test2:
            print('WARNING: I will skip the CAT_SIMS module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['cat_sims'] = False
            warning = True
    else:
        print('I will skip the CAT_SIMS module. Output files already there!')
        sys.stdout.flush()
    # - Cl simulations
    if is_run['cl_sims']:
        nofile1 = any([not(paths['mask_'+f].exists) for f in fields])
        nofile2 = any([not(paths['cl_'+f].exists) for f in fields])
        nofile3 = not(paths['cat_sims'].exists)
        if paths['cat_sims'].exists:
            is_files = True
            for f in fields:
                sims = [x for x in os.listdir(paths['cat_sims'].path)
                        if re.match('.+{}\.fits'.format(f), x)]  # noqa:W605
                if len(sims) != n_sims_cov:
                    is_files = False
            nofile3 = not(is_files)
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cl']) and nofile2
        test3 = not(is_run['cat_sims']) and nofile3
        if test1 or test2 or test3:
            print('WARNING: I will skip the CL_SIMS module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['cl_sims'] = False
            warning = True
    else:
        print('I will skip the CL_SIMS module. Output files already there!')
        sys.stdout.flush()

    return is_run, warning


# ------------------- Function to calculate the mask -------------------------#

def run_mask(paths, fields, z_bins, size_pix, remove_files, want_plots):

    print('Running MASK module')
    sys.stdout.flush()
    warning = False

    # Input files
    cat_full_file = paths['cat_full']
    mask_file = {f: paths['mask_'+f] for f in fields}
    mask_sec_file = {f: paths['mask_sec_'+f] for f in fields}
    mask_url_file = paths['mask_url']
    badfields_file = paths['badfields']
    if want_plots:
        plots_file = paths['plots']

    # Read galaxy catalogue
    tabname = 'data'
    try:
        cat = cat_full_file.read_key(tabname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping '
              'calculation!'.format(tabname, cat_full_file.path))
        sys.stdout.flush()
        return True

    # Check that the table has the correct columns
    table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'id']
    for key in table_keys:
        if key not in cat.columns.names:
            print('WARNING: No key {} in table of {}. Skipping '
                  'calculation!'.format(key, cat_full_file.path))
            sys.stdout.flush()
            return True

    # First loop: scan over the fields and generate new maps
    for f in fields:

        # Check existence of files and in case skip
        if mask_file[f].exists:
            keys = mask_file[f].get_keys()
            mnow = 'MASK_NOW_'+f in keys
            mzs = all(['MASK_{}_Z{}'.format(f, x+1) in keys
                      for x in range(len(z_bins))])
        if mask_file[f].exists and mnow and mzs:
            print('----> Skipping MASK calculation for field {}. Output file '
                  'already there!'.format(f))
            sys.stdout.flush()
            continue

        print('Calculating mask for field {}:'.format(f))
        sys.stdout.flush()

        # Remove old output file to avoid confusion
        try:
            mask_file[f].remove()
        except FileNotFoundError:
            pass

        # Read urls where to find coordinates of the bad fields
        urls = []
        for line in mask_url_file.readlines():
            cond1 = re.match('.+'+f+'.+finalmask_mosaic.fits', line)
            cond2 = not(np.array([re.match('.+'+x+'.+', line)
                                 for x in setts.good_fit_patterns])).any()
            if cond1 and cond2:
                urls.append(line.rstrip())
        # Read mask data
        imname = 'primary'
        try:
            mask_sec = mask_sec_file[f].read_key(imname, dtype=np.uint16)
        except KeyError:
            print('WARNING: No key {} in {}. Skipping '
                  'calculation!'.format(imname, mask_sec_file[f].path))
            sys.stdout.flush()
            return True
        # Read mask header and check necessary keys
        try:
            hd_sec = mask_sec_file[f].get_header(imname)
        except KeyError:
            print('WARNING: No header in {}. Skipping '
                  'calculation!'.format(mask_sec_file[f].path))
            sys.stdout.flush()
            return True
        for key in ['CRPIX1', 'CRPIX2', 'CD1_1', 'CD2_2', 'CRVAL1',
                    'CRVAL2', 'CTYPE1', 'CTYPE2']:
            if key not in list(hd_sec.keys()):
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(key, mask_sec_file[f].path))
                sys.stdout.flush()
                return True

        # Convert mask to boolean
        mask_sec = 1 - np.array(mask_sec, dtype=bool).astype(np.int8)

        # Determine how many pixels should be grouped together
        # in the degraded mask
        dim_ratio = int(np.round(abs(size_pix/(hd_sec['CD1_1']*60.**2))))
        cond1 = abs(dim_ratio/abs(size_pix/(hd_sec['CD1_1']*60.**2))-1) > 1e-6
        cond2 = abs(dim_ratio/abs(size_pix/(hd_sec['CD2_2']*60.**2))-1) > 1e-6
        if cond1 or cond2:
            print('WARNING: Invalid pixel dimensions. Skipping calculation!')
            sys.stdout.flush()
            return True

        # Calculate how many pixels should be added to the original mask
        div1, mod1 = np.divmod(mask_sec.shape[0], dim_ratio)
        div2, mod2 = np.divmod(mask_sec.shape[1], dim_ratio)
        if mod1 == 0:
            x1 = div1
        else:
            x1 = div1 + 1
        if mod2 == 0:
            x2 = div2
        else:
            x2 = div2 + 1
        start1 = int(np.round((x1*dim_ratio - mask_sec.shape[0])/2.))
        start2 = int(np.round((x2*dim_ratio - mask_sec.shape[1])/2.))
        end1 = start1 + mask_sec.shape[0]
        end2 = start2 + mask_sec.shape[1]

        # Add borders to the mask
        mask_ext = np.zeros((x1*dim_ratio, x2*dim_ratio), dtype=np.int8)
        mask_ext[start1:end1, start2:end2] = mask_sec

        # Calculate new mask
        mask = np.zeros((x1, x2))
        for count1 in range(x1):
            for count2 in range(x2):
                s1 = count1*dim_ratio
                s2 = count2*dim_ratio
                new_pix = \
                    mask_ext[s1:s1+dim_ratio, s2:s2+dim_ratio].astype(float)
                mask[count1, count2] = np.average(new_pix)

        # Create header
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = np.array([start1+hd_sec['CRPIX1'],
                                start2+hd_sec['CRPIX2']])/dim_ratio
        w.wcs.cdelt = np.array([hd_sec['CD1_1'], hd_sec['CD2_2']])*dim_ratio
        w.wcs.crval = np.array([hd_sec['CRVAL1'], hd_sec['CRVAL2']])
        w.wcs.ctype = [hd_sec['CTYPE1'], hd_sec['CTYPE2']]
        hd = w.to_header()

        # Print message
        print('----> Degraded mask for {}. Now I will remove the bad '
              'fields!'.format(f))
        sys.stdout.flush()

        # Remove bad fields from mask
        imname = 'primary'
        for nurl, url in enumerate(urls):
            badname = os.path.join(badfields_file.path, os.path.split(url)[1])
            # Get the file if it is not there
            if not(os.path.exists(badname) or os.path.exists(badname+'.gz')):
                urllib.request.urlretrieve(url, badname)
            # Compress file
            if os.path.exists(badname):
                with open(badname, 'rb') as f_in:
                    with gzip.open(badname+'.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            if os.path.exists(badname+'.gz'):
                try:
                    os.remove(badname)
                except FileNotFoundError:
                    pass
                badname = badname + '.gz'
            # Read the mask
            mask_bad_file = io.FitsFile(path=badname)
            mask_bad = mask_bad_file.read_key(imname, dtype=np.uint16)
            hd_bad = mask_bad_file.get_header(imname)
            w_bad = wcs.WCS(hd_bad)
            # Find pixels inside the field
            n_arrs = 7  # This is just a trick to make the problem tractable
            div = np.divide(mask_bad.shape[0], n_arrs)
            starts = np.array([x*div for x in range(n_arrs+1)], dtype=np.int32)
            for start, end in np.vstack((starts[:-1], starts[1:])).T:
                pos_bad = (start, 0) + np.stack(np.where(
                    mask_bad[start:end] < 8192), axis=-1).astype(np.int32)
                pos_bad = np.flip(pos_bad, axis=1)  # Invert the columns
                pos_bad = w_bad.wcs_pix2world(pos_bad, 0).astype(np.float32)
                pos_bad = np.around(
                    w.wcs_world2pix(pos_bad, 0)).astype(np.int32)
                pos_bad = np.flip(pos_bad, axis=1)  # Invert the columns
                pos_bad = np.unique(pos_bad, axis=0)
                mask[pos_bad[:, 0], pos_bad[:, 1]] = 0
            # Print message
            print('----> Removed bad field {} from {} mask! ({}/{})'
                  ''.format(os.path.split(url)[1], f, nurl+1, len(urls)))
            sys.stdout.flush()
            # Remove file to save space
            if remove_files:
                mask_bad_file.remove()

        # Remove bad galaxies from mask
        bad_fields = [x.split('_')[0].split('/')[-1] for x in urls]
        filter = np.array([x[:6] in bad_fields for x in cat['id']])
        pos = np.vstack(
            (cat['ALPHA_J2000'][filter], cat['DELTA_J2000'][filter])).T
        # Calculate Pixel position of each galaxy
        pos = w.wcs_world2pix(pos, 0).astype(int)
        pos = np.flip(pos, axis=1)  # Need to invert the columns
        # Pixels where at least one galaxy has been found
        pos = np.unique(pos, axis=0)
        mask[pos[:, 0], pos[:, 1]] = 0
        print('----> Removed galaxies in bad fields for {}!'.format(f))
        sys.stdout.flush()

        # Save to file the mask
        name = 'MASK_NOW_{}'.format(f)
        warning = mask_file[f].write(
            mask, name, header=hd, type='image') or warning

        # Generate plots
        if want_plots:
            plt.imshow(mask, interpolation='nearest')
            plt.colorbar()
            plt.savefig('{}/mask_now_{}.pdf'.format(plots_file.path, f))
            plt.close()

        # Second loop: calculate mask of weights
        mask = mask.astype(bool).astype(np.int8)
        for n_z_bin, z_bin in enumerate(z_bins):
            # Create an empty array for the weight mask
            weights_mask = np.zeros(mask.shape)

            # Filter galaxies
            sel = setts.filter_galaxies(cat, z_bin[0], z_bin[1], f)
            gals = cat[sel]
            # Get World position of each galaxy
            pos = np.vstack((gals['ALPHA_J2000'], gals['DELTA_J2000'])).T
            # Calculate Pixel position of each galaxy
            pos = w.wcs_world2pix(pos, 0).astype(int)
            pos = np.flip(pos, axis=1)  # Need to invert the columns
            # Pixels where at least one galaxy has been found
            pos_unique = np.unique(pos, axis=0)
            # Scan over the populated pixels
            for count, pix in enumerate(pos_unique):
                # Select galaxies in range of pixels
                sel = pos[:, 0] == pix[0]
                sel = (pos[:, 1] == pix[1])*sel
                weight = gals[sel]['weight']
                weights_mask[tuple(pix)] = np.sum(weight)

            # Get final mask
            weights_mask = mask*weights_mask

            # Save to file the mask
            name = 'MASK_{}_Z{}'.format(f, n_z_bin+1)
            warning = mask_file[f].write(
                weights_mask, name, header=hd, type='image') or warning

            print('----> Created mask for field {} and '
                  'bin {}'.format(f, n_z_bin+1))
            sys.stdout.flush()

            # Generate plots
            if want_plots:
                plt.imshow(weights_mask, interpolation='nearest')
                plt.colorbar()
                plt.savefig('{}/mask_{}_z{}.pdf'
                            ''.format(plots_file.path, f, n_z_bin+1))
                plt.close()

        mask_file[f].print_info()

    return warning


# ------------------- Function to calculate the multiplicative correction ----#

def run_mult(paths, fields, z_bins, n_avg_m, want_plots):

    print('Running MULT_CORR module')
    sys.stdout.flush()
    warning = False

    # Input files
    cat_full_file = paths['cat_full']
    mask_file = {f: paths['mask_'+f] for f in fields}
    mult_file = {f: paths['mult_'+f] for f in fields}
    if want_plots:
        plots_file = paths['plots']

    # Read galaxy catalogue
    tabname = 'data'
    try:
        cat = cat_full_file.read_key(tabname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping '
              'calculation!'.format(tabname, cat_full_file.path))
        sys.stdout.flush()
        return True

    # Check that the table has the correct columns
    table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'm', 'weight', 'id', 'Z_B',
                  'MASK', 'star_flag']
    for key in table_keys:
        if key not in cat.columns.names:
            print('WARNING: No key {} in table of {}. Skipping '
                  'calculation!'.format(key, cat_full_file.path))
            sys.stdout.flush()
            return True

    # First loop: scan over the fields
    for f in fields:

        # Check existence of files and in case skip
        if mult_file[f].exists:
            keys = mult_file[f].get_keys()
            mzs = all(['MULT_CORR_{}_Z{}'.format(f, x+1) in keys
                      for x in range(len(z_bins))])
        if mult_file[f].exists and mzs:
            print('----> Skipping MULT_CORR calculation for field {}. Output '
                  'file already there!'.format(f))
            sys.stdout.flush()
            continue

        # Remove old output file to avoid confusion
        try:
            mult_file[f].remove()
        except FileNotFoundError:
            pass

        # Second loop: divide galaxies in redshift bins
        for n_z_bin, z_bin in enumerate(z_bins):

            print('Calculating multiplicative correction for field'
                  ' {} and bin {}:'.format(f, n_z_bin+1))
            sys.stdout.flush()

            # Read mask and create WCS object
            imname = 'MASK_{}_Z{}'.format(f, n_z_bin+1)
            try:
                mask = mask_file[f].read_key(imname)
                hd = mask_file[f].get_header(imname)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(imname, mask_file[f].path))
                sys.stdout.flush()
                return True

            # Create a new WCS object
            w = wcs.WCS(hd)
            # Create an empty array for the multiplicative correction
            mult_corr = np.zeros(mask.shape)
            # Filter galaxies
            filter = setts.filter_galaxies(cat, z_bin[0], z_bin[1], f)
            gals = cat[filter]
            # Get World position of each galaxy
            pos = np.vstack((gals['ALPHA_J2000'], gals['DELTA_J2000'])).T
            # Calculate Pixel position of each galaxy
            pos = w.wcs_world2pix(pos, 0).astype(int)
            pos = np.flip(pos, axis=1)  # Need to invert the columns
            # Pixels where at least one galaxy has been found
            pos_unique = np.unique(pos, axis=0)

            # Scan over the populated pixels
            for count, pix in enumerate(pos_unique):
                # Calculate range of pixels to average
                if pix[0] - n_avg_m < 0:
                    s1 = 0
                elif pix[0] + n_avg_m >= mult_corr.shape[0]:
                    s1 = mult_corr.shape[0]-(2*n_avg_m + 1)
                else:
                    s1 = pix[0] - n_avg_m
                if pix[1] - n_avg_m < 0:
                    s2 = 0
                elif pix[1] + n_avg_m >= mult_corr.shape[1]:
                    s2 = mult_corr.shape[1] - (2*n_avg_m + 1)
                else:
                    s2 = pix[1] - n_avg_m
                # Select galaxies in range of pixels
                sel = pos[:, 0] >= s1
                sel = (pos[:, 0] < s1 + 2*n_avg_m + 1)*sel
                sel = (pos[:, 1] >= s2)*sel
                sel = (pos[:, 1] < s2 + 2*n_avg_m + 1)*sel
                m = gals[sel]['m']
                weight = gals[sel]['weight']
                mult_corr[tuple(pix)] = np.average(m, weights=weight)

                # Print message every some step
                if (count + 1) % 1e3 == 0:
                    print('----> Done {0:5.1%} of the pixels ({1:d})'
                          ''.format(float(count+1)/len(pos_unique),
                                    len(pos_unique)))
                    sys.stdout.flush()

            # Save to file the map
            name = 'MULT_CORR_{}_Z{}'.format(f, n_z_bin+1)
            warning = mult_file[f].write(mult_corr, name,
                                         header=hd, type='image') or warning

            # Generate plots
            if want_plots:
                plt.imshow(mult_corr, interpolation='nearest')
                plt.colorbar()
                plt.savefig('{}/mult_corr_{}_z{}.pdf'
                            ''.format(plots_file.path, f, n_z_bin+1))
                plt.close()

        mult_file[f].print_info()

    return warning


# ------------------- Function to calculate the photo_z ----------------------#

def run_photo_z(paths, fields, z_bins, size_pix, want_plots):

    print('Running PHOTO_Z module')
    sys.stdout.flush()
    warning = False

    # Input files
    cat_full_file = paths['cat_full']
    photo_z_file = paths['photo_z']
    mask_file = {f: paths['mask_'+f] for f in fields}
    mult_file = {f: paths['mult_'+f] for f in fields}
    if want_plots:
        plots_file = paths['plots']

    # Remove old output file to avoid confusion
    try:
        photo_z_file.remove()
    except FileNotFoundError:
        pass

    # Read mask
    mask = {}
    w = {}
    for f in fields:
        # Read mask and create WCS object
        imname = 'MASK_NOW_{}'.format(f)
        try:
            mask[f] = mask_file[f].read_key(imname)
            hd = mask_file[f].get_header(imname)
        except KeyError:
            print('WARNING: No key {} in {}. Skipping '
                  'calculation!'.format(imname, mask_file[f].path))
            sys.stdout.flush()
            return True
        # Create a new WCS object
        w[f] = wcs.WCS(hd)

    # Read galaxy catalogue
    tabname = 'data'
    imname = 'pz_full'
    try:
        cat = cat_full_file.read_key(tabname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping '
              'calculation!'.format(tabname, cat_full_file.path))
        sys.stdout.flush()
        return True
    try:
        pz_full = cat_full_file.read_key(imname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping '
              'calculation!'.format(imname, cat_full_file.path))
        sys.stdout.flush()
        return True

    # Check that the table has the correct columns
    table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'm', 'weight', 'id', 'Z_B',
                  'MASK', 'star_flag']
    for key in table_keys:
        if key not in cat.columns.names:
            print('WARNING: No key {} in table of {}. Skipping '
                  'calculation!'.format(key, cat_full_file.path))
            sys.stdout.flush()
            return True

    # Read multiplicative corrections
    m = {}
    for f in fields:
        m[f] = {}
        for n_z_bin, z_bin in enumerate(z_bins):
            imname = 'MULT_CORR_{}_Z{}'.format(f, n_z_bin+1)
            try:
                m[f][n_z_bin] = mult_file[f].read_key(imname)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(imname, mult_file[f].path))
                sys.stdout.flush()
                return True

    # Create filters for each bin and field
    filter = {}
    for f in fields:
        filter[f] = {}
        for n_z_bin, z_bin in enumerate(z_bins):
            filt = setts.filter_galaxies(cat, z_bin[0], z_bin[1], f)
            pix = np.transpose(
                [cat[filt]['ALPHA_J2000'], cat[filt]['DELTA_J2000']])
            pix = w[f].wcs_world2pix(pix, 0).astype(int)
            masked = np.where(
                np.array([mask[f][iy, ix] for ix, iy in pix]) <= 0)[0]
            filt[filt][masked] = False
            filter[f][n_z_bin] = filt
    # Print progress message
    print('----> Created filters!')
    sys.stdout.flush()

    # Correct ellipticities
    m_corr = np.zeros(len(cat))
    for f in fields:
        for n_z_bin, z_bin in enumerate(z_bins):
            filt = filter[f][n_z_bin]
            pix = np.transpose(
                [cat[filt]['ALPHA_J2000'], cat[filt]['DELTA_J2000']])
            pix = w[f].wcs_world2pix(pix, 0).astype(int)
            m_corr[filt] = np.array([m[f][n_z_bin][iy, ix] for ix, iy in pix])
    cat['e1'] = cat['e1']/(1+m_corr)
    cat['e2'] = (cat['e2']-cat['c2'])/(1+m_corr)
    # Print progress message
    print('----> Corrected ellipticities!')
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
    area = np.zeros(len(z_bins))
    sigma_g = np.zeros(len(z_bins))
    photo_z = np.zeros((len(z_bins)+1, len(pz_full[0])))
    photo_z[0] = (np.arange(len(pz_full[0]))+1./2.)*setts.dz_cfhtlens
    n_eff_f = np.zeros((len(fields), len(z_bins)))
    area_f = np.zeros((len(fields), len(z_bins)))
    sigma_g_f = np.zeros((len(fields), len(z_bins)))
    photo_z_f = np.zeros((len(fields), len(z_bins) + 1, len(pz_full[0])))
    for count in range(len(fields)):
        photo_z_f[count, 0] = \
            (np.arange(len(pz_full[0])) + 1./2.)*setts.dz_cfhtlens

    # First loop: scan over redshift bins
    for n_z_bin, z_bin in enumerate(z_bins):

        # Merge filters
        sel = np.zeros(len(cat), dtype=bool)
        for f in fields:
            sel += filter[f][n_z_bin]
        # Filter galaxies
        gals = cat[sel]
        pz_z = pz_full[sel]
        # Get n_eff
        area[n_z_bin] = get_area(fields)
        n_eff[n_z_bin] = get_n_eff(gals, area[n_z_bin])
        # Get sigma_g
        sigma_g[n_z_bin] = get_sigma_g(gals)
        # Get photo_z
        photo_z[n_z_bin+1] = np.average(pz_z, weights=gals['weight'], axis=0)

        # Second loop: scan over fields
        for count, f in enumerate(fields):
            # Filter galaxies
            gals = cat[filter[f][n_z_bin]]
            pz_z = pz_full[filter[f][n_z_bin]]
            # Get n_eff
            area_f[count, n_z_bin] = get_area([f])
            n_eff_f[count, n_z_bin] = get_n_eff(gals, area_f[count, n_z_bin])
            # Get sigma_g
            sigma_g_f[count, n_z_bin] = get_sigma_g(gals)
            # Get photo_z
            photo_z_f[count, n_z_bin+1] = \
                np.average(pz_z, weights=gals['weight'], axis=0)

        # Print progress message
        print('----> Completed bin {}'.format(n_z_bin+1))
        sys.stdout.flush()

    # Save to file the results
    warning = photo_z_file.write(photo_z, 'PHOTO_Z', type='image') or warning
    warning = photo_z_file.write(n_eff, 'N_EFF', type='image') or warning
    warning = photo_z_file.write(sigma_g, 'SIGMA_G', type='image') or warning
    warning = photo_z_file.write(
        photo_z_f, 'PHOTO_Z_PF', type='image') or warning
    warning = photo_z_file.write(n_eff_f, 'N_EFF_PF', type='image') or warning
    warning = photo_z_file.write(
        sigma_g_f, 'SIGMA_G_PF', type='image') or warning
    warning = photo_z_file.write(area, 'AREA', type='image') or warning
    warning = photo_z_file.write(area_f, 'AREA_PF', type='image') or warning

    # Generate plots
    if want_plots:
        x = photo_z[0]
        for count in range(1, len(photo_z)):
            y = photo_z[count]
            plt.plot(x, y, label='Bin ' + str(count))
        plt.xlim(0., 2.)
        plt.xlabel('$z$', fontsize=14)
        plt.ylabel('Probability distribution', fontsize=14)
        plt.title('Photo-z')
        plt.legend(loc="upper right", frameon=False, fontsize=9,
                   labelspacing=0.01)
        plt.savefig('{}/photo_z.pdf'.format(plots_file.path))
        plt.close()
        for n_f, f in enumerate(fields):
            x = photo_z_f[n_f, 0]
            for count in range(1, photo_z_f.shape[1]):
                y = photo_z_f[n_f, count]
                plt.plot(x, y, label='Bin ' + str(count))
            plt.xlim(0., 2.)
            plt.xlabel('$z$', fontsize=14)
            plt.ylabel('Probability distribution', fontsize=14)
            plt.title('Photo-z {}'.format(f))
            plt.legend(loc="upper right", frameon=False, fontsize=9,
                       labelspacing=0.01)
            plt.savefig('{}/photo_z_{}.pdf'.format(plots_file.path, f))
            plt.close()

    photo_z_file.print_info()

    return warning


# ------------------- Function to calculate the clean catalogue --------------#

def run_cat(paths, fields, z_bins):

    print('Running CATALOGUE module')
    sys.stdout.flush()
    warning = False

    # Input files
    cat_full_file = paths['cat_full']
    cat_file = {f: paths['cat_'+f] for f in fields}
    mult_file = {f: paths['mult_'+f] for f in fields}

    # Read galaxy catalogue
    tabname = 'data'
    try:
        cat = cat_full_file.read_key(tabname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping '
              'calculation!'.format(tabname, cat_full_file.path))
        sys.stdout.flush()
        return True

    # Check that the table has the correct columns
    table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'c2', 'weight',
                  'id', 'Z_B', 'MASK', 'star_flag']
    for key in table_keys:
        if key not in cat.columns.names:
            print('WARNING: No key {} in table of {}. Skipping '
                  'calculation!'.format(key, cat_full_file.path))
            sys.stdout.flush()
            return True

    # First loop: scan over the fields
    for f in fields:

        # Check existence of files and in case skip
        if cat_file[f].exists:
            keys = cat_file[f].get_keys()
            mzs = all(['CAT_{}_Z{}'.format(f, x+1) in keys
                      for x in range(len(z_bins))])
        if cat_file[f].exists and mzs:
            print('----> Skipping CATALOGUE calculation for field {}. Output '
                  'file already there!'.format(f))
            sys.stdout.flush()
            continue

        # Remove old output file to avoid confusion
        try:
            cat_file[f].remove()
        except FileNotFoundError:
            pass

        # Second loop: divide galaxies in redshift bins
        for n_z_bin, z_bin in enumerate(z_bins):

            print('Calculating catalogue for field {} and bin {}:'
                  ''.format(f, n_z_bin+1))
            sys.stdout.flush()

            # Read multiplicative corrections
            imname = 'MULT_CORR_{}_Z{}'.format(f, n_z_bin+1)
            try:
                m = mult_file[f].read_key(imname)
                hd = mult_file[f].get_header(imname)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(imname, mult_file[f].path))
                sys.stdout.flush()
                return True
            # Create a new WCS object
            w = wcs.WCS(hd)

            # Filter galaxies
            filter = setts.filter_galaxies(cat, z_bin[0], z_bin[1], f)
            gals = cat[filter]

            # Calculate corrected ellipticities
            def find_m_correction(gal):
                pix = w.wcs_world2pix(
                    [[gal['ALPHA_J2000'], gal['DELTA_J2000']]], 0)
                pix = tuple(np.flip(pix.astype(int), axis=1)[0])
                return m[pix]

            m_corr = np.array([find_m_correction(gal) for gal in gals])
            # Ellipticities
            gals['e1'] = gals['e1']/(1+m_corr)
            gals['e2'] = (gals['e2']-gals['c2'])/(1+m_corr)

            # Create Table and save it
            table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']
            columns = []
            for key in table_keys:
                columns.append(
                    fits.Column(name=key, array=gals[key], format='E'))
            name = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
            gals = fits.BinTableHDU.from_columns(columns, name=name)
            warning = cat_file[f].write(gals, name, type='table') or warning

        cat_file[f].print_info()

    return warning


# ------------------- Function to calculate the map --------------------------#

def run_map(paths, fields, z_bins, want_plots):

    print('Running MAP module')
    sys.stdout.flush()
    warning = False

    # Input files
    cat_file = {f: paths['cat_'+f] for f in fields}
    map_file = {f: paths['map_'+f] for f in fields}
    mask_file = {f: paths['mask_'+f] for f in fields}
    if want_plots:
        plots_file = paths['plots']

    # First loop: scan over the fields
    for f in fields:

        # Check existence of files and in case skip
        if map_file[f].exists:
            keys = map_file[f].get_keys()
            mzs1 = all(['CAT_{}_Z{}_G1'.format(f, x+1) in keys
                       for x in range(len(z_bins))])
            mzs2 = all(['CAT_{}_Z{}_G2'.format(f, x+1) in keys
                       for x in range(len(z_bins))])
        if map_file[f].exists and mzs1 and mzs2:
            print('----> Skipping MAP calculation for field {}. Output '
                  'file already there!'.format(f))
            sys.stdout.flush()
            continue

        # Remove old output file to avoid confusion
        try:
            map_file[f].remove()
        except FileNotFoundError:
            pass

        # Second loop: divide galaxies in redshift bins
        for n_z_bin, z_bin in enumerate(z_bins):

            print('Calculating map for field {} and bin {}:'
                  ''.format(f, n_z_bin+1))
            sys.stdout.flush()

            # Read mask and create WCS object
            imname = 'MASK_{}_Z{}'.format(f, n_z_bin+1)
            try:
                mask = mask_file[f].read_key(imname)
                hd = mask_file[f].get_header(imname)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(imname, mask_file[f].path))
                sys.stdout.flush()
                return True
            # Create a new WCS object
            w = wcs.WCS(hd)

            # Read galaxy catalogue
            tabname = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
            try:
                cat = cat_file[f].read_key(tabname)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(tabname, cat_file[f]))
                sys.stdout.flush()
                return True

            # Check that the table has the correct columns
            table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']
            for key in table_keys:
                if key not in cat.columns.names:
                    print('WARNING: No key {} in table of {}. Skipping '
                          'calculation!'.format(key, cat_file[f]))
                    sys.stdout.flush()
                    return True

            # Get map
            map, _ = get_map(w, mask, cat)

            # Save to file the map
            name = 'MAP_{}_Z{}_G1'.format(f, n_z_bin+1)
            warning = map_file[f].write(map[0], name, header=hd,
                                        type='image') or warning
            name = 'MAP_{}_Z{}_G2'.format(f, n_z_bin+1)
            warning = map_file[f].write(map[1], name, header=hd,
                                        type='image') or warning

            # Generate plots
            if want_plots:
                plt.imshow(map[0], interpolation='nearest')
                plt.colorbar()
                plt.savefig('{}/map_{}_z{}_g1.pdf'
                            ''.format(plots_file.path, f, n_z_bin+1))
                plt.close()
                plt.imshow(map[1], interpolation='nearest')
                plt.colorbar()
                plt.savefig('{}/map_{}_z{}_g2.pdf'
                            ''.format(plots_file.path, f, n_z_bin+1))
                plt.close()

        map_file[f].print_info()

    return warning


# ------------------- Function to calculate the cl ---------------------------#

def run_cl(paths, fields, z_bins, bp, n_sims_noise, decouple_cls, want_plots):

    print('Running CL module')
    sys.stdout.flush()
    warning = False

    # Input files
    cl_file = {f: paths['cl_'+f] for f in fields}
    mask_file = {f: paths['mask_'+f] for f in fields}
    cat_file = {f: paths['cat_'+f] for f in fields}
    mcm_file = paths['mcm']
    if want_plots:
        plots_file = paths['plots']

    # First loop: scan over the fields
    for f in fields:

        # Check existence of files and in case skip
        if cl_file[f].exists:
            keys = cl_file[f].get_keys()
            mzs = all([x in keys for x in ['ELL_'+f, 'CL_'+f, 'CL_NOISE_'+f]])
        if cl_file[f].exists and mzs:
            print('----> Skipping CL calculation for field {}. Output '
                  'file already there!'.format(f))
            sys.stdout.flush()
            continue

        print('Calculating cl for field {}:'.format(f))
        sys.stdout.flush()

        # Remove old output file to avoid confusion
        try:
            cl_file[f].remove()
        except FileNotFoundError:
            pass

        # Read mask
        t = 'MASK_{}_Z1'.format(f)
        try:
            # Header is the same for each bin
            hd = mask_file[f].get_header(t)
        except KeyError:
            print('WARNING: No key {} in {}. Skipping '
                  'calculation!'.format(t, mask_file[f].path))
            sys.stdout.flush()
            return True
        w = wcs.WCS(hd)
        mask = np.zeros((len(z_bins), hd['NAXIS2'], hd['NAXIS1']))
        for n_z_bin, z_bin in enumerate(z_bins):
            t = 'MASK_{}_Z{}'.format(f, n_z_bin+1)
            try:
                mask[n_z_bin] = mask_file[f].read_key(t)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(t, mask_file[f].path))
                sys.stdout.flush()
                return True

        # Read galaxy catalogue
        cat = {}
        for n_z_bin, z_bin in enumerate(z_bins):
            t = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
            try:
                cat[n_z_bin] = cat_file[f].read_key(t)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping '
                      'calculation!'.format(t, cat_file[f].path))
                sys.stdout.flush()
                return True

            # Check that the table has the correct columns
            table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']
            for key in table_keys:
                if key not in cat[n_z_bin].columns.names:
                    print('WARNING: No key {} in table of {}. Skipping '
                          'calculation!'.format(key, cat_file[f].path))
                    sys.stdout.flush()
                    return True

        # Get maps
        map = np.zeros((len(z_bins), 2, hd['NAXIS2'], hd['NAXIS1']))
        pos = {}
        for n_z_bin, z_bin in enumerate(z_bins):
            map[n_z_bin], pos[n_z_bin] = \
                get_map(w, mask[n_z_bin], cat[n_z_bin])

        # Get Cl's
        map = np.transpose(map, axes=(1, 0, 2, 3))
        ell, cl, mcm_paths = \
            get_cl(f, bp, hd, mask, map, decouple_cls,
                   tmp_path=mcm_file.path)

        # Save to file the map
        warning = cl_file[f].write(ell, 'ELL_{}'.format(f),
                                   type='image') or warning
        warning = cl_file[f].write(cl, 'CL_{}'.format(f),
                                   type='image') or warning

        # Get noise
        noise_sims = np.zeros((n_sims_noise,) + cl.shape)
        for ns in range(n_sims_noise):
            map = np.zeros((len(z_bins), 2, hd['NAXIS2'], hd['NAXIS1']))
            # Generate random ellipticities
            for n_z_bin, z_bin in enumerate(z_bins):
                cat_sim = cat[n_z_bin].copy()
                n_gals = len(cat_sim)
                phi = 2*np.pi*np.random.rand(n_gals)
                cos = np.cos(2*phi)
                sin = np.sin(2*phi)
                cat_sim['e1'] = cat_sim['e1']*cos-cat_sim['e2']*sin
                cat_sim['e2'] = cat_sim['e1']*sin+cat_sim['e2']*cos
                # Get map
                map[n_z_bin], _ = get_map(w, mask[n_z_bin], cat_sim,
                                          pos_in=pos[n_z_bin])
            # Print message every some step
            if (ns+1) % 1e2 == 0:
                print('----> Done {0:5.1%} of the noise Cls ({1:d})'
                      ''.format(float(ns+1)/n_sims_noise, n_sims_noise))
                sys.stdout.flush()
            # Get Cl's
            map = np.transpose(map, axes=(1, 0, 2, 3))
            _, noise_sims[ns], _ = \
                get_cl(f, bp, hd, mask, map, decouple_cls,
                       tmp_path=mcm_file.path)

        # Get mean shape noise
        noise = np.mean(noise_sims, axis=0)

        # Save to file the map
        name = 'CL_NOISE_{}'.format(f)
        warning = cl_file[f].write(noise, name, type='image') or warning

        # Generate plots
        if want_plots:
            factor = 1.
            x = ell
            for nb1 in range(len(z_bins)):
                for nb2 in range(nb1, len(z_bins)):
                    ax = plt.gca()
                    for ng1 in range(2):
                        for ng2 in range(ng1, 2):
                            y = factor*(cl[ng1, ng2, :, nb1, nb2]
                                        - noise[ng1, ng2, :, nb1, nb2])
                            color = next(ax._get_lines.prop_cycler)['color']
                            plt.plot(x, y, 'o', label='$C_l^{{{}{}}}$'
                                     ''.format(ng1+1, ng2+1), color=color)
                            plt.plot(x, -y, '*', color=color)
                    plt.title('Z = {}{}'.format(nb1+1, nb2+1))
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('$\\ell$')
                    plt.ylabel('$C_\\ell$')
                    plt.legend(loc='best')
                    plt.savefig('{}/cl_{}_z{}{}.pdf'
                                ''.format(plots_file.path, f, nb1+1, nb2+1))
                    plt.close()

        cl_file[f].print_info()

    return warning


# ------------------- Function to calculate the cat_sims ---------------------#

def run_cat_sims(paths, fields, z_bins, n_sims_cov):

    print('Running CAT_SIMS module')
    sys.stdout.flush()
    warning = False

    # Input files
    cat_sims_file = paths['cat_sims']
    cat_file = {f: paths['cat_'+f] for f in fields}
    mask_file = {f: paths['mask_'+f] for f in fields}

    # Define local variables
    join = os.path.join
    n_bins = len(z_bins)
    sign_e2 = -1
    noise_factor = 1
    seed_ini = 1000

    # Define main function
    def get_gaussian_sim(ww, masks, cl_mat, catals, ipixs, seed, fname, field):
        n_bins = len(catals)
        if (n_bins != len(ipixs)) or (n_bins != len(cl_mat)/2):
            print("SHIT")
            exit(1)
        nmaps = 2*n_bins
        clarr = []
        sparr = 2*np.ones(n_bins, dtype=int)
        for i1 in range(nmaps):
            for i2 in range(i1, nmaps):
                clarr.append(cl_mat[i1, i2])
        clarr = np.array(clarr)

        ny, nx = masks[0].shape
        lx = np.fabs(np.radians(nx*ww.wcs.cdelt[0]))
        ly = np.fabs(np.radians(ny*ww.wcs.cdelt[1]))
        print('lx = {}'.format(lx))  # REMOVE
        print('ly = {}'.format(ly))  # REMOVE

        maps = nmt.synfast_flat(nx, ny, lx, ly, clarr, sparr,
                                seed=seed).reshape([n_bins, 2, ny, nx])

        cats = []
        cats.append(fits.PrimaryHDU())

        for i_c, c in enumerate(catals):
            # Get randomly rotated ellipticities
            phi = 2*np.pi*np.random.rand(len(c))
            cs = np.cos(2*phi)
            sn = np.sin(2*phi)
            e1n = c['e1']*cs - c['e2']*sn
            e2n = c['e1']*sn + c['e2']*cs
            # Add signal and noise
            e1 = maps[i_c, 0][ipixs[i_c][1], ipixs[i_c][0]] + noise_factor*e1n
            e2 = sign_e2*(
                maps[i_c, 1][ipixs[i_c][1], ipixs[i_c][0]] + noise_factor*e2n)

            cats.append(fits.BinTableHDU.from_columns([
                fits.Column(name='ALPHA_J2000', format='E',
                            array=c['ALPHA_J2000']),
                fits.Column(name='DELTA_J2000', format='E',
                            array=c['DELTA_J2000']),
                fits.Column(name='e1', format='E', array=e1),
                fits.Column(name='e2', format='E', array=e2),
                fits.Column(name='weight', format='E', array=c['weight'])],
                name='CAT_{}_Z{}'.format(f, i_c+1)))
        hdul = fits.HDUList(cats)
        hdul.writeto(fname, overwrite=True)
        return

    # Create reference theory Cl
    # TODO_EB: for now I am just importing the preexisting ones
    bf = io.FitsFile(path=join(paths['raw_data'].path, 'cls_bf.fits'))
    ells = bf.read_key('ELL')
    cl_ee = bf.read_key('CLS_BF')
    cl_matrix = np.zeros([2*n_bins, 2*n_bins, len(ells)])
    cl0 = np.zeros(len(ells))
    for i in range(n_bins):
        for j in range(n_bins):
            cl_matrix[2*i, 2*j, :] = cl_ee[:, i, j]  # EE
            cl_matrix[2*i, 2*j+1, :] = cl0  # EB
            cl_matrix[2*i+1, 2*j, :] = cl0  # BE
            cl_matrix[2*i+1, 2*j+1, :] = cl0  # BB

    # Scan over the fields
    for f in fields:

        print('Calculating gaussian simulations for field {}:'.format(f))
        sys.stdout.flush()

        # Remove old output file to avoid confusion
        for ns in range(n_sims_cov):
            try:
                os.remove(join(
                    cat_sims_file.path, 'sim_{}_cat_{}.fits'.format(ns, f)))
            except FileNotFoundError:
                pass

        # Read masks
        imnames = ['MASK_{}_Z{}'.format(f, x+1) for x in range(n_bins)]
        try:
            masks = [mask_file[f].read_key(x) for x in imnames]
            hds = [mask_file[f].get_header(x) for x in imnames]
        except KeyError:
            print('WARNING: At least one of these keys {} is missing {}. '
                  'Skipping calculation!'.format(imnames, mask_file[f].path))
            sys.stdout.flush()
            return True

        # Create a new WCS object
        w = wcs.WCS(hds[0])

        # Read catalogues
        tabnames = ['CAT_{}_Z{}'.format(f, x+1) for x in range(n_bins)]
        try:
            cats = [cat_file[f].read_key(x) for x in tabnames]
        except KeyError:
            print('WARNING: At least one of these keys {} is missing in {}. '
                  'Skipping calculation!'.format(tabnames, cat_file[f].path))
            sys.stdout.flush()
            return True

        # Check that the table has the correct columns
        table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2', 'weight']
        for cat in cats:
            for key in table_keys:
                if key not in cat.columns.names:
                    print('WARNING: No key {} in table of {}. Skipping '
                          'calculation!'.format(key, cat_file[f].path))
                    sys.stdout.flush()
                    return True

        # Transform to pixel positions
        ipixs = [np.transpose((w.wcs_world2pix(
            np.transpose([c['ALPHA_J2000'], c['DELTA_J2000']]),
            0, ra_dec_order=True)).astype(int)) for c in cats]

        # Generate simulations for each field
        for i in range(n_sims_cov):
            seed = seed_ini + i
            fname = \
                join(cat_sims_file.path, 'sim_{}_cat_{}.fits'.format(seed, f))
            get_gaussian_sim(w, masks, cl_matrix, cats, ipixs, seed, fname, f)
            # Print message every some step
            if (i+1) % 1e2 == 0:
                print('----> Calculated {0:5.1%} of the simulation catalogues '
                      '({1:d}) for field {2}'.format(float(i+1)/n_sims_cov,
                                                     n_sims_cov, f))
                sys.stdout.flush()

    return warning


# ------------------- Function to calculate the cl_sims ----------------------#

def run_cl_sims(paths, fields, z_bins, bp, decouple_cls):

    print('Running CL_SIMS module')
    sys.stdout.flush()
    warning = False
    join = os.path.join

    # Input files
    cat_sims_file = paths['cat_sims']
    cl_sims_file = {f: paths['cl_sims_'+f] for f in fields}
    mask_file = {f: paths['mask_'+f] for f in fields}
    mcm_file = paths['mcm']

    # First loop: scan over the fields
    for nf, f in enumerate(fields):

        # Check existence of files and in case skip
        if cl_sims_file[f].exists:
            keys = cl_sims_file[f].get_keys()
            mzs = all([x in keys for x in ['ELL_'+f, 'CL_SIM_'+f]])
        if cl_sims_file[f].exists and mzs:
            print('----> Skipping CL_SIMS calculation for field {}. Output '
                  'file already there!'.format(f))
            sys.stdout.flush()
            continue

        print('Calculating Cls from simulations for field {}:'.format(f))
        sys.stdout.flush()

        # Remove old output file to avoid confusion
        try:
            cl_sims_file[f].remove()
        except FileNotFoundError:
            pass

        # List files present in the sims directory
        list_sims = [join(cat_sims_file.path, x) for x
                     in os.listdir(cat_sims_file.path)
                     if re.match('.+{}.fits'.format(f), x)]
        list_sims = sorted(list_sims)
        # Initialize array with Cl's
        ns = len(list_sims)
        nl = len(bp)
        nb = len(z_bins)
        nm = 2
        cl = np.zeros((ns, nm, nm, nl, nb, nb))

        # Read mask
        t = 'MASK_{}_Z1'.format(f)
        try:
            # Header is the same for each bin
            hd = mask_file[f].get_header(t)
        except KeyError:
            print('WARNING: No key {} in {}. Skipping calculation!'
                  ''.format(t, mask_file[f].path))
            sys.stdout.flush()
            return True
        w = wcs.WCS(hd)
        mask = np.zeros((len(z_bins), hd['NAXIS2'], hd['NAXIS1']))
        for n_z_bin, z_bin in enumerate(z_bins):
            t = 'MASK_{}_Z{}'.format(f, n_z_bin+1)
            try:
                mask[n_z_bin] = mask_file[f].read_key(t)
            except KeyError:
                print('WARNING: No key {} in {}. Skipping calculation!'
                      ''.format(t, mask_file[f].path))
                sys.stdout.flush()
                return True

        # Calculate Cl's for each simulation
        for ns, fname in enumerate(list_sims):
            single_cat = io.FitsFile(path=fname)
            # Read galaxy catalogue
            cat = {}
            for n_z_bin, z_bin in enumerate(z_bins):
                t = 'CAT_{}_Z{}'.format(f, n_z_bin+1)
                try:
                    cat[n_z_bin] = single_cat.read_key(t)
                except KeyError:
                    print('WARNING: No key {} in {}. Skipping calculation!'
                          ''.format(t, fname))
                    sys.stdout.flush()
                    return True

                # Check that the table has the correct columns
                table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'e1', 'e2',
                              'weight']
                for key in table_keys:
                    if key not in cat[n_z_bin].columns.names:
                        print('WARNING: No key {} in {}. Skipping calculation!'
                              ''.format(key, fname))
                        sys.stdout.flush()
                        return True

            # Get maps
            map = np.zeros((len(z_bins), 2, hd['NAXIS2'], hd['NAXIS1']))
            pos = {}
            for n_z_bin, z_bin in enumerate(z_bins):
                map[n_z_bin], pos[n_z_bin] = \
                    get_map(w, mask[n_z_bin], cat[n_z_bin])

            # Get Cl's
            map = np.transpose(map, axes=(1, 0, 2, 3))
            ell, cl[ns], mcm_paths = \
                get_cl(f, bp, hd, mask, map, decouple_cls,
                       tmp_path=mcm_file.path)

            # Print message every some step
            if (ns+1) % 1e2 == 0:
                print('----> Done {0:5.1%} of the simulations ({1:d})'
                      ''.format(float(ns+1)/len(list_sims), len(list_sims)))
                sys.stdout.flush()

        # Save to file the map
        name = 'ELL_{}'.format(f)
        warning = cl_sims_file[f].write(ell, name, type='image') or warning
        name = 'CL_SIM_{}'.format(f)
        warning = cl_sims_file[f].write(cl, name, type='image') or warning

        cl_sims_file[f].print_info()

    return warning


# ------------------- Function to unify all ----------------------------------#

def run_final(paths, fields, z_bins, bp):

    print('Running Final module')
    sys.stdout.flush()
    warning = False

    # Input files
    data_file = paths['data']
    photo_z_file = paths['photo_z']
    cl_file = {f: paths['cl_'+f] for f in fields}
    cl_sims_file = {f: paths['cl_sims_'+f] for f in fields}

    # Remove old output file to avoid confusion
    try:
        data_file.remove()
    except FileNotFoundError:
        pass

    # Read cl and cl noise
    cl = []
    cl_n = []
    for nf, f in enumerate(fields):
        imname = 'ELL_{}'.format(f)
        try:
            ell = cl_file[f].read_key(imname)
        except KeyError:
            print('WARNING: No key {} in {}. Skipping calculation!'
                  ''.format(imname, cl_file[f].path))
            sys.stdout.flush()
            return True
        imname = 'CL_{}'.format(f)
        try:
            cl.append(cl_file[f].read_key(imname))
        except KeyError:
            print('WARNING: No key {} in {}. Skipping calculation!'
                  ''.format(imname, cl_file[f].path))
            sys.stdout.flush()
            return True
        imname = 'CL_NOISE_{}'.format(f)
        try:
            cl_n.append(cl_file[f].read_key(imname))
        except KeyError:
            print('WARNING: No key {} in {}. Skipping calculation!'
                  ''.format(imname, cl_file[f].path))
            sys.stdout.flush()
            return True
    # Write ells
    warning = data_file.write(ell, 'ELL', type='image') or warning
    # Write cl
    cl = np.array(cl)
    warning = data_file.write(cl[:, 0, 0], 'CL_EE', type='image') or warning
    warning = data_file.write(cl[:, 0, 1], 'CL_EB', type='image') or warning
    warning = data_file.write(cl[:, 1, 1], 'CL_BB', type='image') or warning
    # Write cl noise
    cl_n = np.array(cl_n)
    warning = data_file.write(cl_n[:, 0, 0], 'CL_EE_NOISE',
                              type='image') or warning
    warning = data_file.write(cl_n[:, 0, 1], 'CL_EB_NOISE',
                              type='image') or warning
    warning = data_file.write(cl_n[:, 1, 1], 'CL_BB_NOISE',
                              type='image') or warning

    # Read cl sims
    cl = []
    for nf, f in enumerate(fields):
        imname = 'CL_SIM_{}'.format(f)
        try:
            cl.append(cl_sims_file[f].read_key(imname))
        except KeyError:
            print('WARNING: No key {} in {}. Skipping calculation!'
                  ''.format(imname, cl_sims_file[f].path))
            sys.stdout.flush()
            return True
    # Write cl sims
    cl = np.array(cl)
    warning = data_file.write(cl[:, :, 0, 0], 'CL_SIM_EE',
                              type='image') or warning
    warning = data_file.write(cl[:, :, 0, 1], 'CL_SIM_EB',
                              type='image') or warning
    warning = data_file.write(cl[:, :, 1, 1], 'CL_SIM_BB',
                              type='image') or warning

    # Read and write photo_z
    imname = 'PHOTO_Z'
    try:
        image = photo_z_file.read_key(imname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping calculation!'
              ''.format(imname, photo_z_file.path))
        sys.stdout.flush()
        return True
    warning = data_file.write(image, imname, type='image') or warning

    # Read and write n_eff
    imname = 'N_EFF'
    try:
        image = photo_z_file.read_key(imname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping calculation!'
              ''.format(imname, photo_z_file.path))
        sys.stdout.flush()
        return True
    warning = data_file.write(image, imname, type='image') or warning

    # Read and write sigma_g
    imname = 'SIGMA_G'
    try:
        image = photo_z_file.read_key(imname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping calculation!'
              ''.format(imname, photo_z_file.path))
        sys.stdout.flush()
        return True
    warning = data_file.write(image, imname, type='image') or warning

    data_file.print_info()

    return warning


# ------------------- Get map ------------------------------------------------#

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
        pos = np.vstack(
            (cat['ALPHA_J2000'], cat['DELTA_J2000'])).T
        # Calculate Pixel position of each galaxy
        pos = w.wcs_world2pix(pos, 0).astype(int)
        pos = np.flip(pos, axis=1)  # Need to invert the columns
    else:
        pos = pos_in.copy()

    # Perform lex sort and get the sorted indices
    sorted_idx = np.lexsort(pos.T)
    sorted_pos = pos[sorted_idx, :]
    # Differentiation along rows for sorted array
    diff_pos = np.diff(sorted_pos, axis=0)
    diff_pos = np.append([True], np.any(diff_pos != 0, 1), 0)
    # Get unique sorted labels
    sorted_labels = diff_pos.cumsum(0)-1
    # Get labels
    labels = np.zeros_like(sorted_idx)
    labels[sorted_idx] = sorted_labels
    # Get unique indices
    unq_idx = sorted_idx[diff_pos]
    # Get unique pos's and ellipticities
    pos_unique = pos[unq_idx, :]
    w_at_pos = np.bincount(labels, weights=cat['weight'])
    g1_at_pos = np.bincount(labels, weights=cat['e1']*cat['weight'])/w_at_pos
    g2_at_pos = np.bincount(labels, weights=cat['e2']*cat['weight'])/w_at_pos
    # Create the maps
    map_1[pos_unique[:, 0], pos_unique[:, 1]] = g1_at_pos
    map_2[pos_unique[:, 0], pos_unique[:, 1]] = g2_at_pos

    # empty = 1.-np.array(
    #     [mask[tuple(x)] for x in pos_unique]).sum()/mask.flatten().sum()
    # print('----> Empty pixels: {0:5.2%}'.format(empty))
    # sys.stdout.flush()

    return np.array([map_1, map_2]), pos


# ------------------- Get cls ------------------------------------------------#

def get_cl(field, bp, hd, mask, map, decouple_cls, tmp_path=None):
    """ Generate cl's from a mask and a map.

    Args:
        field: field.
        bp: bandpowers for ell.
        hd: header with infos about the mask and maps.
        mask: array with mask.
        decouple_cls: decouple cls with mcm
        map: maps for each bin and polarization.

    Returns:
        cl: array with cl (E/B, ell, bins).
        mcm_path: path to the mode coupling matrix.

    """

    # Initialize Cls
    n_bins = map.shape[1]
    n_ells = len(bp)
    cl = np.zeros((2, 2, n_ells, n_bins, n_bins))

    # Dimensions
    Nx = hd['NAXIS1']
    Ny = hd['NAXIS2']
    Lx = Nx*abs(hd['CDELT1'])*np.pi/180  # Mask dimension in radians
    Ly = Ny*abs(hd['CDELT2'])*np.pi/180  # Mask dimension in radians

    # Fields definition
    fd = np.array([nmt.NmtFieldFlat(
        Lx, Ly, mask[x], [map[0, x], -map[1, x]]) for x in range(n_bins)])
    # Bins for flat sky fields
    b = nmt.NmtBinFlat(bp[:, 0], bp[:, 1])
    # Effective ells
    ell = b.get_effective_ells()

    # Iterate over redshift bins to compute Cl's
    mcm_paths = []
    for nb1 in range(n_bins):
        for nb2 in range(nb1, n_bins):
            # Temporary path for mode coupling matrix
            if tmp_path is None:
                mcm_p = os.path.expanduser('~')
            else:
                mcm_p = tmp_path
            mcm_p = mcm_p+'/mcm_{}_Z{}{}.dat'.format(field, nb1+1, nb2+1)
            mcm_paths.append(mcm_p)
            # Define workspace for mode coupling matrix
            wf = nmt.NmtWorkspaceFlat()
            try:
                wf.read_from(mcm_p)
            except RuntimeError:
                wf.compute_coupling_matrix(fd[nb1], fd[nb2], b)
                wf.write_to(mcm_p)
                print('Calculated mode coupling matrix for bins {}{}'
                      ''.format(nb1+1, nb2+1))
                sys.stdout.flush()
            # Calculate Cl's
            cl_c = nmt.compute_coupled_cell_flat(fd[nb1], fd[nb2], b)
            if decouple_cls is True:
                cl_d = wf.decouple_cell(cl_c)
            else:
                cl_d = cl_c
            cl_d = np.reshape(cl_d, (2, 2, n_ells))
            cl[:, :, :, nb1, nb2] = cl_d
            cl[:, :, :, nb2, nb1] = cl_d

    return ell, cl, mcm_paths
