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
import shutil
import urllib
import kl_sample.io as io
import kl_sample.settings as setts
from astropy import wcs


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
    fields = setts.fields_cfhtlens  # List of CFHTlens fields
    default = setts.default_params  # Dictionary with of default parameters
    # Number of simulations used to calculate the covariance matrix
    n_sims_cov = ini.read_param('n_sims_cov', 'settings', default=default)
    # List of redshift bins
    z_bins = ini.read_param('z_bins', 'settings', default=default)
    z_bins = np.vstack((z_bins[:-1], z_bins[1:])).T
    # Size pixels masks in arcsecs (it has to be an integer number)
    size_pix = int(ini.read_param('size_pix', 'settings', default=default))

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
            raise IOError('Ini files are different! Skipping the calculation!')
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
        is_run['pz'] = True
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
    cat_file = paths['cat_full']
    mask_file = {f: paths['mask_'+f] for f in fields}
    mask_sec_file = {f: paths['mask_sec_'+f] for f in fields}
    mask_url_file = paths['mask_url']
    badfields_file = paths['badfields']
    if want_plots:
        plots_file = paths['plots']

    # Read galaxy catalogue
    tabname = 'data'
    try:
        cat_file.read_key(tabname)
    except KeyError:
        print('WARNING: No key {} in {}. Skipping '
              'calculation!'.format(tabname, cat_file.path))
        sys.stdout.flush()
        return True
    cat = cat_file.content[tabname]

    # Check that the table has the correct columns
    table_keys = ['ALPHA_J2000', 'DELTA_J2000', 'id']
    for key in table_keys:
        if key not in cat.columns.names:
            print('WARNING: No key {} in table of {}. Skipping '
                  'calculation!'.format(key, cat_file.path))
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
            mask_sec_file[f].read_key(imname, dtype=np.uint16)
        except KeyError:
            print('WARNING: No key {} in {}. Skipping '
                  'calculation!'.format(imname, mask_sec_file[f].path))
            sys.stdout.flush()
            return True
        mask_sec = mask_sec_file[f].content[imname]
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
            mask_bad_file.read_key(imname, dtype=np.uint16)
            mask_bad = mask_bad_file.content[imname]
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
            sel = setts.filter_galaxies(cat, z_bin[0], z_bin[1], field=f)
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
