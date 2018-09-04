"""

This module contains the main function to prepare data in
fourier space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import numpy as np
from astropy.io import fits
from astropy import wcs
import io
import matplotlib
import matplotlib.pyplot as plt

def prep_fourier(args):
    """ Prepare data in fourier space.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to a fits file file the output.

    """

    dim_orig = 1 #Original pixels dimension in arcsec
    dim_new = 120 #New pixels dimension in arcsec
    dim_ratio = int(np.round(float(dim_new)/float(dim_orig)))


    # Define absolute paths and check the existence of each required file
    path = {}
    for count in range(4):
        field = 'W'+str(count+1)
        path['input'] = io.path_exists_or_error(args.input_folder)
        path['mask_'+field] = io.path_exists_or_error(path['input']+'/'+field+'.16bit.small.reg2.fits.gz')
    path['output'] = io.path_exists_or_error(args.input_folder)

    def plot_and_save(table, fname):
        plt.imshow(table,interpolation='nearest')
        plt.colorbar()
        plt.savefig(fname)
        plt.close()
        return


    for count in range(4):

        field = 'W'+str(count+1)
        print 'Field ' + field + ':'

        # Read and plot original data
        data = io.read_from_fits(path['mask_'+field], 'primary').astype(int)
        print '----> Original shape: ', data.shape

        # Convert to boolean and plot
        data = (1-np.array(data, dtype=bool)).astype(bool)

        # Calculate new dimensions
        div1, mod1 = np.divmod(data.shape[0], dim_ratio)
        div2, mod2 = np.divmod(data.shape[1], dim_ratio)
        if mod1 == 0:
            x1 = div1
        else:
            x1 = div1+1
        if mod2 == 0:
            x2 = div2
        else:
            x2 = div2+1
        start1 = int(np.round((x1*dim_ratio-data.shape[0])/2.))
        start2 = int(np.round((x2*dim_ratio-data.shape[1])/2.))
        end1 = start1+data.shape[0]
        end2 = start2+data.shape[1]

        # Add borders to array and plot
        data_ext = np.zeros((x1*dim_ratio, x2*dim_ratio), dtype=bool)
        data_ext[start1:end1,start2:end2] = data
        plot_and_save(data_ext, path['output']+'/'+field+'_orig.pdf')
        print '----> Extended shape: ', data_ext.shape

        # Calculate new array and plot
        data_new = np.zeros((x1, x2))
        for count1 in range(x1):
            for count2 in range(x2):
                s1 = count1*dim_ratio
                s2 = count2*dim_ratio
                new_pixel = data_ext[s1:s1+dim_ratio,s2:s2+dim_ratio].astype(float)
                data_new[count1,count2] = np.average(new_pixel)
        plot_and_save(data_new, path['output']+'/'+field+'_new.pdf')
        print '----> New shape: ', (x1, x2)
        add1 = float(x1)*dim_ratio/data.shape[0]-1.
        add2 = float(x2)*dim_ratio/data.shape[1]-1.
        print '----> Pixels added: '+'({0:5.2%}, '.format(add1)+'{0:5.2%})'.format(add2)



        header = io.read_header_from_fits(path['mask_'+field], 'primary')
        # Create a new WCS object
        w = wcs.WCS(naxis=2)
        # Write header
        w.wcs.crpix = np.array([start1+header['CRPIX1'], start2+header['CRPIX2']])/dim_ratio
        w.wcs.cdelt = np.array([header['CD1_1'], header['CD2_2']])*dim_ratio
        w.wcs.crval = np.array([header['CRVAL1'], header['CRVAL2']])
        w.wcs.ctype = [header['CTYPE1'], header['CTYPE2']]
        header = w.to_header()
        # Write to file
        fname = path['input']+'/'+field+'_mask.fits'
        io.write_to_fits(fname, data_new, field+'_mask', header=header)


    print 'Success!!'
