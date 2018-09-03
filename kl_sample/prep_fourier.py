"""

This module contains the main function to prepare data in
fourier space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import numpy as np
from astropy.io import fits
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

    arcsec_orig = 1 #Pixels dimension in arcsec
    arcsec_new = 120 #Pixels dimension in arcsec
    arcsec_ratio = int(np.round(float(arcsec_new)/float(arcsec_orig)))


    # Define absolute paths and check the existence of each required file
    path = {}
    for count in range(4):
        field = 'W'+str(count+1)
        path['mask_'+field] = io.path_exists_or_error(args.input_folder +field+'.16bit.small.reg2.fits.gz')
    path['output'] = io.path_exists_or_error(args.input_folder)


    for count in range(4):

        field = 'W'+str(count+1)
        # Read
        data = io.read_from_fits(path['mask_'+field], 'primary').astype(int)

        # Convert to boolean
        data_bool = (1-np.array(data, dtype=bool)).astype(int)

        # Decrease pixels
        div1, mod1 = np.divmod(data_bool.shape[0], arcsec_ratio)
        div2, mod2 = np.divmod(data_bool.shape[1], arcsec_ratio)
        if mod1 == 0:
            x1 = div1
        else:
            x1 = div1+1
        if mod2 == 0:
            x2 = div2
        else:
            x2 = div2+1

        start1 = int(np.round((x1*arcsec_ratio-data_bool.shape[0])/2.))
        start2 = int(np.round((x2*arcsec_ratio-data_bool.shape[1])/2.))
        end1 = start1+data_bool.shape[0]
        end2 = start2+data_bool.shape[1]

        data_bool_ext = np.zeros((x1*arcsec_ratio, x2*arcsec_ratio)).astype(int)
        data_bool_ext[start1:end1,start2:end2] = data_bool


        data_new = np.zeros((x1, x2))

        for count1 in range(x1):
            for count2 in range(x1):
                start1 = count1*arcsec_ratio
                start2 = count2*arcsec_ratio
                new_pixel = data_bool_ext[start1:start1+arcsec_ratio,start2:start2+arcsec_ratio]
                data_new[count1,count2] = np.average(new_pixel)

        print 'Field ' + field + ':'
        print '----> Original shape: ', data_bool.shape
        print '----> Extended shape: ', data_bool_ext.shape
        print '----> New shape: ', (x1, x2)
        add1 = float(x1)*arcsec_ratio/data_bool.shape[0]-1.
        add2 = float(x2)*arcsec_ratio/data_bool.shape[1]-1.
        print '----> Pixels added: '+'({0:5.2%}, '.format(add1)+'{0:5.2%})'.format(add2)



        plt.imshow(data)
        plt.colorbar()
        plt.savefig(path['output']+'/'+field+'_full.pdf')
        plt.close()

        plt.imshow(data_bool)
        plt.colorbar()
        plt.savefig(path['output']+'/'+field+'_bool.pdf')
        plt.close()

        plt.imshow(data_bool_ext)
        plt.colorbar()
        plt.savefig(path['output']+'/'+field+'_bool_ext.pdf')
        plt.close()

        plt.imshow(data_new)
        plt.colorbar()
        plt.savefig(path['output']+'/'+field+'_new.pdf')
        plt.close()

    print 'Success!!'
