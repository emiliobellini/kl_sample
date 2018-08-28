"""

This module contains the main function to prepare data in
fourier space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import numpy as np
from astropy.io import fits
import io

def prep_fourier(args):
    """ Prepare data in fourier space.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to a fits file file the output.

    """


    # Define absolute paths and check the existence of each required file
    path = {
        'mask_w4'    : io.path_exists_or_error(args.input_folder + 'W4.16bit.small.reg2.fits'),
    }


    # Read
    w4data = io.read_from_fits(path['mask_w4'], 'primary')
    w4hd = io.read_header_from_fits(path['mask_w4'], 'primary')
    io.print_info_fits('/home/emilio/Codes/kl_sample/data/data_real.fits')

    pz =  io.read_from_fits('/home/emilio/Codes/kl_sample/data/data_real.fits', 'photo_z')
    print [(pz[0]*pz[x]).sum() for x in range(1,8)]

    # print w4data.shape
    # for key in w4hd.keys():
    #     print key, w4hd[key]
    tot = 0
    totx, toty = w4data.shape
    # for nx in range(totx):
    #     for ny in range(toty):
    #         if w4data[nx,ny]<1:
    #             tot+=1
    # print tot
    # print tot/(60.**4.)
