"""

This module contains all the samplers implemented.

Functions:
 - run_emcee()
 - run_fisher()
 - run_single_point()

"""

import sys
import numpy as np
import emcee
import likelihood as lkl
import cosmo as cosmo_tools


# ------------------- emcee ---------------------------------------------------#

def run_emcee(args, cosmo, data, settings, path):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        file with chains.

    """

    # Local variables
    full = cosmo['params']
    mask = cosmo['mask']
    obs = data['corr_obs']
    icov = data['inv_cov_mat']
    ns = settings['n_steps']
    nw = settings['n_walkers']
    nt = settings['n_threads']
    nd = len(mask[mask])

    #Print useful stuff
    print 'Starting the chains!'
    for key in settings.keys():
        print key + ' = ' + str(settings[key])
    sys.stdout.flush()

    # pardd = np.array([[0.61128301, 0.12959581, 0.03298559, 2.33950179, 1.17948114],
    #     [0.62179704, 0.08651784, 0.03176021, 3.1725831, 1.21425305],
    #     [0.61387389, 0.12716369, 0.03295057, 2.30825387, 1.2787784],
    #     [0.63933645, 0.07419231, 0.03268715, 3.76112382, 0.80378302],
    #     [0.61411828, 0.1198132, 0.0326422, 2.54119524, 1.13263974],
    #     [0.61292237, 0.11211861, 0.0326977, 2.43247255, 1.25343547],
    #     [0.61169319, 0.1179573, 0.03275583, 2.40587882, 1.22732628],
    #     [0.61223887, 0.12022914, 0.03277267, 2.3724876, 1.2836256],
    #     [0.61205213, 0.12068099, 0.03282129, 2.50119757, 1.18081108],
    #     [0.61215958, 0.11286177, 0.03278035, 2.56833138, 1.24804087],
    #     [0.6129472, 0.1115795, 0.03281857, 2.55147159, 1.26568931],
    #     [0.61571051, 0.1116597, 0.0329496, 2.58508717, 1.22223027],
    #     [0.61763372, 0.11725815, 0.02923001, 2.77332373, 0.89240313],
    #     [0.62128665, 0.12058442, 0.0322747, 2.44978128, 1.17813878],
    #     [0.61264369, 0.10465044, 0.03261911, 2.83914941, 1.143915],
    #     [0.61171242, 0.10290258, 0.03264447, 2.7879307, 1.24346498],
    #     [0.61394416, 0.12374908, 0.03228716, 2.31440055, 1.29006757],
    #     [0.61328645, 0.0911836, 0.03291535, 3.20588058, 1.08550888],
    #     [0.6153054, 0.11576915, 0.03249731, 2.56060304, 1.21832985],
    #     [0.62512858, 0.07580462, 0.03280936, 3.46367013, 1.22909326],
    #     [0.61301924, 0.11891865, 0.03295068, 2.43908244, 1.19616036],
    #     [0.63011881, 0.13329137, 0.03195289, 2.63679745, 0.96030607],
    #     [0.61127603, 0.09835023, 0.03247089, 2.83294951, 1.26192743],
    #     [0.64862911, 0.02020534, 0.02904658, 4.95076794, 1.28636231],
    #     [0.61247009, 0.06869975, 0.03217391, 3.56957904, 1.21753416],
    #     [0.61157204, 0.11079607, 0.03268598, 2.56069047, 1.27348838],
    #     [0.61323221, 0.13065968, 0.03268495, 2.45645662, 1.17275215],
    #     [0.63383016, 0.09955236, 0.03055703, 2.96953964, 1.15794649],
    #     [0.61675077, 0.11793387, 0.03270023, 2.56763535, 1.23498297],
    #     [0.61121919, 0.08109625, 0.03267563, 3.34680312, 1.28764626],
    #     [0.61070881, 0.1203262, 0.03267456, 2.43235623, 1.27751646],
    #     [0.61455332, 0.12286326, 0.03243348, 2.31554197, 1.2528238],
    #     [0.6179589, 0.08639524, 0.03185354, 3.25803913, 1.18779354],
    #     [0.61158236, 0.08543936, 0.03269354, 3.17705304, 1.28075016],
    #     [0.61281566, 0.11259656, 0.03251761, 2.7449084, 1.10337015],
    #     [0.61258616, 0.1267327, 0.03264357, 2.32720987, 1.23363464],
    #     [0.61535752, 0.11449351, 0.03279954, 2.50336177, 1.27482777],
    #     [0.62992803, 0.06459925, 0.03113823, 3.77504683, 1.1218812],
    #     [0.61352897, 0.11415567, 0.03276948, 2.59239712, 1.26000178],
    #     [0.63382981, 0.07774867, 0.03268166, 3.55977153, 0.91425536],
    #     [0.61750238, 0.10372658, 0.03119846, 2.33176623, 1.29965226],
    #     [0.6140671, 0.11513705, 0.03284596, 2.53599469, 1.22397995],
    #     [0.61465499, 0.12025004, 0.03249156, 2.59106119, 1.10161997],
    #     [0.61274573, 0.10608437, 0.03270542, 2.41765244, 1.28589483],
    #     [0.61522662, 0.11014689, 0.03193737, 2.51474397, 1.21811812],
    #     [0.61379514, 0.1178444, 0.03280236, 2.50343511, 1.21403934],
    #     [0.61331762, 0.11811586, 0.03266404, 2.48424027, 1.24613599],
    #     [0.61390262, 0.1138219, 0.03110468, 2.71025167, 1.05103847],
    #     [0.62087046, 0.12021376, 0.03229997, 2.45192166, 1.18184842],
    #     [0.61472052, 0.1089155, 0.03297873, 2.85446083, 0.96266382],
    #     [0.61916172, 0.12550445, 0.03246847, 2.48292743, 1.10767924],
    #     [0.61004751, 0.10863307, 0.03287404, 2.63871255, 1.2547043],
    #     [0.61198867, 0.11369022, 0.03294623, 2.49714929, 1.25096708],
    #     [0.61684015, 0.09877948, 0.03260872, 2.8957037, 1.17946169],
    #     [0.61938487, 0.11685056, 0.03211736, 2.61887448, 1.1654817],
    #     [0.63078175, 0.05837585, 0.03273001, 3.90217814, 1.21263366],
    #     [0.61316953, 0.1183214, 0.03288168, 2.44423408, 1.19765542],
    #     [0.62598223, 0.12832491, 0.03173652, 2.65553485, 0.98345096],
    #     [0.61142058, 0.10187813, 0.03252611, 2.77661714, 1.25017056],
    #     [0.61223341, 0.11883085, 0.03211565, 2.52299761, 1.18598409],
    #     [0.62470167, 0.08003713, 0.03124839, 3.32898282, 1.24451487],
    #     [0.61330053, 0.03847789, 0.03171518, 4.18897555, 1.21961175],
    #     [0.61159434, 0.1135446, 0.032741, 2.50634777, 1.27900973],
    #     [0.61372298, 0.12626185, 0.03269675, 2.47784008, 1.18293996],
    #     [0.62736807, 0.10453152, 0.03132701, 2.85233851, 1.16944357],
    #     [0.61835206, 0.11742769, 0.03264977, 2.62406409, 1.21739925],
    #     [0.61094351, 0.09849793, 0.03252256, 2.94387763, 1.23062626],
    #     [0.6202024, 0.07038012, 0.03169517, 3.59784027, 1.21791763],
    #     [0.61026318, 0.07688392, 0.03288638, 3.42410397, 1.29362317]])
    #
    # for count, p in enumerate(pardd):
    #     loglkl = lkl.lnprob(p, full, mask, data, settings)
    #     sigma8 = cosmo_tools.get_sigma_8(p, full, mask)
    #     print 'Done {} points over {}. loglkl={}, sigma8={}'.format(count+1,len(pardd),loglkl,sigma8)
    # exit(1)


    #Initialize sampler
    sampler = emcee.EnsembleSampler(nw, nd, lkl.lnprob, args=[full, mask, data, settings], threads=nt)


    if args.restart:
        # Initial point from data
        vars_0 = np.loadtxt(path['output'],unpack=True)
        vars_0 = vars_0[2:2+nd]
        vars_0 = vars_0[:,-nw:].T
    else:
        # Initial point
        vars_0 = np.array([lkl.get_random(full[mask], 1.e3) for x in range(nw)])
        # Create file
        f = open(path['output'], 'w')
        f.close()

    for count, result in enumerate(sampler.sample(vars_0, iterations=ns, storechain=False)):
#        print 'Here!'
        pos = result[0]
        prob = result[1]
#        print 'There!'
        f = open(path['output'], 'a')
        for k in range(pos.shape[0]):
            out = np.append(np.array([1., -prob[k]]), pos[k])
            out = np.append(out, cosmo_tools.get_sigma_8(pos[k], full, mask))
            f.write('    '.join(['{0:.10e}'.format(x) for x in out]) + '\n')
        f.close()
        print '----> Computed ' + '{0:5.1%}'.format(float(count+1) / ns) + ' of the steps'
        sys.stdout.flush()

    return


# ------------------- fisher --------------------------------------------------#

def run_fisher(cosmo, data, settings, path):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        file with chains.

    """
    return


# ------------------- single_point --------------------------------------------#

def run_single_point(cosmo, data, settings):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        output in terminal likelihood.

    """

    # Local variables
    full = cosmo['params']
    mask = cosmo['mask']
    obs = data['corr_obs']
    icov = data['inv_cov_mat']

    var = full[:,1][mask]
    post = lkl.lnprob(var, full, mask, data, settings)
    sigma8 = cosmo_tools.get_sigma_8(var, full, mask)

    print 'Cosmological parameters:'
    print '----> h             = ' + '{0:2.4e}'.format(full[0,1])
    print '----> Omega_c h^2   = ' + '{0:2.4e}'.format(full[1,1])
    print '----> Omega_b h^2   = ' + '{0:2.4e}'.format(full[2,1])
    print '----> ln(10^10 A_s) = ' + '{0:2.4e}'.format(full[3,1])
    print '----> n_s           = ' + '{0:2.4e}'.format(full[4,1])
    print 'Derived parameters:'
    print '----> sigma_8       = ' + '{0:2.4e}'.format(sigma8)
    print 'Likelihood:'
    print '----> -ln(like)     = ' + '{0:4.4f}'.format(-post)

    return
