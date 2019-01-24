import numpy as np
import healpy as hp
import scipy.linalg as lng
import subprocess
import time
import sys
import os


try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf


### Write Class Parameter File
###---------------------------------------
def write_class_param_file (file_name, cosmo, lmaxs=lmaxs, lmaxt=lmaxt, lmax_lss=lmax_lss, zpk_array=zpk_array,
        shells=shells, output_dir=output_dir, out_prefix=out_prefix, k_max_hinvMpc=k_max_hinvMpc) :

    if output_dir[-1] != '/' :
        output_dir += '/'

    firstline = '### CLASS Parameter File \n'

    with open(file_name, 'w') as f :
        f.write(firstline)
        f.write('h = '+str(cosmo['HUBBLE_H100'])+'\n')
        f.write('T_cmb = '+str(cosmo['T_CMB'])+'\n')
        f.write('Omega_b = '+str(cosmo['OMEGA_B'])+'\n')
        f.write('N_ur = '+str(cosmo['N_MASSLESS_NU'])+'\n')
        f.write('Omega_k = '+str(cosmo['OMEGA_K'])+'\n')
        if cosmo['W_DARK_ENERGY'] == -1 :
            f.write('Omega_fld = 0\n')
        else :
            f.write('Omega_fld = '+str(cosmo['OMEGA_DE'])+'\n')
            f.write('w0_fld = '+str(cosmo['W_DARK_ENERGY'])+'\n')
            f.write('wa_fld = 0\n')
            f.write('cs2_fld = 1\n')
        f.write('Omega_cdm = '+str(cosmo['OMEGA_M']-cosmo['OMEGA_B'])+'\n')
        f.write('tau_reio = '+str(cosmo['TAU_REION'])+'\n')
        f.write('YHe = '+str(cosmo['HE_FRACTION'])+'\n')
        f.write('A_s = '+str(cosmo['SCALAR_AMPLITUDE'])+'\n')
        f.write('k_pivot = '+str(cosmo['K_PIVOT'])+'\n')
        f.write('N_ncdm = '+str(cosmo['N_NONCOLD_DM'])+'\n')
        f.write('m_ncdm = '+str(cosmo['M_NCDM'])+'\n')
        f.write('T_ncdm = '+str(cosmo['T_NCDM'])+'\n')
        f.write('n_s = '+str(cosmo['N_S'])+'\n')
        f.write('alpha_s = '+str(cosmo['N_T_RUNNING'])+'\n')
        f.write('r = '+str(cosmo['TENSOR_R'])+'\n')
        if cosmo['TENSOR_R'] != 0 :
            f.write('modes = s,t\n')
        else :
            f.write('modes = s\n')
        f.write('n_t = '+str(cosmo['N_T'])+'\n')
        f.write('alpha_t = 0.\n')
        f.write('l_max_scalars = '+str(lmaxs)+'\n')
        f.write('l_max_tensors = '+str(lmaxt)+'\n')
        f.write('P_k_max_h/Mpc = '+str(k_max_hinvMpc)+'\n')
        f.write('l_max_lss = '+str(lmax_lss)+'\n')
        f.write('selection = '+str(shells['window'])+'\n')
        f.write('selection_mean = '+str(shells['z'][0]))
        for i in range(1, len(shells['z'])) :
            f.write(', ')
            f.write(str(shells['z'][i]))
        f.write('\n')
        f.write('selection_width = '+str(shells['dz'][0]))
        for i in range(1, len(shells['dz'])) :
            f.write(', ')
            f.write(str(shells['dz'][i]))
        f.write('\n')
        if shells['non_diagonal'] != 0 :
            f.write('non_diagonal = '+str(shells['non_diagonal'])+'\n')
        if shells['non_linear'] != 'none' :
            f.write('non linear = '+str(shells['non_linear'])+'\n')

        if not np.isscalar(zpk_array) :
            f.write('z_pk = '+str(zpk_array[0]))
            for i in range(1, len(zpk_array)) :
                f.write(', ')
                f.write(str(zpk_array[i]))
            f.write('\n')
            f.write('output = tCl,pCl,dCl,lCl,mPk\n')
        else :
            f.write('output = tCl,pCl,dCl,lCl\n')

        f.write('root = '+output_dir+out_prefix+'\n')
        f.write('lensing = yes\n')

        f.write('headers = yes\n')
        f.write('format = '+format_)
        f.write('bessel file = no\n')

        f.write('write parameters = yes\n')
        f.write('background_verbose = 1\n')
        f.write('thermodynamics_verbose = 1\n')
        f.write('perturbations_verbose = 1\n')
        f.write('bessels_verbose = 1\n')
        f.write('transfer_verbose = 1\n')
        f.write('primordial_verbose = 1\n')
        f.write('spectra_verbose = 1\n')
        f.write('nonlinear_verbose = 1\n')
        f.write('lensing_verbose = 1\n')
        f.write('output_verbose = 1\n')

        if write_background :
            f.write('write background = yes\n')
        else :
            f.write('write background = no\n')

    f.close()


def write_alms(filename,alms,out_dtype=None,lmax=-1,mmax=-1,mmax_in=-1):
    """Write alms to a fits file.
    In the fits file the alms are written
    with explicit index scheme, index = l*l + l + m +1, possibly out of order.
    By default write_alm makes a table with the same precision as the alms.
    If specified, the lmax and mmax parameters truncate the input data to
    include only alms for which l <= lmax and m <= mmax.
    Parameters
    ----------
    filename : str
      The filename of the output fits file
    alms : array, complex or list of arrays
      A complex ndarray holding the alms, index = m*(2*lmax+1-m)/2+l, see Alm.getidx
    lmax : int, optional
      The maximum l in the output file
    mmax : int, optional
      The maximum m in the output file
    out_dtype : data type, optional
      data type in the output file (must be a numpy dtype). Default: *alms*.real.dtype
    mmax_in : int, optional
      maximum m in the input array
    """

    alms = np.array(alms)
    if len(alms.shape) > 1 :
        npol = alms.shape[0]
    else :
        npol = 1

    if mmax_in != -1 :
        l2max = hp.Alm.getlmax(alms.shape[-1], mmax=mmax_in)
    else :
        l2max = hp.Alm.getlmax(alms.shape[-1])

    if (lmax != -1 and lmax > l2max):
        raise ValueError("Too big lmax in parameter")
    elif lmax == -1:
        lmax = l2max

    if mmax_in == -1:
        mmax_in = l2max

    if mmax == -1:
        mmax = lmax
    if mmax > mmax_in:
        mmax = mmax_in

    if (out_dtype == None):
        out_dtype = alms.real.dtype

    l,m = hp.Alm.getlm(lmax)
    idx = np.where((l <= lmax)*(m <= mmax))
    l = l[idx]
    m = m[idx]

    idx_in_original = hp.Alm.getidx(l2max, l=l, m=m)

    index = l**2 + l + m + 1

    hdulist = pf.HDUList()

    for ilm in npol :

        if npol > 1 :
            alm = alms[ilm]
        else :
            alm = np.copy(alms)

        out_data = np.empty(len(index),
                   dtype=[('INDEX','i'),
                          ('REAL',out_dtype),
                          ('IMAGINARY',out_dtype)])
        out_data['INDEX'] = index
        out_data['REAL'] = alm.real[idx_in_original]
        out_data['IMAGINARY'] = alm.imag[idx_in_original]

        cindex = pf.Column(name="INDEX", format=hp.fitsfunc.getformat(np.int32), unit="l*l+l+m+1",                      array=out_data['INDEX'])
        creal = pf.Column(name="REAL", format=hp.fitsfunc.getformat(out_dtype), unit="unknown", array=out_data['REAL'])
        cimag = pf.Column(name="IMAGINARY", format=hp.fitsfunc.getformat(out_dtype), unit="unknown",                    array=out_data['IMAGINARY'])

        cols = pf.ColDefs([cindex,creal,cimag])
        tbhdu = pf.BinTableHDU.from_columns(cols)

        #tbhdu = pf.new_table([cindex,creal,cimag])
        hdulist.append(tbhdu)

    hdulist.writeto(filename, clobber=True)




def main(argv=None):

    if argv is None:

        args = len(sys.argv)

        print ('\n')
        print ('###################################')
        print ('##                               ##')
        print ('##   CMB with LOGNORMAL SHELLS   ##')
        print ('##                               ##')
        print ('###################################')

        if args < 2 :
            print ('\n Enter an Configuration File Name')
            print (' Syntax : cmb_lens.py \"config_file.ini\" ')
            print (' Quitting!!')
            sys.exit()

        confile = sys.argv[1]

        start = time.time()

        outfold = 'LensingOUT/'
        cmbulmap = 'cmb_unlensed_map.fits'
        cmbmap = 'cmb_map.fits'
        lensmap = 'potential_map.fits'
        cmbalm = 'cmb_unlensed_alm.fits'
        lensalm = 'potential_alm.fits'
        nside = 256
        l_max = 2*nside
        lmax = l_max+nside//2
        rand_sd = 1
        npol = 3
        class_root = ''
        matter_shells = True
        n_low = 4
        n_medium = 6
        n_high = 10
        n_diag = 0
        n_bin = 0
        non_linear = 'none'
        beam = 0.
        lenspix_root = ''
        alm_out = True
        map_out = False
        precision = 1e-3

        cosmo = {
            'T_CMB' : 2.72548,
            'HUBBLE_H100' : 0.6766,
            'OMEGA_M' : 0.3089,
            'OMEGA_B' : 0.0485976,
            'OMEGA_K' : 0.,
            'OMEGA_NU' : 0.,
            'OMEGA_DE' : 0.,
            'SIGMA_8' : 0.8159,
            'N_S' : 0.9667,
            'N_S_RUNNING' : 0,
            'N_T' : 0,
            'N_T_RUNNING' : 0,
            'TENSOR_R' : 0.003,
            'TAU_REION' : 0.055,
            'HE_FRACTION' : 0.2453,
            'N_MASSLESS_NU' : 2.0328,
            'N_NONCOLD_DM' : 1,
            'M_NCDM' : 0.06,
            'T_NCDM' : 0.71611,
            'W_DARK_ENERGY' : -1,
            'K_PIVOT_SCALAR' : 0.002,
            'SCALAR_AMPLITUDE' : 2.142E-9
            }


        print ('\n Reading \''+confile+'\' and Parsing...')
        sys.stdout.flush()

        with open(confile, 'r') as f:
            for line in f.readlines() :
                li = line.lstrip()
                if not li.startswith("#") and '=' in li :
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    value = value.split(',')
                    value = np.array(value)
                    if key == 'outFold' :
                        outfold = value[0]
                        if outfold[-1] != '/' :
                            outfold += '/'
                    elif key == 'nSide' :
                        value = value.astype(np.int)
                        nside = value
                    elif key == 'lmaxDens' :
                        value = value.astype(np.int)
                        l_max = value[0]
                    elif key == 'lmaxLens' :
                        value = value.astype(np.int)
                        lmax = value[0]
                    elif key == 'seed' :
                        value = value.astype(np.int)
                        rand_sd = value[0]
                    elif key == 'pol' :
                        if value[0] == 'no' :
                            npol = 1
                    elif key == 'rootCLASS' :
                        class_root = value[0]
                        if class_root[-1] != '/' :
                            class_root += '/'
                    elif key == 'doMatterShells' :
                        if value[0] == 'no' :
                            matter_shells = False
                    elif key == 'low_redshift_nbins' :
                        value = value.astype(np.int)
                        n_low = value[0]
                    elif key == 'medium_redshift_nbins' :
                        value = value.astype(np.int)
                        n_medium = value[0]
                    elif key == 'high_redshift_nbins' :
                        value = value.astype(np.int)
                        n_high = value[0]
                    elif key == 'offDiagonal' :
                        value = value.astype(np.int)
                        n_diag = value[0]
                    elif key == 'nonLinear' :
                        non_linear = value[0]
                    elif key == 'beam' :
                        value = value.astype(np.float)
                        beam = value[0]
                    elif key == 'rootLenspix' :
                        lenspix_root = value[0]
                        if lenspix_root[-1] != '/' :
                            lenspix_root += '/'
                    elif key == 'writeMaps' :
                        if value[0] == 'no' :
                            map_out = False
                    elif key == 'writeAlms' :
                        if value[0] == 'yes' :
                            alm_out = True
                    elif key == 'svdTreshold' :
                        value = value.astype(np.float)
                        precision = value[0]
                    elif key == 'T_CMB' :
                        value = value.astype(np.float)
                        cosmo['T_CMB'] = value
                    elif key == 'HUBBLE_H100' :
                        value = value.astype(np.float)
                        cosmo['HUBBLE_H100'] = value
                    elif key == 'OMEGA_M' :
                        value = value.astype(np.float)
                        cosmo['OMEGA_M'] = value
                    elif key == 'OMEGA_B' :
                        value = value.astype(np.float)
                        cosmo['OMEGA_B'] = value
                    elif key == 'OMEGA_K' :
                        value = value.astype(np.float)
                        cosmo['OMEGA_K'] = value
                    elif key == 'SIGMA_8' :
                        value = value.astype(np.float)
                        cosmo['SIGMA_8'] = value
                    elif key == 'N_S' :
                        value = value.astype(np.float)
                        cosmo['N_S'] = value
                    elif key == 'N_S_RUNNING' :
                        value = value.astype(np.float)
                        cosmo['N_S_RUNNING'] = value
                    elif key == 'N_T' :
                        value = value.astype(np.float)
                        cosmo['N_T'] = value
                    elif key == 'N_T_RUNNING' :
                        value = value.astype(np.float)
                        cosmo['N_T_RUNNING'] = value
                    elif key == 'TENSOR_R' :
                        value = value.astype(np.float)
                        cosmo['TENSOR_R'] = value
                    elif key == 'TAU_REION' :
                        value = value.astype(np.float)
                        cosmo['TAU_REION'] = value
                    elif key == 'HE_FRACTION' :
                        value = value.astype(np.float)
                        cosmo['HE_FRACTION'] = value
                    elif key == 'N_MASSLESS_NU' :
                        value = value.astype(np.float)
                        cosmo['N_MASSLESS_NU'] = value
                    elif key == 'N_NONCOLD_DM' :
                        value = value.astype(np.float)
                        cosmo['N_NONCOLD_DM'] = value
                    elif key == 'M_NCDM' :
                        value = value.astype(np.float)
                        cosmo['M_NCDM'] = value
                    elif key == 'T_NCDM' :
                        value = value.astype(np.float)
                        cosmo['T_NCDM'] = value
                    elif key == 'W_DARK_ENERGY' :
                        value = value.astype(np.float)
                        cosmo['W_DARK_ENERGY'] = value
                    elif key == 'K_PIVOT_SCALAR' :
                        value = value.astype(np.float)
                        cosmo['K_PIVOT_SCALAR'] = value
                    elif key == 'SCALAR_AMPLITUDE' :
                        value = value.astype(np.float)
                        cosmo['SCALAR_AMPLITUDE'] = value

        cosmo['OMEGA_NU'] = cosmo['M_NCDM']*cosmo['N_NONCOLD_DM'] / 93.14063 / (cosmo['HUBBLE_H100'])**2
        cosmo['OMEGA_DE'] = 1. - cosmo['OMEGA_K'] - cosmo['OMEGA_M'] - cosmo['OMEGA_NU']

        if not os.path.exists(outfold) :
            os.mkdir(outfold)

        if l_max > lmax :
            l_max = lmax
            print ('\n MATTER L_MAX CANNOT EXCEED CMB_LMAX !!!')
            print (' Matter l_max reset to '+str(lmax))

        lens_lmax = lmax
        dens_lmax = l_max

        if matter_shells :
            n_bin = n_low + n_medium + n_high

        n_dim = 1 + npol
        dm = n_bin + n_dim

        k = 4 + 2*(npol-1)

        n_cl = k + n_bin + (n_diag+1)*(n_bin-n_diag) + n_diag*(n_diag+1)//2

        zlow = 0.01*np.power(10, 1./n_low * np.arange(n_low+1))
        zmedium = 0.1 + (.9/n_medium)*(np.arange(n_medium)+1)
        zhigh = 1+(5./n_high)*(np.arange(n_high)+1)

        zborders = np.concatenate([zlow, zmedium, zhigh])
        nz = len(zborders)-1

        z  = np.zeros(nz)
        dz = np.zeros(nz)

        binfile = outfold + 'bins.txt'
        with open(binfile, 'w') as f :
            f.write('### z         dz   \n')
            for iz in range(nz) :
                z[iz] = (zborders[iz+1] + zborders[iz]) / 2.
                dz[iz] = (zborders[iz+1] - zborders[iz]) / 2.
                f.write(str(z[iz])+'   '+str(dz[iz])+'\n')
        f.close()

        shells = {
                'window' : 'tophat',
                'z' : z,
                'dz' : dz,
                'non_diagonal' : n_diag,
                'non_linear' : non_linear
                }

        zpk_array = [0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,15,20,30,40]
        k_max_hinvMpc = 1.

        classdir = outfold+'class/'
        if not os.path.exists(classdir) :
            os.mkdir(classdir)

        class_param_file = classdir+'class.ini'
        cl_unlensed = classdir + 'class_cl.dat'
        ok = os.path.isfile(cl_unlensed)

        if ok :

            cl_in = np.loadtxt(cl_unlensed)

            ok = n_cl == cl_in.shape[-1]

        if not ok :

            class_start = time.time()

            print ('\n Running CLASS Software !!!\n')

            write_class_param_file(class_param_file, cosmo, output_dir=classdir, out_prefix='class_',
                    lmaxs=lmax+300, lmaxt=lmax+100, lmax_lss=l_max, shells=shells, zpk_array=zpk_array,
                    k_max_hinvMpc=k_max_hinvMpc, write_background=True)

            return_code = subprocess.run(class_root+"class "+class_param_file)

            if return_code == 0 :
                print("\nCLASS code run in "+str(time.time() - class_start)+" seconds !!!\n")
            else :
                print("\ncmb_lens has not been able to call the CLASS software !!!")
                print("\nCheck your installation of CLASS !!!")
                print (' Quitting!!')
                sys.exit()

        sys.stdout.flush()

        scale = 1e9
        murra = 1.

        if lmax > len(cl_in[:,0])+1 :
            lmax = len(cl_in[:,0])+1
        if l_max >= lmax :
            l_max = lmax

        px_sz = hp.nside2resol(nside)

        bl = hp.gauss_beam(beam*np.pi/180./60., lmax=lens_lmax)

        n_lm = hp.Alm.getsize(lens_lmax)
        klm = np.tile(np.array([1j*0.]),(n_lm,dm))
        means = np.zeros(n_bin)

        l = np.arange(lmax+1)
        ell=l[2::]
        bare = ell*(ell+1.)/(2.*np.pi)
        w = np.sqrt( (2.*l+1.)/4./np.pi )

        a = np.zeros((lmax-1,len(cl_in[0])))

        if lmax > len(cl_in) :
            a[0:len(cl_in),:] = np.copy(cl_in)
        else :
            a[0:lmax-1,:] = np.copy(cl_in[0:lmax-1,:])

        lb_max = lmax
        if npol == 3 :
            wh, = np.where(a[:,4] != 0.)
            if len(wh) != 0 :
                lb_max = np.max(wh) + 2

        for i in range(1,k) :
            a[0:lmax-1,i] = a[0:lmax-1,i]/bare


        print ('\n Computing power spectra of log-normal density contrast fields...')
        sys.stdout.flush()

        r = len(a[0][k:-n_bin])
        dl = np.zeros((lmax+1,r))

        ## Transformation from rho_i X rho_j to s_i X s_j (s = log(rho) is Gaussian) :
        for i in range(k,r+k) :
            a[l_max-1::,i] = 0.
            wh, = np.where(a[:,i] != 0.)
            if len(wh) != 0 :
                ip = np.max(wh)
                for jp in range(ip,lmax-11) :
                    a[jp,i] = a[ip,i]*np.cos((jp-ip)*np.pi/2./(lmax-10.-ip))**2

            cl = np.concatenate(([1.,1.],a[:,i]/bare))
            cl[1] = 2.*cl[2]-cl[3]
            if cl[1] <= 0. :
                cl[1] = cl[2]
            cl[0] = 2.*cl[1]-cl[2]
            if cl[0] <= 0. :
                cl[0] = cl[1]
            alm = 0j + cl
            xi =  0j + hp.alm2map(w*alm,nside,lmax=lmax,mmax=0,verbose=False)
            Cl = hp.map2alm(np.real(np.log(1.+xi)),lmax=lmax,mmax=0)/w
            a[0:lmax-1,i] = np.real(Cl[2:lmax+1])
            dl[:,i-k] = np.real(Cl[0:lmax+1])
            j = (i-k)/(n_diag+1)
            if (i-k)%(n_diag+1) == 0 and j <= n_bin-n_diag :
                #means[j] = -np.real(Cl[0])/2.
                means[j] = np.sum(bl[0:l_max+1]*bl[0:l_max+1]*cl[0:l_max+1]*l[0:l_max+1]*(l[0:l_max+1]+1.)/(2.*np.pi))
            i0 = k + (n_diag+1)*(n_bin-n_diag)
            if i > i0 :
                for gro in range(1,n_diag) :
                    alp = gro*n_diag - (gro-1)*gro/2
                    if i == i0 + alp :
                        j = n_bin-n_diag+gro
                        #means[j] = -np.real(Cl[0])/2.
                        means[j] = np.sum(bl[0:l_max+1]*bl[0:l_max+1]*cl[0:l_max+1]*l[0:l_max+1]*(l[0:l_max+1]+1.)/(2.*np.pi))

        ## Transformation from phi X rho_i to phi X s_i :
        komsik = k+r - n_diag*(n_diag+1)/2 -1
        incr = 0
        ix0 = k
        for i in range(k+r,len(cl_in[0])) :
            a[0:lmax-1,i] = a[0:lmax-1,i]/bare
            terim = 1.
            if ix0 >= komsik :
                incr += 1
            ix0 = k + (i-k-r)*(n_diag+1-incr)
            terim += (np.sqrt(np.e) - 1.) * a[0:lmax-1, ix0]
            a[0:lmax-1,i] = a[0:lmax-1,i]/terim

        np.random.seed(seed=rand_sd)

        rd = np.concatenate(([0,1],np.arange(l_max+1,lmax)))

        print ('\n Constructing Harmonic Coefficients for CMB, Lensing Potential, and Density Perturbation Fields...')
        sys.stdout.flush()

        for p in rd :
            #for p in range(2) :
            b = np.zeros((n_bin,n_bin))*1j
            for i in range(n_bin) :
                for j in range(i, n_bin) :
                    if i < n_bin-n_diag :
                        if j-i <= n_diag :
                            q = k + i*(n_diag+1) + j - i
                        else :
                            q = 0
                    else :
                        q0 = k + (n_bin-n_diag-1)*(n_diag+1) + n_diag + 1
                        i0 = i - (n_bin-n_diag)
                        j0 = j - (n_bin-n_diag)

                        q = q0 + j0 + i0*n_diag - i0*(i0+1)/2

                    if i != j :
                        q = 0

                    if q != 0 :
                        b[i][j] = dl[p,q-k]
                    if i != j :
                        b[j][i] = b[i][j]

            s,uh = lng.eigh(b)

            b_chk = lng.norm((uh*s).dot(uh.T)-b)/lng.norm(b)
            if b_chk > precision :
                print ('  Decomposition Failed at l = '+str(p))
            uh *= np.sqrt(s+0j)

            for jp in range(p+1) :

                idx = hp.Alm.getidx(lens_lmax, p, jp)

                x = np.random.normal(0., 1., n_bin)
                y = np.random.normal(0., 1., n_bin)

                if jp == 0 :
                    z = x + 1j*0.
                else :
                    z = (x + 1j*y)/np.sqrt(2.)
                z = uh.dot(z)

                klm[idx,n_dim::] = z*bl[p]


        for p in range(2,l_max+1) :
            b = np.zeros((dm,dm))*1j
            if p > lb_max :
                b = np.zeros((dm-1,dm-1))*1j
            for i in range(dm) :
                for j in range(i, dm) :
                    if i == 0 and j == 0 :
                        q = 1
                    elif i == 0 and j == 1:
                        q = 3
                    elif i == 1 and j == 1 :
                        q = 2
                    elif i == 2 and j == 2 and npol == 3 :
                        q = 4
                    elif i == 3 and j == 3 and npol == 3 :
                        q = 5
                    elif i == 0 and j == 3 and npol == 3 :
                        q = 6
                    elif i == 1 and j == 3 and npol == 3 :
                        q = 7
                    elif i == npol and j > i :
                        q = -n_bin+j-npol-1
                    elif i > npol :
                        if i <= npol+n_bin-n_diag :
                            if j-i <= n_diag :
                                q = k + (i-npol-1)*(n_diag+1) + j - i
                            else :
                                q = 0
                        else :
                            q0 = k + (n_bin-n_diag-1)*(n_diag+1) + n_diag + 1
                            i0 = i - (npol+n_bin-n_diag) - 1
                            j0 = j - (npol+n_bin-n_diag) - 1

                            q = q0 + j0 + i0*n_diag - i0*(i0+1)/2

                        if i != j :
                            q = 0

                    else :
                        q = 0

                    if i <= npol and j <= npol :
                        murra = scale*scale
                    elif i == npol and j > i :
                        murra = scale
                    else :
                        murra = 1.

                    inw, jnw = i, j
                    if p > lb_max :
                        if i == 2 or j == 2 :
                            continue
                        if i > 2 :
                            inw = i-1
                        if j > 2 :
                            jnw = j-1

                    if q != 0 :
                        b[inw][jnw] = a[p-2,q]*murra
                    if i != j :
                        b[jnw][inw] = b[inw][jnw]

            s,uh = lng.eigh(b)
            #u, s, v = lng.svd(b)
            #uh = u + v.T
            #uh /= 2.

            b_chk = lng.norm((uh*s).dot(uh.T)-b)/lng.norm(b)
            if b_chk > precision :
                print ('  Decomposition Failed at l = '+str(p))
            uh *= np.sqrt(s+0j)

            #uh = np.real(uh)

            #cov = np.real(b)
            #mean = np.zeros(dm)

            for jp in range(p+1) :

                idx = hp.Alm.getidx(lens_lmax, p, jp)

                x = np.random.normal(0., 1., len(b))
                y = np.random.normal(0., 1., len(b))

                #x = np.random.multivariate_normal(mean,cov)
                #y = np.random.multivariate_normal(mean,cov)

                if jp == 0 :
                    z = x + 1j*0.
                else :
                    z = (x + 1j*y)/np.sqrt(2.)
                z = uh.dot(z)

                if p <= lb_max :
                    klm[idx] = z*bl[p]
                else :
                    rd = np.concatenate(([0,1],np.arange(3,dm)))
                    klm[idx,rd] = z*bl[p]

        for p in range(l_max+1, lmax+1) :
            b = np.zeros((n_dim,n_dim))*1j
            cl_s = [0,1,2,3]
            if p > lb_max :
                b = np.zeros((n_dim-1,n_dim-1))*1j
                cl_s = [0,1,3]
            if npol == 1 :
                cl_s = [0,1]
            for i in range(n_dim) :
                for j in range(i, n_dim) :
                    if i == 0 and j == 0 :
                        q = 1
                    elif i == 0 and j == 1:
                        q = 3
                    elif i == 1 and j == 1 :
                        q = 2
                    elif i == 2 and j == 2 and npol == 3 :
                        q = 4
                    elif i == 3 and j == 3 and npol == 3 :
                        q = 5
                    elif i == 0 and j == 3 and npol == 3 :
                        q = 6
                    elif i == 1 and j == 3 and npol == 3 :
                        q = 7
                    else :
                        q = 0

                    inw, jnw = i, j
                    if p > lb_max :
                        if i == 2 or j == 2 :
                            continue
                        if i > 2 :
                            inw = i-1
                        if j > 2 :
                            jnw = j-1

                    if q != 0 :
                        b[inw][jnw] = a[p-2,q]*scale*scale
                    if i != j :
                        b[jnw][inw] = b[inw][jnw]

            s,uh = lng.eigh(b)

            b_chk = lng.norm((uh*s).dot(uh.T)-b)/lng.norm(b)
            if b_chk > precision :
                print ('  Decomposition Failed for CMB only at l = '+str(p))
            uh *= np.sqrt(s+0j)

            for jp in range(p+1) :

                idx = hp.Alm.getidx(lens_lmax, p, jp)

                x = np.random.normal(0., 1., len(b))
                y = np.random.normal(0., 1., len(b))

                if jp == 0 :
                    z = x + 1j*0.
                else :
                    z = (x + 1j*y)/np.sqrt(2.)
                z = uh.dot(z)

                klm[idx, cl_s] = z*bl[p]

        klm[:,0:n_dim] /= scale

        densfold =  outfold+'Density_shells/'
        if not os.path.exists(densfold) :
            os.mkdir(densfold)

        np.save(outfold+'all_alms.npy', klm)

        klm = np.load(outfold+'all_alms.npy')

        print ('\n Constructing CMB Maps \n')
        sys.stdout.flush()

        alm = np.ascontiguousarray(klm.T[0:npol])
        if alm_out :
            write_alms(outfold+cmbalm, alm, lmax=lens_lmax)
        if map_out :
            if npol == 3 :
                alm = (alm[0],alm[1],alm[2])
            xi = hp.alm2map(alm,nside,verbose=False)
            hp.write_map(outfold+cmbmap, xi, overwrite=True)

        print ('\n Constructing Lensing Potential Map')
        sys.stdout.flush()

        alm = np.ascontiguousarray(klm.T[npol])
        if alm_out :
            write_alms(outfold+lensalm, alm, lmax=lens_lmax)
        if map_out :
            xi = hp.alm2map(alm,nside,verbose=False)
            hp.write_map(outfold+lensmap, xi, overwrite=True)

        for i in range(n_dim, dm) :
            print ('\n Constructing Density Contrast Map '+str(i-npol))
            alm = np.ascontiguousarray(klm.T[i])
            xi = hp.alm2map(alm,nside,verbose=False)
            #xi += means[i-n_dim]-np.mean(xi)
            yi = np.exp(xi)
            zl = hp.anafast(yi, alm=True)
            powi = np.sum(zl[0][0:l_max+1]*l[0:l_max+1]*(l[0:l_max+1]+1.)/(2.*np.pi))
            yi *= np.sqrt(means[i-n_dim]/powi)
            print (' Mean = '+str(np.mean(yi)))
            sys.stdout.flush()
            #yi -= 1.
            hp.write_map(densfold+'density_map'+str(i-npol)+'.fits', yi, overwrite=True)

        firstline = '### Lenspix Parameter File \n'

        if npol == 3 :
            signal='ALL'
            val = 'T'
        else :
            signal='T'
            val = 'F'

        hpxdata = os.environ['HEALPIX']
        if hpxdata[-1] != '/' :
            hpxdata += '/'

        w8dir = hpxdata+'data'

        infact = 2048/nside*1.5

        lenspix_param_file = outfold+'lenspix.ini'

        with open(lenspix_param_file, 'w') as f :
            f.write(firstline)
            f.write('unlensed_alm = '+outfold+cmbalm+'\n')
            f.write('potential_alm = '+outfold+lensalm+'\n')
            f.write('nside = '+str(nside)+'\n')
            f.write('lmax = '+str(lens_lmax)'\n')
            f.write('output_map = '+outfold+cmbmap+'\n')
            f.write('want_pol = '+ val+'\n')
            f.write('interp_factor = '+str(infact)+'\n')
            f.write('mpi_division_method = 3\n')
            f.write('rand_seed = '+str(rand_sd)+'\n')
            f.write('w8dir = '+w8dir+'\n')
        f.close()

        lenspix_start = time.time()

        return_code = subprocess.run(lenspix_root+"dolens "+lenspix_param_file)

        if return_code == 0 :
            print("\nLenspix code run in "+str(time.time() - lenspix_start)+" seconds !!!\n")
        else :
            print("\ncmb_lens has not been able to call the Lenspix software !!!")
            print("\nCheck your installation of Lenspix !!!")
            print (' Quitting!!')
            sys.exit()

        sys.stdout.flush()

        print("\ncpu time =  "+str(time.time() - start)+" seconds !!!\n")


        print ('\n')
        print ('###################################')
        print ('#              Done!              #')
        print ('###################################')
        print ('\n')


        return(0)


if __name__ == "__main__":
    sys.exit(main())

