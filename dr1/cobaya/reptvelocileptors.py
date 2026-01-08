import numpy as np

from cobaya.theory import Theory

import cosmoprimo
from cosmoprimo import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D, PowerSpectrumBAOFilter, Cosmology


def f_over_f0_EH(z, k, Omega0_m, h, fnu, Nnu=3, Neff=3.044):
    r"""
    Computes f(k)/f0, adapted from https://github.com/henoriega/FOLPS-nu, following H&E (1998).

    Reference
    ---------
    https://arxiv.org/pdf/astro-ph/9710216

    Parameters
    ----------
    z : float
        Redshift.
    k : array
        Wavenumber.
    Omega0_m : float
        :math:`\Omega_\mathrm{b} + \Omega_\mathrm{c} + \Omega_\nu` (dimensionless matter density parameter).
    h : float
        :math:`H_0 / 100`.
    fnu : float
        :math:`\Omega_\nu / \Omega_\mathrm{m}`.
    Nnu : int, default=3
        Number of massive neutrinos.
    Neff : int, default=3.044
        Effective number of relativistic species.

    Returns
    -------
    fk : array
        :math:`f(k) / f0`
    """
    eta = np.log(1 / (1 + z))  # log of scale factor
    Omega0_r = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3) * Neff))  # rad: including neutrinos
    aeq = Omega0_r / Omega0_m  # matter-radiation equality

    pcb = 5./4 - np.sqrt(1 + 24*(1 - fnu)) / 4  # neutrino supression
    c = 0.7
    theta272 = (1.00)**2  # T_{CMB} = 2.7*(theta272)
    pf = (k * theta272) / (Omega0_m * h**2)
    DEdS = np.exp(eta) / aeq  # growth function: EdS cosmology

    fnunonzero = np.where(fnu != 0., fnu, 1.)
    yFS = 17.2*fnu*(1 + 0.488*fnunonzero**(-7/6)) * (pf*Nnu / fnunonzero)**2  #yFreeStreaming
    # pcb = 0. and yFS = 0. when fnu = 0.
    rf = DEdS/(1 + yFS)
    return 1 - pcb/(1 + (rf)**c)  # f(k)/f0


_convert_cosmoprimo_to_camb_params = {'H0': 'H0', 'theta_mc': 'cosmomc_theta', 'omega_b': 'ombh2', 'omega_cdm': 'omch2', 'A_s': 'As', 'n_s': 'ns', 'N_eff': 'nnu', 'm_ncdm': 'mnu', 'Omega_k': 'omk', 'w0_fld': 'w', 'wa_fld': 'wa', 'tau_reio': 'tau'}
_convert_cosmoprimo_to_classy_params = {name: name for name in ['H0', 'h', 'A_s', 'sigma8', 'n_s', 'omega_b', 'Omega_b', 'omega_cdm', 'Omega_cdm', 'omega_m', 'Omega_m', 'Omega_ncdm', 'omega_ncdm', 'm_ncdm', 'omega_k', 'Omega_k', 'w0_fld', 'wa_fld', 'tau_reio']}
for name in ['logA', 'ln10^{10}A_s', 'ln10^10A_s', 'ln_A_s_1e10']: _convert_cosmoprimo_to_classy_params[name] = 'ln_A_s_1e10'
_convert_camb_or_classy_to_cosmoprimo_params = {}
for name, value in _convert_cosmoprimo_to_camb_params.items():
    _convert_camb_or_classy_to_cosmoprimo_params[value] = name
for name, value in _convert_cosmoprimo_to_classy_params.items():
    _convert_camb_or_classy_to_cosmoprimo_params[value] = name


class reptvelocileptors(Theory):

    _kinlim = (5e-4, 1.0, 500)
    options = {}
    is_physical_prior = True
    stop_at_error = True

    def initialize(self):
        super().initialize()
        self.options = dict(rbao=110, sbao=None, beyond_gauss=True,
                            one_loop=True, shear=True, cutoff=20, jn=5, N=4000,
                            threads=2, extrap_min=-4, extrap_max=3, import_wisdom=False) | self.options

    def must_provide(self, **requirements):
        """Computed quantities required by the likelihood."""
        super().must_provide(**requirements)
        for k, v in requirements.items():
            if k == 'pkpoles':
                self.z, self.k, self.ells, fiducial = np.asarray(v['z']), np.asarray(v['k']), tuple(v['ells']), v['fiducial']
        self.kin = np.geomspace(min(self._kinlim[0], self.k[0] / 2), max(self._kinlim[1], self.k[-1] * 2), self._kinlim[2])  # margin for AP effect
        self.fiducial = getattr(cosmoprimo.fiducial, fiducial)(engine='camb')
        pk_dd_interpolator_fid = self.fiducial.get_fourier().pk_interpolator(of='delta_cb').to_1d(z=self.z)
        filter = PowerSpectrumBAOFilter(pk_dd_interpolator_fid, engine='peakaverage', cosmo=self.fiducial, cosmo_fid=self.fiducial)
        self._template = dict(filter=filter)
        provide = {}
        provide['Pk_grid'] = {}
        provide['Pk_grid']['nonlinear'] = False
        provide['Pk_grid']['z'] = self.z
        provide['Pk_grid']['k_max'] = 1e2
        provide['Pk_grid']['vars_pairs'] = [('delta_nonu', 'delta_nonu'), ('v_newtonian_cdm', 'v_newtonian_cdm'), ('v_newtonian_cdm', 'v_newtonian_baryon'), ('v_newtonian_baryon', 'v_newtonian_baryon')]
        provide['Hubble'] = {'z': np.concatenate([[0.], self.z])}
        #require_f = self.z
        #provide['fsigma8'] = {'z': require_f}
        #provide['sigma8_z'] = {'z': self.z}
        provide['angular_diameter_distance'] = {'z': self.z}
        provide['rdrag'] = None
        for name in ['Omega_b', 'Omega_cdm', 'Omega_nu_massive']:
            provide[name] = {'z': [0.]}
        return provide

    def set_template(self):
        """Set linear power spectrum (delta-delta, theta-theta) and no-wiggle version."""
        h = np.squeeze(self.provider.get_Hubble(0.)) / 100.
        
        def pk_grid(*var_pair):
            k, z, pk = self.provider.get_Pk_grid(var_pair=var_pair, nonlinear=False)
            iz = np.concatenate([np.flatnonzero(np.isclose(z, zz)) for zz in self.z])
            return k / h, pk.T[..., iz] * h**3

        k, pk = pk_grid('delta_nonu', 'delta_nonu')
        pk_dd_interpolator = PowerSpectrumInterpolator1D(k, pk)

        Omega_b = np.squeeze(self.provider.get_Omega_b(z=0.))
        Omega_cdm = np.squeeze(self.provider.get_Omega_cdm(z=0.))
        Omega_ncdm = np.squeeze(self.provider.get_Omega_nu_massive(z=0.))
        
        # Prepare cosmology
        params = dict(self.provider.params)
        cstate = {_convert_camb_or_classy_to_cosmoprimo_params[param]: value for param, value in params.items() if param in _convert_camb_or_classy_to_cosmoprimo_params}
        cstate = {name: value for name, value in cstate.items() if name in ['n_s']}
        cstate['Omega_b'] = Omega_b
        cstate['Omega_cdm'] = Omega_cdm
        cstate['Omega_ncdm'] = Omega_ncdm
        cstate['H0'] = 100. * h
        #cosmo = self.fiducial.clone(**cstate)
        # NOTE: best to provide omega_b, omega_cdm, H0, ns
        cosmo = Cosmology(**cstate)

        #cosmo = type(object)('cosmo', (), {'rs_drag': self.provider.get_param('rdrag') * h})
        cosmo.rs_drag = self.provider.get_param('rdrag') * h
        self._template['filter'](pk_dd_interpolator, cosmo=cosmo)
        sigma8 = pk_dd_interpolator.sigma8()
        pknow_dd_interpolator = self._template['filter'].smooth_pk_interpolator()

        Omega_m = Omega_cdm + Omega_b + Omega_ncdm
        fnu = Omega_ncdm / Omega_m
        fb = Omega_b / (Omega_cdm + Omega_b)
        pk = fb**2 * pk_grid('v_newtonian_baryon', 'v_newtonian_baryon')[-1] + 2 * fb * (1. - fb) * pk_grid('v_newtonian_baryon', 'v_newtonian_cdm')[-1] + (1. - fb)**2 * pk_grid('v_newtonian_cdm', 'v_newtonian_cdm')[-1]
        pk_tt_interpolator = PowerSpectrumInterpolator1D(k, pk)
        self._template.update(pk_dd_interpolator=pk_dd_interpolator, pknow_dd_interpolator=pknow_dd_interpolator, pk_tt_interpolator=pk_tt_interpolator)
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """Set theory power spectrum multipoles."""
        self.set_template()
        sigma8 = self._template['pk_dd_interpolator'].sigma8()
        fsigma8 = self._template['pk_tt_interpolator'].sigma8()
        pk_dd = self._template['pk_dd_interpolator'](self.kin)
        pknow_dd = self._template['pknow_dd_interpolator'](self.kin)
        pk_tt = self._template['pk_tt_interpolator'](self.kin)

        h = np.squeeze(self.provider.get_Hubble(0.)) / 100.
        qpar = self.fiducial.efunc(self.z) / (self.provider.get_Hubble(self.z, units="km/s/Mpc") / (100. * h))
        qper = self.provider.get_angular_diameter_distance(self.z) * h / self.fiducial.angular_diameter_distance(self.z)

        from velocileptors.EPT.ept_fullresum_varyDz_nu_fftw import REPT
        self.pt = REPT(self.kin, pk_dd[..., 0], pnw=pknow_dd[..., 0], kmin=self.k[0], kmax=self.k[-1], nk=200, **self.options)
        # print(self.template.f, self.k.shape, self.template.qpar, self.template.qper, self.template.k.shape, self.template.pk_dd.shape)
        pktable = {ell: [] for ell in [0, 2, 4]}

        from scipy import interpolate
        pcb, pcb_nw, ptt = [10**interpolate.interp1d(np.log10(self.kin), np.log10(pk), kind='cubic', fill_value='extrapolate', axis=0, assume_sorted=True)(np.log10(np.append(self.pt.kv, 1.))) for pk in [pk_dd, pknow_dd, pk_tt]]
        # print(pk_dd.sum(), pknow_dd.sum(), pk_tt.sum(), self.kin[0], self.kin[-1], self.kin.shape)
        for iz, z in enumerate(self.z):
            #Dz = sigma8[iz] / sigma8[0]
            Dz = np.sqrt(pcb[-1, iz] / pcb[-1, 0])
            # fk = f0[iz] * f_over_f0_EH(z, self.pt.kv, Omega_m, h, fnu)
            fk = np.sqrt(ptt[:-1, iz] / pcb[:-1, iz])
            # print(iz, z, qpar[iz], qper[iz], Dz, Omega_m, fnu)
            # print(iz, qpar[iz], qper[iz], Dz, pcb[:-1, iz].sum(), pcb_nw[:-1, iz].sum(), fk.sum(), self.pt.kv.min(), self.pt.kv.max(), self.pt.kv.shape)
            pks = self.pt.compute_redshift_space_power_multipoles_tables(fk, apar=qpar[iz], aperp=qper[iz], ngauss=4, pcb=pcb[:-1, iz], pcb_nw=pcb_nw[:-1, iz], Dz=Dz)[1:]
            for ill, ell in enumerate(pktable): pktable[ell].append(pks[ill])
        pktable = {ell: np.concatenate([v[..., None] for v in value], axis=-1) for ell, value in pktable.items()}
        pktable = interpolate.interp1d(self.pt.kv, np.array([pktable[ell] for ell in self.ells]), kind='cubic', fill_value='extrapolate', axis=1, assume_sorted=True)(self.k)
        state['pkpoles'] = pktable
        state['sigma8'] = sigma8
        state['fsigma8'] = fsigma8

    def get_pkpoles(self, params, z=None, sigv=15., fsat=0.1, sn=1e4, return_gradient=False):
        pktable, sigma8, fsigma8 = self.current_state['pkpoles'], self.current_state['sigma8'], self.current_state['fsigma8']
        f = fsigma8 / sigma8
        nd = 1e-4
        # Gradient only for alpha, sn
        gradient = np.zeros((7,) * 2, dtype='f8')
        iz = list(self.z).index(z)
        if self.is_physical_prior:
            sigma8, f = sigma8[iz], f[iz]
            pars = b1L, b2L, bsL, b3L = [params['b1p'] / sigma8 - 1., params['b2p'] / sigma8**2, params['bsp'] / sigma8**2, params['b3p'] / sigma8**3]
            pars = [1. + b1L, 8. / 21. * b1L + b2L, bsL - (2 / 7) * b1L, 3 * b3L + b1L]
            #pars += [(1 + b1L)**2 * params['alpha0p'], f * (1 + b1L) * (params['alpha0p'] + params['alpha2p']), f * (f * params['alpha2p'] + (1 + b1L) * params['alpha4p']), f**2 * params['alpha4p']]
            params['alpha6p'] = 0.
            pars += [params[name] for name in ['alpha0p', 'alpha2p', 'alpha4p', 'alpha6p']]
            gradient[0, 0] = (1 + b1L)**2
            gradient[1, 0] = gradient[1, 1] = f * (1 + b1L)
            gradient[2, 1] = f**2
            gradient[2, 2] = f * (1 + b1L)
            gradient[3, 2] = f**2
            pars += [params['sn{:d}p'.format(i)] for i in [0, 2, 4]]
            for ii, i in enumerate([0, 2, 4]):
                gradient[ii - 3, ii - 3] = sn * (fsat if i > 0 else 1.) * sigv**i
        else:
            pars = [params[name] for name in ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4']]
            for ii, i in enumerate([0, 2, 4]):
                gradient[ii - 3, ii - 3] = 1. / nd
        
        pars = list(pars)
        #b1 = pars[0]
        #pars[2] = pars[2] - (2 / 7) * (b1 - 1.)  # bs
        #pars[3] = 3 * pars[3] + (b1 - 1.)  # b3
        if z is not None: pktable = pktable[..., iz]

        def tablevel_combine_bias_terms_poles(pktable, pars):
            """Sum bias terms."""
            b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4 = pars
            bias_monomials = np.concatenate([np.array([1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3]), gradient.dot(np.array([alpha0, alpha2, alpha4, alpha6, sn0, sn2, sn4]))])
            return np.sum(pktable * bias_monomials, axis=-1)

        poles = tablevel_combine_bias_terms_poles(pktable, pars)
        if return_gradient:
            return poles, pktable[..., -7:].dot(gradient)
        return poles