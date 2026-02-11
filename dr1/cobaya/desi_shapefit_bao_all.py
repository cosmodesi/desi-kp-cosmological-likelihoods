from pathlib import Path
import numpy as np

from cobaya.likelihood import Likelihood
import cosmoprimo
from cosmoprimo import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D, PowerSpectrumBAOFilter, Cosmology


list_zrange = [('BGS_BRIGHT-21.5', 0, (0.1, 0.4)), ('LRG', 0, (0.4, 0.6)), ('LRG', 1, (0.6, 0.8)), ('LRG', 2, (0.8, 1.1)), ('ELG_LOPnotqso', 1, (1.1, 1.6)), ('QSO', 0, (0.8, 2.1)), ('Lya', 0, (1.8, 4.2))]


def dataset_fn(data_dir, tracer, zrange, observable_name='spectrum-poles+bao-recon'):
    data_dir = Path(data_dir)
    if observable_name == 'bao-recon':
        if 'lya' in tracer.lower():
            return data_dir / f'likelihood_bao_syst_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
        return data_dir / f'likelihood_{observable_name}_syst_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
    return data_dir / f'likelihood_shapefit_{observable_name}_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'


def get_tracer_label(tracer):
    return tracer.split('_')[0].replace('+', 'plus')


_convert_cosmoprimo_to_camb_params = {'H0': 'H0', 'theta_mc': 'cosmomc_theta', 'omega_b': 'ombh2', 'omega_cdm': 'omch2', 'A_s': 'As', 'n_s': 'ns', 'N_eff': 'nnu', 'm_ncdm': 'mnu', 'Omega_k': 'omk', 'w0_fld': 'w', 'wa_fld': 'wa', 'tau_reio': 'tau'}
_convert_cosmoprimo_to_classy_params = {name: name for name in ['H0', 'h', 'A_s', 'sigma8', 'n_s', 'omega_b', 'Omega_b', 'omega_cdm', 'Omega_cdm', 'omega_m', 'Omega_m', 'Omega_ncdm', 'omega_ncdm', 'm_ncdm', 'omega_k', 'Omega_k', 'w0_fld', 'wa_fld', 'tau_reio']}
for name in ['logA', 'ln10^{10}A_s', 'ln10^10A_s', 'ln_A_s_1e10']: _convert_cosmoprimo_to_classy_params[name] = 'ln_A_s_1e10'
_convert_camb_or_classy_to_cosmoprimo_params = {}
for name, value in _convert_cosmoprimo_to_camb_params.items():
    _convert_camb_or_classy_to_cosmoprimo_params[value] = name
for name, value in _convert_cosmoprimo_to_classy_params.items():
    _convert_camb_or_classy_to_cosmoprimo_params[value] = name


class desi_shapefit_bao_all(Likelihood):
    
    # Only meaningful deviation to exact is using camb as engine for fiducial cosmology
    # Typicall offset in the loglikelihood of 0.02 for 1176.98

    def initialize(self):
        """Prepare any computation, importing any necessary code, files, etc."""
        import lsstypes as types
        assert self.observable_name in ['spectrum-poles-rotated', 'spectrum-poles-rotated+bao-recon']
        self.tracers = [tracer.lower() for tracer in self.tracers]
        from cosmoprimo.fiducial import DESI
        self.fiducial = DESI(engine='camb')
        self.zbins = []
        # Select tracer / z-bin to be fitted
        for tracer, iz, zrange in list_zrange:
            tracer_label = get_tracer_label(tracer)
            namespace = '{tracer}_z{iz}'.format(tracer=tracer_label, iz=iz)
            if self.tracers is not None and namespace.lower() not in self.tracers: continue
            self.zbins.append((tracer, zrange, namespace))
        self.log.info('Fitting {}.'.format(self.zbins))
        self.flatdata, self.precision, self._requirements = [], [], {}
        self._requirements.update({'angular_diameter_distance': {'z': []}, 'Hubble': {'z': [0.]}, 'rdrag': None})
        for name in ['Omega_b', 'Omega_cdm', 'Omega_nu_massive']:
            self._requirements[name] = {'z': [0.]}
        self._requirements['Pk_grid'] = {'nonlinear': False, 'z': [], 'k_max': 1e2, 'vars_pairs': [('delta_nonu', 'delta_nonu'), ('v_newtonian_cdm', 'v_newtonian_cdm'), ('v_newtonian_cdm', 'v_newtonian_baryon'), ('v_newtonian_baryon', 'v_newtonian_baryon')]}
        # Read data
        self._quantities = []
        for tracer, zrange, namespace in self.zbins:
            has_no_fs = 'Lya' in tracer
            flatdata = []
            likelihood_data = types.read(dataset_fn(self.data_dir, tracer, zrange, observable_name='bao-recon' if has_no_fs else self.observable_name))
            if not has_no_fs:
                shapefit = likelihood_data.observable.get('shapefit')  # ShapeFit
                flatdata.append(shapefit.value())
                z = shapefit.attrs['zeff']
                self._requirements['Pk_grid']['z'].append(z)
                self._requirements['angular_diameter_distance']['z'].append(z)
                self._requirements['Hubble']['z'].append(z)
                self._quantities.append((z, list(shapefit.parameters)))
            else:
                bao = likelihood_data.observable.get('baorecon')  # BAO
                flatdata.append(bao.value())
                z = bao.attrs['zeff']
                self._requirements['angular_diameter_distance']['z'].append(z)
                self._requirements['Hubble']['z'].append(z)
                self._quantities.append((z, list(bao.parameters)))
            self.flatdata.append(np.concatenate(flatdata))
            covariance = likelihood_data.covariance.value()
            #assert np.allclose(covariance.T, covariance)
            precision = np.linalg.inv(covariance)
            self.precision.append(precision)
        self.z = list(self._requirements['Pk_grid']['z'])
        self._template = dict()
        if self.z:
            fo = self.fiducial.get_fourier()
            pk_dd_interpolator = fo.pk_interpolator(of='delta_cb').to_1d(z=self.z)
            pk_tt_interpolator = fo.pk_interpolator(of='theta_cb').to_1d(z=self.z)
            filter = PowerSpectrumBAOFilter(pk_dd_interpolator, engine='peakaverage', cosmo=self.fiducial, cosmo_fid=self.fiducial)
            filter(pk_dd_interpolator, cosmo=self.fiducial)
            pknow_dd_interpolator = filter.smooth_pk_interpolator()
            shapefit = self._get_f_m(pk_dd_interpolator, pknow_dd_interpolator, pk_tt_interpolator)
            self._template.update(pk_dd_interpolator_fid=pk_dd_interpolator,
                                  pknow_dd_interpolator_fid=pknow_dd_interpolator,
                                  pk_tt_interpolator_fid=pk_tt_interpolator,
                                  filter=filter)
            self._template.update({name + '_fid': value for name, value in shapefit.items()})
      
    def get_requirements(self):
        """Return dictionary specifying quantities calculated by a theory code are needed."""
        return self._requirements

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

    def _get_f_m(self, pk_dd_interpolator, pknow_dd_interpolator, pk_tt_interpolator, s=1.):
        """Return theory ShapeFit parameters."""
        kp = 0.03  # pivot k in fiducial coordinates
        kp = kp / s
        dk = 1e-2
        k = kp * np.array([1. - dk, 1. + dk])
        pknow_dd = pknow_dd_interpolator(k)
        m = np.diff(np.log(pknow_dd), axis=0)[0] / np.diff(np.log(k))[0]
        Ap = 1. / s**3 * pk_dd_interpolator(kp)
        f = pk_tt_interpolator.sigma8() / pk_dd_interpolator.sigma8()
        f_sqrt_Ap = f * Ap**0.5
        return dict(m=m, f=f, Ap=Ap, f_sqrt_Ap=f_sqrt_Ap)
    
    def get_flattheory(self, z, params, **shapefit):
        """Return the (flattened) theory vector for the ShapeFit or BAO parameters."""
        rdrag = self.provider.get_param('rdrag')
        apar = np.squeeze(1. / (self.provider.get_Hubble(z, units="km/s/Mpc") / 100.) / rdrag / (1. / self.fiducial.efunc(z) / self.fiducial.rs_drag))
        aper = np.squeeze(self.provider.get_angular_diameter_distance(z) / rdrag / (self.fiducial.angular_diameter_distance(z) / self.fiducial.rs_drag))
        s = rdrag / self.fiducial.rs_drag
        flattheory = [np.nan] * len(params)
        for iparam, param in enumerate(params):
            if param in ['qpar', 'qper', 'qiso', 'qap']:
                coeff = {'iso': (1./3., 2./3.), 'par': (1., 0.), 'per': (0., 1.), 'ap': (1., -1.)}[param[1:]]
                flattheory[iparam] = apar**coeff[0] * aper**coeff[1]
            elif param == 'df':
                idx = self.z.index(z)
                flattheory[iparam] = shapefit['f_sqrt_Ap'][idx] / self._template['f_sqrt_Ap_fid'][idx]
            elif param == 'dm':
                idx = self.z.index(z)
                flattheory[iparam] = shapefit['m'][idx] - self._template['m_fid'][idx]
            else:
                raise ValueError(f'{param} not known')
        return np.array(flattheory)

    def logp(self, _derived=None, **params_values):
        """
        Take a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        logp = 0.
        shapefit = dict()
        if self.z:
            self.set_template()
            h = np.squeeze(self.provider.get_Hubble(0.)) / 100.
            s = (self.provider.get_param('rdrag') * h) / self.fiducial.rs_drag
            shapefit = self._get_f_m(self._template['pk_dd_interpolator'], self._template['pknow_dd_interpolator'], self._template['pk_tt_interpolator'], s=s)
        for i, (tracer, zrange, name) in enumerate(self.zbins):
            flattheory = self.get_flattheory(*self._quantities[i], **shapefit)
            diff = flattheory - self.flatdata[i]
            loglikelihood = - 1. / 2. * diff.T.dot(self.precision[i]).dot(diff)
            logp += loglikelihood
            if _derived is not None:
                _derived['{}.loglikelihood'.format(name)] = loglikelihood
        return logp