from pathlib import Path
import numpy as np

from cobaya.likelihood import Likelihood


list_zrange = [('BGS_BRIGHT-21.5', 0, (0.1, 0.4)), ('LRG', 0, (0.4, 0.6)), ('LRG', 1, (0.6, 0.8)), ('LRG', 2, (0.8, 1.1)), ('ELG_LOPnotqso', 1, (1.1, 1.6)), ('QSO', 0, (0.8, 2.1)), ('Lya', 0, (1.8, 4.2))]


def dataset_fn(data_dir, tracer, zrange, observable_name='spectrum-poles+bao-recon'):
    if observable_name == 'bao-recon':
        if 'lya' in tracer.lower():
            return data_dir / f'likelihood_bao_syst_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
        return data_dir / f'likelihood_{observable_name}_syst_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
    return data_dir / f'likelihood_{observable_name}_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'


def get_tracer_label(tracer):
    return tracer.split('_')[0].replace('+', 'plus')


def get_physical_stochastic_settings(tracer=None):
    if tracer is not None:
        tracer = str(tracer).upper()
        # Mark Maus, Ruiyang Zhao
        settings = {'BGS': {'fsat': 0.15, 'sigv': 150*(10)**(1/3)*(1+0.2)**(1/2)/70.},
                    'LRG': {'fsat': 0.15, 'sigv': 150*(10)**(1/3)*(1+0.8)**(1/2)/70.},
                    'ELG': {'fsat': 0.10, 'sigv': 150*2.1**(1/2)/70.},
                    'QSO': {'fsat': 0.03, 'sigv': 150*(10)**(0.7/3)*(2.4)**(1/2)/70.}}
        try:
            settings = settings[tracer]
        except KeyError:
            raise ValueError('unknown tracer: {}, please use any of {}'.format(tracer, list(settings.keys())))
    else:
        settings = {'fsat': 0.1, 'sigv': 5.}
    return settings


class desi_fs_bao_all(Likelihood):
    
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
        self.flatdata, self.window, self.precision, self.shotnoise, self.bao_quantities, self._requirements = [], [], [], [], [], {}
        if 'bao' in self.observable_name:
            self._requirements.update({'angular_diameter_distance': {'z': []}, 'Hubble': {'z': []}, 'rdrag': None})
        # Read data
        for tracer, zrange, namespace in self.zbins:
            has_no_fs = 'Lya' in tracer
            flatdata = []
            likelihood_data = types.read(dataset_fn(self.data_dir, tracer, zrange, observable_name='bao-recon' if has_no_fs else self.observable_name))
            # Full shape (power spectrum)
            if not has_no_fs:
                spectrum = likelihood_data.observable.get('spectrum')  # power spectrum
                flatdata.append(spectrum.value())
                spectrum_theory = likelihood_data.window.theory.get('spectrum')
                if 'pkpoles' in self._requirements:  # already set
                    for pole in spectrum_theory:
                        assert np.allclose(pole.coords('k'), self.kin)
                else:
                    self.kin = spectrum_theory.get(0).coords('k')
                    self._requirements['pkpoles'] = {'k': self.kin, 'z': [], 'ells': (0, 2, 4), 'fiducial': 'DESI'}
                # Select the "spectrum" block only
                # Same as likelihood_data.window.value()[-size_bao:, -size_bao:] where size_bao = likelihood_data.observable.size - spectrum.size
                window = likelihood_data.window.at.observable.get('spectrum').at.theory.get('spectrum')
                self.window.append(window.value())
                self._requirements['pkpoles']['z'].append(spectrum.attrs['zeff'])
            self.shotnoise.append(np.mean(spectrum.get(0).values('shotnoise')))
            if 'bao' in self.observable_name:
                # BA0
                bao = likelihood_data.observable.get('baorecon')  # BAO
                flatdata.append(bao.value())
                z = bao.attrs['zeff']
                if has_no_fs: assert all('iso' not in param for param in bao.parameters)  # just qpar, qper
                self.bao_quantities.append((z, list(bao.parameters)))
                self._requirements['angular_diameter_distance']['z'].append(z)
                self._requirements['Hubble']['z'].append(z)
                self._requirements['rdrag'] = None
            # Joint
            self.flatdata.append(np.concatenate(flatdata))
            covariance = likelihood_data.covariance.value()
            precision = np.linalg.inv(covariance)
            self.precision.append(precision)
      
    def get_requirements(self):
        """Return dictionary specifying quantities calculated by a theory code are needed."""
        return self._requirements

    def get_bao_flattheory(self, z, params):
        """Return the (flattened) theory vector for BAO parameters."""
        rdrag = self.provider.get_param('rdrag')
        apar = np.squeeze(1. / (self.provider.get_Hubble(z, units="km/s/Mpc") / 100.) / rdrag / (1. / self.fiducial.efunc(z) / self.fiducial.rs_drag))
        aper = np.squeeze(self.provider.get_angular_diameter_distance(z) / rdrag / (self.fiducial.angular_diameter_distance(z) / self.fiducial.rs_drag))
        coeffs = [{'iso': (1./3., 2./3.), 'par': (1., 0.), 'per': (0., 1.), 'ap': (1., -1.)}[param[1:]] for param in params]
        return np.array([apar**coeff[0] * aper**coeff[1] for coeff in coeffs])

    def logp(self, _derived=None, **params_values):
        """
        Take a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        logp = 0.
        # Parameters that can be marginalized over, and the scale of their Gaussian priors
        all_marg_params = ['alpha0p', 'alpha2p', 'alpha4p', 'alpha6p', 'sn0p', 'sn2p', 'sn4p']
        all_marg_scales = [12.5] * 4 + [2.] + [5.] * 2
        all_marg_scales = np.array(all_marg_scales)
        # What parameters to marginalize over
        marg_params = ['alpha0p', 'alpha2p', 'sn0p', 'sn2p']

        gradient_indices = np.array([all_marg_params.index(param) for param in marg_params])  # alpha0, alpha2, sn0, sn2
        prior_hessian = -np.diag(np.array([all_marg_scales[idx]**(-2.) for idx in gradient_indices]))
        # Iterate on all tracers / z-bins
        for i, (tracer, zrange, name) in enumerate(self.zbins):
            has_no_fs = 'Lya' in tracer
            flattheory, gradient = [], []
            # First power spectrum
            if not has_no_fs:
                namespace = 'pre_' + name
                fs_params = {key[len(namespace) + 1:]: value for key, value in params_values.items() if key.startswith(namespace)} | {param: 0. for param in all_marg_params}
                #fs_params |= {'alpha0p': 10., 'alpha2p': 5., 'sn0p': 5., 'sn2p': 8.} 
                settings = get_physical_stochastic_settings(tracer=tracer.upper()[:3])
                z = self._requirements['pkpoles']['z'][i]
                fs_theory, fs_gradient = self.provider.get_pkpoles(fs_params, z=z, **settings, sn=self.shotnoise[i], return_gradient=True)
                flattheory.append(self.window[i].dot(fs_theory.ravel()))
                fs_gradient = fs_gradient[..., gradient_indices].reshape(-1, *gradient_indices.shape)
                gradient.append(self.window[i].dot(fs_gradient))
            # Then BAO
            if 'bao' in self.observable_name:
                bao_theory = self.get_bao_flattheory(*self.bao_quantities[i])
                flattheory.append(bao_theory)
                bao_gradient = np.zeros(bao_theory.shape + gradient_indices.shape)
                gradient.append(bao_gradient)
            flattheory = np.concatenate(flattheory)
            diff = flattheory - self.flatdata[i]
            loglikelihood = - 1. / 2. * diff.T.dot(self.precision[i]).dot(diff)
            marg_logprior = 0.
            
            if not has_no_fs:
                # Now analytic marginalization
                gradient = np.concatenate(gradient)
                pgrad = self.precision[i].dot(gradient)
                likelihood_gradient = - pgrad.T.dot(diff)
                posterior_gradient = likelihood_gradient
                likelihood_hessian = - gradient.T.dot(pgrad)
                posterior_hessian = prior_hessian + likelihood_hessian
                dx = - np.linalg.solve(posterior_hessian, posterior_gradient)
                loglikelihood += 1. / 2. * dx.dot(likelihood_hessian).dot(dx)
                loglikelihood += posterior_gradient.dot(dx)
                marg_logprior += 1. / 2. * dx.dot(prior_hessian).dot(dx)
                if self.solve == 'marg':
                    loglikelihood += -1. / 2. * np.linalg.slogdet(- posterior_hessian)[1]
            # Sum loglikelihoods over all z-bins
            #print(marg_logprior)
            if _derived is not None:
                _derived['{}.loglikelihood'.format(name)] = loglikelihood
                #_derived['{}.logprior'.format(name)] = marg_logprior
            logp += loglikelihood + marg_logprior
            #if i == 0: print(logp, loglikelihood, marg_logprior)
        return logp