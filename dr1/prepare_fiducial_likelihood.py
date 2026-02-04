from pathlib import Path
import numpy as np

import lsstypes as types

# CHANGE Path here
#dr_base_dir = Path('/global/cfs/cdirs/desi/users/adematti/dr1_release/dr1-fs-bao-clustering-measurements')
dr_base_dir = Path('.')


def get_observable(tracer, zrange, dataset='spectrum-poles-rotated', check=False):
    """
    Return the fiducial data vector: rotated and corrected power spectrum multipoles, optionally with post-reconstruction BAO.

    Parameters
    ----------
    tracer : str
        Tracer name, one of ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']
    zrange : tuple of floats
        Redshift range.
    dataset : str, default='spectrum-poles-rotated'
        Data vector, 'spectrum-poles-rotated' for power spectrum-only, 'spectrum-poles-rotated+bao-recon' for joint power spectrum + BAO fit.
    check : bool, default=False
        Whether to check that "cooked" observable matches that on disk.

    Returns
    -------
    observable : types.ObservableTree
    """
    assert dataset in ['spectrum-poles-rotated', 'spectrum-poles-rotated+bao-recon']
    # Raw power spectrum
    regions = ['NGC', 'SGC']
    fns = [dr_base_dir / f'data/spectrum/spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5' for region in regions]
    spectrum_raw = types.sum([types.read(fn) for fn in fns])  # norm-weighted average of NGC and SGC
    spectrum_raw = spectrum_raw.select(k=slice(0, None, 5)).select(k=(0., 0.4))
    # Corrected power spectrum
    # Compensate for RIC and angular mode removal
    fn_ric = dr_base_dir / f'data/templates_spectrum/template_ric_spectrum-poles_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    fn_amr = dr_base_dir / f'data/templates_spectrum/template_amr_spectrum-poles_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    spectrum_raw_cut = spectrum_raw.select(k=(0.02, 0.4))  # rebin and select k-range
    spectrum_corrected = spectrum_raw_cut.clone(value=spectrum_raw_cut.value()
                                            - types.read(fn_ric).match(spectrum_raw_cut).value()
                                            - types.read(fn_amr).match(spectrum_raw_cut).value())
    if check:
        # Check the obtained "corrected power spectrum" matches the one on disk
        assert np.allclose(types.read(dr_base_dir / f'data/spectrum/spectrum-poles-corrected_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5').match(spectrum_corrected).value(), spectrum_corrected.value())
    # Rotated power spectrum
    fn = dr_base_dir / f'data/rotation/rotation_spectrum-poles_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    rotation = types.read(fn)
    # Concatenate raw power spectrum, and best-fit s coefficients, eq. 5.4 of https://arxiv.org/pdf/2406.04804
    value = rotation['M'].value().dot(spectrum_raw.value()) - sum(s.value() * mo.value() for s, mo in zip(rotation['s'], rotation['mo']))
    spectrum_rotated = spectrum_raw.clone(value=value)
    spectrum_rotated = spectrum_rotated.select(k=(0., 0.2))
    tmp = types.read(dr_base_dir / f'data/spectrum/spectrum-poles-rotated_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5')
    if check:
        assert np.allclose(types.read(dr_base_dir / f'data/spectrum/spectrum-poles-rotated_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5').value(), spectrum_rotated.value())
    # Rotated, corrected power spectrum
    fn_ric = dr_base_dir / f'data/templates_spectrum/template_ric_spectrum-poles-rotated_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    fn_amr = dr_base_dir / f'data/templates_spectrum/template_amr_spectrum-poles-rotated_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    spectrum_rotated_cut = spectrum_rotated.select(k=(0.02, 0.2))
    spectrum_rotated_corrected = spectrum_rotated_cut.clone(value=spectrum_rotated_cut.value()
                                                        - types.read(fn_ric).match(spectrum_rotated_cut).value()
                                                        - types.read(fn_amr).match(spectrum_rotated_cut).value())
    observable = spectrum_rotated_corrected
    if check:
        assert np.allclose(types.read(dr_base_dir / f'data/spectrum/spectrum-poles-rotated-corrected_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5').value(), spectrum_rotated_corrected.value())
    if dataset == 'spectrum-poles-rotated+bao-recon':
        fn = dr_base_dir / f'data/likelihood/likelihood_bao-recon_stat-only_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
        observable_bao = types.read(fn).observable.get('baorecon')
        observable = types.ObservableTree([spectrum_rotated_corrected, observable_bao], observables=['spectrum', 'baorecon'])
    return observable


def rebin_window_matrix(window):
    # Rebin window matrix
    from scipy import linalg
    kin = np.arange(0.001, 0.35, 0.001)

    def matrix_lininterp(xin, xout):
        # Matrix for linear interpolation
        toret = np.zeros((len(xin), len(xout)), dtype='f8')
        for iout, xout in enumerate(xout):
            iin = np.searchsorted(xin, xout, side='right') - 1
            if 0 <= iin < len(xin) - 1:
                frac = (xout - xin[iin]) / (xin[iin + 1] - xin[iin])
                toret[iin, iout] = 1. - frac
                toret[iin + 1, iout] = frac
            elif np.isclose(xout, xin[-1]):
                toret[iin, iout] = 1.
        assert np.all(toret <= 1.)
        return toret

    rebin = linalg.block_diag(*[matrix_lininterp(kin, pole.coords('k')) for pole in window.theory])
    value = window.value().dot(rebin.T)  # rebinned window matrix

    theory = types.Mesh2SpectrumPoles([types.Mesh2SpectrumPole(k=kin, num_raw=np.zeros_like(kin)) for ell in window.theory.ells], ells=window.theory.ells)
    return window.clone(value=value, theory=theory)


def get_window_matrix(tracer, zrange, dataset='spectrum-poles-rotated', check=False):
    """
    Return the fiducial, rotated, window matrix.

    Parameters
    ----------
    tracer : str
        Tracer name, one of ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']
    zrange : tuple of floats
        Redshift range.
    dataset : str, default='spectrum-poles-rotated'
        Data vector, 'spectrum-poles-rotated' for power spectrum-only, 'spectrum-poles-rotated+bao-recon' for joint power spectrum + BAO fit.
    check : bool, default=False
        Whether to check that "cooked" window matrix matches that on disk.

    Returns
    -------
    window : types.WindowMatrix
    """
    assert dataset in ['spectrum-poles-rotated', 'spectrum-poles-rotated+bao-recon']
    # Raw window matrix
    from scipy import linalg
    regions = ['NGC', 'SGC']
    fns = [dr_base_dir / f'data/spectrum/window_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5' for region in regions]
    window_raw = types.sum([types.read(fn) for fn in fns])
    if check:
        assert np.allclose(window_raw.value(), types.read(dr_base_dir / f'data/spectrum/window_spectrum-poles_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5').value())
    # Rotated window matrix
    fn = dr_base_dir / f'data/rotation/rotation_spectrum-poles_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    rotation = types.read(fn)
    # First select / rebin to the binning used in the rotation (0 < k < 0.4 h/Mpc with dk = 0.005 h/Mpc)
    window_raw = window_raw.at.observable.match(rotation['M'].theory)
    # And cut theory
    kin = rotation['theory'].get(0).coords('k')
    window_raw = window_raw.at.theory.select(k=(kin.min() * (1 - 1e-9), kin.max() * (1 + 1e-9)))
    # eq. 5.5 of https://arxiv.org/pdf/2406.04804
    value = rotation['M'].value().dot(window_raw.value()) - sum(mo.value()[:, None] * mt.value() for mo, mt in zip(rotation['mo'], rotation['mt']))
    window_rotated = window_raw.clone(value=value)
    if check:
        assert np.allclose(window_rotated.at.observable.select(k=(0., 0.2)).value(), types.read(dr_base_dir / f'data/spectrum/window_spectrum-poles-rotated_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5').value())

    # Rebin window matrix
    window = window_rotated = rebin_window_matrix(window_rotated)

    if dataset == 'spectrum-poles-rotated+bao-recon':
        # Pad with identity matrix for the BAO
        # Just to get the size of the BAO data vector (1 if isotropic, 2 if anisotropic)
        fn = dr_base_dir / f'data/likelihood/likelihood_bao-recon_stat-only_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
        observable_bao = types.read(fn).observable.get('baorecon')
        window = types.WindowMatrix(value=linalg.block_diag(window.value(), np.eye(observable_bao.size)),
                                    observable=types.ObservableTree([window.observable, observable_bao], observables=['spectrum', 'baorecon']),
                                    theory=types.ObservableTree([window.theory, observable_bao], observables=['spectrum', 'baorecon']))
    return window


def get_covariance_matrix(tracer, zrange, dataset='spectrum-poles-rotated', check=False):
    """
    Return the fiducial, rotated, covariance matrix including systematic contributions.

    Parameters
    ----------
    tracer : str
        Tracer name, one of ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']
    zrange : tuple of floats
        Redshift range.
    dataset : str, default='spectrum-poles-rotated'
        Data vector, 'spectrum-poles-rotated' for power spectrum-only, 'spectrum-poles-rotated+bao-recon' for joint power spectrum + BAO fit.
    check : bool, default=False
        Whether to check that "cooked" covariance matrix matches that on disk.

    Returns
    -------
    covariance : types.CovarianceMatrix
    """
    # dataset = 'spectrum-poles-rotated' or 'spectrum-poles-rotated+bao-recon'
    assert dataset in ['spectrum-poles-rotated', 'spectrum-poles-rotated+bao-recon']
    from scipy import linalg
    # Raw covariance matrix
    fn = dr_base_dir / f'data/covariance/EZmock/covariance_{dataset.replace("-rotated", "")}_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    covariance_raw = types.read(fn)
    if check:
        imocks = range(1, 1001)
        mtracer = {'BGS_BRIGHT-21.5': 'BGS', 'ELG_LOPnotqso': 'ELG_LOP'}.get(tracer, tracer)
        observables = [types.read(dr_base_dir / f'EZmock/ffa/spectrum/spectrum-poles_{mtracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05_{imock:d}.h5') for imock in imocks]
        if dataset == 'spectrum-poles-rotated+bao-recon':
            observables_bao = [types.read(dr_base_dir / f'EZmock/ffa/recsym/bao/bao-recsym_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5') for imock in imocks]
            observables = [types.ObservableTree([spectrum, bao], observables=['spectrum', 'baorecon']) for spectrum, bao in zip(observables, observables_bao)]
        covariance_mock = types.cov(observables).at.observable.match(covariance_raw.observable)
        assert np.allclose(covariance_mock.value(), covariance_raw.value())
    # Rotated covariance matrix
    fn = dr_base_dir / f'data/rotation/rotation_spectrum-poles_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    rotation = types.read(fn)
    M = rotation['M'].value()
    # Just get the "M" rotation matrix
    if dataset != 'spectrum-poles-rotated':  # pad with diagonal block
        M = linalg.block_diag(M, np.eye(covariance_raw.shape[0] - M.shape[0]))
    # eq. 5.6 of https://arxiv.org/pdf/2406.04804
    covariance_rotated = covariance_raw.clone(value=M.dot(covariance_raw.value()).dot(M.T))
    if check:
        fn = dr_base_dir / f'data/covariance/EZmock/covariance_{dataset}_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
        assert np.allclose(types.read(fn).value(), covariance_rotated.value())

    # Now applying the final selection
    ells, klim = [0, 2], (0.02, 0.2)
    if dataset != 'spectrum-poles-rotated':
        # Apply selection to the "spectrum" part
        observable = covariance_rotated.observable.at(observables='spectrum').get(ells=ells).select(k=klim)
    else:
        observable = covariance_rotated.observable.get(ells=ells).select(k=klim)
    covariance_rotated = covariance_rotated.at.observable.match(observable)

    # Rescaling factor, see KP3 paper, Table 6
    factor = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 1.39, ('LRG', (0.4, 0.6)): 1.15, ('LRG', (0.6, 0.8)): 1.15, ('LRG', (0.8, 1.1)): 1.22, ('ELG_LOPnotqso', (0.8, 1.1)): 1.25, ('ELG_LOPnotqso', (1.1, 1.6)): 1.29, ('QSO', (0.8, 2.1)): 1.11}[tracer, zrange]

    nobs = 1000
    nbins = covariance_rotated.shape[0]
    # Percival2014 and Hartlap2007 rescaling factors for the mock-based covariance matrix
    def get_percival2014_factor(nobs, nbins, nparams):
        A = 2. / (nobs - nbins - 1.) / (nobs - nbins - 4.)
        B = (nobs - nbins - 2.) / (nobs - nbins - 1.) / (nobs - nbins - 4.)
        return (1 + B * (nbins - nparams)) / (1 + A + B * (nparams + 1))

    def get_hartlap2007_factor(nobs, nbins):
        return (nobs - nbins - 2.) / (nobs - 1.)

    factor *= get_percival2014_factor(nobs, nbins, 7) / get_hartlap2007_factor(nobs, nbins)
    covariance_rotated = covariance_rotated.clone(value=factor * covariance_rotated.value())

    if dataset == 'spectrum-poles-rotated+bao-recon':
        fn = dr_base_dir / f'data/likelihood/likelihood_bao-recon_stat-only_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
        covariance_bao = types.read(fn).covariance

        def covariance_to_std_corrcoef(covariance):
            std = np.diag(covariance)**0.5
            corrcoef = covariance / (std[..., None] * std)
            return std, corrcoef

        # Decompose BAO covariance into standard deviation and correlation matrix
        std_bao, corr_bao = covariance_to_std_corrcoef(covariance_bao.value())
        # Decompose full EZmock-based covariance into standard deviation and correlation matrix
        std_full, corr_full = covariance_to_std_corrcoef(covariance_rotated.value())
        index_bao = np.arange(covariance_rotated.shape[0] - covariance_bao.shape[0], covariance_rotated.shape[0])
        # Replace BAO part
        std_full[index_bao] = std_bao
        corr_full[np.ix_(index_bao, index_bao)] = corr_bao
        covariance_rotated = covariance_rotated.clone(value=corr_full * std_full[..., None] * std_full)
        covariance_rotated_spectrum = covariance_rotated.at.observable.get('spectrum')
    else:
        covariance_rotated_spectrum = covariance_rotated
    observable_spectrum = covariance_rotated_spectrum.observable

    # Systematics
    # HOD systematics
    fn = dr_base_dir / f'data/covariance/syst/covariance_hod_spectrum-poles-rotated_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    covariance_hod = types.read(fn).at.observable.match(observable_spectrum)  # match to the cut power spectrum
    # Photometric systematics (non-zero only for ELG and QSO)
    fn = dr_base_dir / f'data/templates_spectrum/template_photo_spectrum-poles-rotated_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
    template = types.read(fn).match(observable_spectrum)  # match to the cut power spectrum
    value = np.diag(np.atleast_1d(template.attrs['prior_variance'])) * template.value()[:, None] * template.value()
    covariance_photo = covariance_rotated_spectrum.clone(value=value)
    # Rotation systematics
    # sigma^2 * template * template.T
    rotation_marg = sum(s.value()**2 * mo.value()[:, None] * mo.value() for s, mo in zip(rotation['s'], rotation['mo']))
    # Create "rotation" covariance matrix
    covariance_rotation = types.CovarianceMatrix(observable=rotation['M'].observable, value=rotation_marg)
    # Match to the cut power spectrum
    covariance_rotation = covariance_rotation.at.observable.match(observable_spectrum)

    # Add all systematic contributions together
    covariance_syst = covariance_rotation.value() + covariance_hod.value() + covariance_photo.value()
    if dataset == 'spectrum-poles-rotated+bao-recon':
        # For the BAO:
        covsyst_qisoqap = np.diag([0.245, 0.3])**2  # eq. 5.1 of https://fr.overleaf.com/project/645d2ce132ee6c4f6baa0ddd
        iso = covariance_rotated.observable.get('baorecon').size == 1
        if iso:
            covariance_syst_bao = covsyst_qisoqap[:1, :1]
        else:
            assert covariance_rotated.observable.get('baorecon').parameters == ['qpar', 'qper']
            eta = 1. / 3.
            jac = np.array([[1., 1. - eta], [1., - eta]])  # ('qisoqap' -> 'qparqper')
            covariance_syst_bao = jac.dot(covsyst_qisoqap).dot(jac.T)
        covariance_syst_bao *= 1e-4  # unit is percent
        covariance_syst = linalg.block_diag(covariance_syst, covariance_syst_bao)

    covariance_with_systematics = covariance_rotated.clone(value=covariance_rotated.value() + covariance_syst)
    if check:
        fn = dr_base_dir / f'data/likelihood/likelihood_{dataset}_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
        assert np.allclose(types.read(fn).covariance.value(), covariance_with_systematics.value())
    return covariance_with_systematics


if __name__ == '__main__':
    list_zrange = [('BGS_BRIGHT-21.5', (0.1, 0.4)),
                    ('LRG', (0.4, 0.6)),
                    ('LRG', (0.6, 0.8)),
                    ('LRG', (0.8, 1.1)),
                    ('ELG_LOPnotqso', (1.1, 1.6)),
                    ('QSO', (0.8, 2.1))]
    check = True
    for tracer, zrange in list_zrange:
        for dataset in ['spectrum-poles-rotated', 'spectrum-poles-rotated+bao-recon']:
            print(tracer, zrange)
            observable = get_observable(tracer, zrange, dataset=dataset, check=check)
            window = get_window_matrix(tracer, zrange, dataset=dataset, check=check)
            covariance = get_covariance_matrix(tracer, zrange, dataset=dataset, check=check)
            # Apply scale cuts of covariance to observable and window
            observable = observable.match(covariance.observable)
            window = window.at.observable.match(covariance.observable)
            likelihood = types.GaussianLikelihood(observable=observable, window=window, covariance=covariance)
            if check:
                fn = dr_base_dir / f'data/likelihood/likelihood_{dataset}_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                likelihood_ref = types.read(fn)
                assert np.allclose(likelihood.observable.value(), likelihood_ref.observable.value())
                assert np.allclose(likelihood.window.value(), likelihood_ref.window.value())
                assert np.allclose(likelihood.covariance.value(), likelihood_ref.covariance.value())
                