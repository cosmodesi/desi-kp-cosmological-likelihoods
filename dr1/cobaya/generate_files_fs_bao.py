import os
import yaml


def load(fn):
    with open(fn, 'r') as file:
        return yaml.load(file.read(), Loader=yaml.SafeLoader)


def dump(data, fn):

    def list_rep(dumper, data):
        return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, list_rep)

    with open(fn, 'w') as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Y1 FS cosmological likelihoods')
    parser.add_argument('--clean', action='store_true', default=False, help='remove generated likelihoods')
    args = parser.parse_args()

    config = load('desi_fs_bao_all.yaml')
    tracers = {'all': ['BGS_z0', 'LRG_z0', 'LRG_z1', 'LRG_z2', 'ELG_z1', 'QSO_z0', 'Lya_z0'], 'all_nolya': ['BGS_z0', 'LRG_z0', 'LRG_z1', 'LRG_z2', 'ELG_z1', 'QSO_z0'], 'bgs': ['BGS_z0'], 'lrg': ['LRG_z0', 'LRG_z1', 'LRG_z2'], 'lrg_z0': ['LRG_z0'], 'lrg_z1': ['LRG_z1'], 'lrg_z2': ['LRG_z2'], 'elg': ['ELG_z1'], 'qso': ['QSO_z0'], 'lya': ['Lya_z0']}
    for tracer, namespaces in tracers.items():
        for observable in ['fs'] + (['fs_bao'] if tracer != 'all' else []):
            basename = f'desi_{observable}_{tracer}'
            if args.clean:
                os.remove(basename + '.yaml')
                os.remove(basename + '.py')
                continue
            config_tracer = dict(config)
            config_tracer['tracers'] = [name.lower() for name in namespaces]
            if observable == 'fs' and tracer == 'all':  # remove Lya
                config_tracer['tracers'] = config_tracer['tracers'][:-1]
            config_tracer['params'] = {key: value for key, value in config['params'].items() if any(name in key.lower() for name in config_tracer['tracers'])}
            config_tracer['observable_name'] = {'fs': 'spectrum-poles-rotated', 'fs_bao': 'spectrum-poles-rotated+bao-recon'}[observable]
            dump(config_tracer, basename + '.yaml')
            py_template = '''from .desi_fs_bao_all import *


class {name}(desi_fs_bao_all):
    r"""
    DESI FS (optionally with post-recon BAO) likelihood for {tracer}.
    """
            '''
            with open(basename + '.py', 'w') as file:
                file.write(py_template.format(name=basename, tracer=tracer))
