import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('setting', [None, '0', '1'])
def test_equilibrium_backend_is_numpy_independent_of_jax_installation(setting):
    env = os.environ.copy()
    if setting is None:
        env.pop('HMFIT_USE_JAX', None)
    else:
        env['HMFIT_USE_JAX'] = setting
    code = '''
import numpy as np
from hmfit_core.utils.np_backend import xp, USE_JAX
from hmfit_core.utils.errors import _as_onp
from hmfit_core.solvers import NewtonRaphson
assert xp is np and not USE_JAX
assert 'jax' not in __import__('sys').modules
_as_onp([1.])
assert 'jax' not in __import__('sys').modules
s = NewtonRaphson(np.array([[.001, .001]]), np.array([[1,0,1],[0,1,1]]), nas=[], model_sett='Free')
assert isinstance(s.concentraciones(np.array([3.]))[0], np.ndarray)
'''
    subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=60)
