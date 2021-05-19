# -*- coding: utf-8 -*-

"""Top-level package for Mixed Data Tools & Models."""

__author__ = """Aris Tritas"""
__email__ = "a.tritas@gmail.com"
__version__ = "0.0.1"


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import os
    import numpy as np
    import random

    # It could have been provided in the environment
    _random_seed = os.environ.get("RANDOM_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * (2 ** 31 - 1)
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
