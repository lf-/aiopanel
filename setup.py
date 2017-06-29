import setuptools
import sys

import pip

pip_version = tuple([int(x) for x in pip.__version__.split('.')[:3]])
if pip_version < (9, 0, 1) :
    raise RuntimeError('Version of pip less than 9.0.1, breaking python ' \
                       'version requirement support')


setuptools.setup(
    name = 'aiopanel',
    version = '0.1',
    py_modules = ['aiopanel'],
    python_requires = '>=3.6',
    entry_points = {
        'console_scripts': [
            'aiopanel=aiopanel:entry_point'
        ]
    },

    install_requires = [
        'gbulb',
        'pydbus',
        'jinja2'
    ],

    extras_require = {
        'bspwm': ['aiobspwm']
    },

    author = 'lf',
    author_email = 'github@lfcode.ca',
    description = 'asyncio-based panel',
    license = 'MIT',
    keywords = 'asyncio panel'.split(' '),
    url = 'https://github.com/lf-/aiopanel'
)
