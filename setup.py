# !/usr/bin/env python
# -*- encoding: utf-8 -*-
from setuptools import setup
import codecs
import io
import re
from glob import glob
from os import path
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

wrkDir = path.abspath(path.dirname(__file__))

with open(path.join(wrkDir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read_file(filename, cb):
    with codecs.open(filename, 'r', 'utf8') as f:
        return cb(f)

setup(
    name='geotime_classify',
    version='0.8.2',
    license='LGPL-3.0-or-later',
    description='Categorizes spatial and temporal columns for csv files. Standardizes date columns for transformations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kyle Marsh',
    author_email='kyle@jataware.com',
    url='https://github.com/jataware/geotime_classify',
    packages=find_packages(),
    package_dir={'geotime_classify': 'geotime_classify'},
    package_data={'geotime_classify': ['models/*','datasets/*']},
    py_modules=['geotime_schema'],
    # include_package_data=True,
    zip_safe=False,

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
    project_urls={

        'Changelog': 'https://github.com/jataware/geotime_classify/blob/master/CHANGELOG.md',
        'Issue Tracker': 'https://github.com/jataware/geotime_classify/issues',
    },
    keywords=['LSTM', 'RNN', 'Classification', 'Date','Datetime', 'Coordinates', 'Latitude', 'Longitude'],

    python_requires='>=3.6',

    install_requires=read_file('requirements.txt', lambda f: list(
            filter(bool, map(str.strip, f)))),


    extras_require={"dev": ["pytest>=3.7", "twine>=3.3.0"]},
    setup_requires=[
        'pytest-runner',
    ],

)

