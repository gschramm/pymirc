import setuptools
import os
import subprocess
import warnings

# this gives only the correct path when using pip install -e
pkg_dir = os.path.abspath(os.path.dirname(__file__))

# in case the package is not a git repo but rather a release
# we try to get the fallback for package version from the dirname
tmp_split = pkg_dir.split('pymirc-')
if len(tmp_split) >= 2:
  fall_back_version = pkg_dir.split('pymirc-')[-1]
else:
  fall_back_version = 'unkown'

with open(os.path.join(pkg_dir,"README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymirc",
    use_scm_version={'fallback_version':fall_back_version},
    setup_requires=['setuptools_scm'],
    author="Georg Schramm, Tom Eelbode, Jeroen Bertels",
    author_email="georg.schramm@kuleuven.be",
    description="Python imaging utilities developed in the medical imaging research center of KU Leuven",
    license='LGPL v3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gschramm/pymirc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LGPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy>=1.15',
                      'scipy>=1.1',
                      'matplotlib>=2.2.2',
                      'pydicom>=1.1',
                      'scikit-image>=0.14',
                      'numba>=0.39'],
)
