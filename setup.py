import setuptools

setuptools.setup(
    name="pymirc",
    version="0.24.0",
    author="Georg Schramm, Tom Eelbode, Jeroen Bertels",
    author_email="georg.schramm@kuleuven.be",
    description="Python imaging utilities developed in the medical imaging research center of KU Leuven",
    long_description="Python imaging utilities developed in the medical imaging research center of KU Leuven",
    license='LGPL v3',
    long_description_content_type="text/markdown",
    url="https://github.com/gschramm/pymirc",
    packages=setuptools.find_packages(exclude = ["data","examples","tutorial"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.9',
    install_requires=['numpy>=1.15',
                      'scipy>=1.1',
                      'matplotlib>=2.2.2',
                      'pydicom>=1.1',
                      'scikit-image>=0.14',
                      'numba>=0.39',
                      'nibabel>=3.0'],
)
