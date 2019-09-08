import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymirc",
    use_scm_version=True,
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
