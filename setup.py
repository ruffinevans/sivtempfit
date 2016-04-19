from setuptools import setup

setup(name='sivtempfit',
      version='0.1',
      description='Bayesian analysis of SiV spectra for temperature fitting',
      url='https://github.com/p201-sp2016/SiVTempFit-RE',
      author='Ruffin Evans',
      author_email='ruffinevans@gmail.com',
      license='GPLv3',
      packages=['sivtempfit'],
      # All dependencies should be here.
      # If dependencies are not on PyPI, use URL. See:
      # https://python-packaging.readthedocs.org/en/latest/dependencies.html
      install_requires=[
          'numpy',
          'scipy',
          'emcee',
          'seaborn',
          'matplotlib',
          'pandas'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
