from setuptools import setup

setup(name='SiVTempFit',
      version='0.1',
      description='Bayesian analysis of SiV spectra for temperature fitting',
      url='https://github.com/p201-sp2016/SiVTempFit-RE',
      author='Ruffin Evans',
      author_email='ruffinevans@gmail.com',
      license='None',
      packages=['SiVTempFit'],
      # All dependencies should be here.
      # If dependencies are not on PyPI, use URL. See:
      # https://python-packaging.readthedocs.org/en/latest/dependencies.html
      install_requires=[
          'numpy',
          'scipy',
          'emcee',
          'seaborn'
      ],
      zip_safe=False)
