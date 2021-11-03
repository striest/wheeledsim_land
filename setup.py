from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['wheeledsim_land'],
  package_dir={'': 'src'}
)

setup(**d)
