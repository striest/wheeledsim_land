from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['wheeledsim_land', 'wheeledsim_land.policies', 'wheeledsim_land.replay_buffers', 'wheeledsim_land.trainers', 'wheeledsim_land.networks', 'wheeledsim_land.util', 'wheeledsim_land.networks.cnn_blocks', 'wheeledsim_land.managers', 'wheeledsim_land.data_augmentation'],
  package_dir={'': 'src'}
)

setup(**d)
