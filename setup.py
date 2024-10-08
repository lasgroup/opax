#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'flax==0.6.8',
    'jax==0.4.13',
    'jaxlib==0.4.13',
    'matplotlib>=3.5.1',
    'numpy==1.24.2',
    'optax==0.1.4',
    'scipy<=1.10.1',
    'wandb>=0.12.11',
    'termcolor>=1.1.0',
    'distrax==0.1.2',
    'tensorflow_probability==0.19.0',
    'gym==0.26.1',
    'argparse-dataclass>=0.2.1',
    'tqdm',
    'seaborn',
    'cloudpickle',
    'pandas',
    'jaxutils==0.0.8',
    'dm_control==1.0.10',
    'trajax @ git+ssh://git@github.com/lenarttreven/trajax.git',
    'jaxtyping~=0.2.28',
    'moviepy',
    'mujoco<=2.3.7',
    'pyelastica==0.3.2',
    'pyglet==2.0.15',
    'vapory==0.1.2',
]

extras = {'dev': ['seaborn', 'control>=0.9.2']}
setup(
    name='opax',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.10',
    include_package_data=True,
    install_requires=required,
    extras_require=extras
)
