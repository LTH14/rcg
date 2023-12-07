from setuptools import setup, find_packages

setup(
    name='rcg',
    version='0.0.1',
    description='Representation-Conditioned image Generation',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
