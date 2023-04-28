from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='ACSM as a package',
    name='acsm',
    packages=find_packages()
)