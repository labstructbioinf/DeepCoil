from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = '2.0'
with open('requirements.txt') as f:
      install_reqs = f.read().splitlines()

setup(name='deepcoil',
      version=VERSION,
      description='Fast and accurate prediction of coiled coil domains in protein sequences',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Jan Ludwiczak',
      author_email='j.ludwiczak@cent.uw.edu.pl',
      url='https://github.com/labstructbioinf/deepcoil',
      packages=find_packages(),
      install_requires=install_reqs,
      python_requires=">=3.6.1",
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      include_package_data=True,
      scripts=['bin/deepcoil'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
      ],
      )
