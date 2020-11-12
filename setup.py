from setuptools import setup, find_packages

VERSION = '2.0rc1'
with open('requirements.txt') as f:
      install_reqs = f.read().splitlines()

setup(name='deepcoil',
      version=VERSION,
      description='Fast and accurate prediction of coiled coil domains in protein sequences.',
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
