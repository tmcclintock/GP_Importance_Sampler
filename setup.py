import sys, os, glob
from setuptools import setup, Extension
import subprocess

dist = setup(name="importance_sampler",
             author="Tom McClintock",
             author_email="mcclintock@bnl.gov",
             description="Tool for importance sampling.",
             license="GNU General Public License v2.0",
             url="https://github.com/tmcclintock/GP_Importance_Sampler",
             packages=['importance_sampler'],
             install_requires=['cffi','numpy','scipy','george'],
             setup_requires=['pytest_runner'],
             tests_require=['pytest'])
