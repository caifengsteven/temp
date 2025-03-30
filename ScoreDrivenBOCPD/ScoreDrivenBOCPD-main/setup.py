from setuptools import setup, find_packages
from codecs import open
from os import path

__author__ = "Yvonni Tsaknaki"
__license__ = "MIT License"
__email__ = "mathematyvian@gmail.com"


here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="ScoreDrivenBOCPD",
    version="0.1.8",
    license="MIT License",
    description="A Score-Driven Bayesian Online Change-Point Detection Package",
    url="https://gitlab.com/YvTsak/ScoreDrivenBOCPD",
    author="Yvonni Tsaknaki",
    author_email="mathematyvian@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    keywords="Regime switching models, Bayesian methods, Online learning algorithm",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    setup_requires=['wheel']
    )
