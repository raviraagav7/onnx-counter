#!/usr/bin/env python
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

readme = open("README.md").read()

VERSION = "0.0.1"

setup(
    # Metadata
    name="THOP",
    version=VERSION,
    author="RaviRaagav, Sanghoon Kwak",
    author_email="raviraagavsr@micron.com, skwak@micron.com",
    url="https://bitbucket.micron.com/bbdc/scm/mdlaml/onnx-opcounter.git",
    description="A tool to count the FLOPs of Onnx model.",
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(exclude=("*test*", "*xlsx")),
    #
    zip_safe=True,
    install_requires=required,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
