from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

description = "2D polynomials"

setup(name="poly2d",
      description=description,
      author="Panagiotis Zestanakis",
      author_email="panosz@gmail.com",
      packages=find_packages(),
      version='0.0.1',
      install_requires=['numpy',
                        'scipy',
                        ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
      ],
      long_description=long_description,
      long_description_content_type="text/markdown",
      python_requires='>=3.7',
      )
