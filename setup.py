from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

test_deps = [
    'pytest',
    'pytest-cov',
    'mypy']

setup(
    name='agape',
    version='0.1',
    description='Automatic Gene Ontology annotation prediction in S. pombe',
    author='Harry Scholes',
    url='https://github.com/harryscholes/agape',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Bioinformaticians',
        'Topic :: Bioinformatics :: Gene Function Prediction',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['docs']),
    install_requires=[
        'numpy>=1.14',
        'scipy>=1.0',
        'pandas>=0.22',
        'scikit-learn>=0.19',
        'networkx>=2.1',
        'biopython>=1.70',
        'goatools>=0.8',
        'matplotlib>=2.2.0',
        'keras>=2.1.5',
        'tensorflow>=1.7.0'],
    tests_require=test_deps,
    setup_requires=[
        'pytest-runner'],
    extras_require={
        "test": test_deps
    },
)
