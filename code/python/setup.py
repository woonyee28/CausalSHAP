# setup.py
from setuptools import setup, find_packages

setup(
    name='causal_shap',  # Use underscores instead of hyphens
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'shap',
        'joblib',
        'matplotlib',
        'networkx',
        'causal-learn',
    ],
    entry_points={
        'console_scripts': [
            'run-causal-shap=causal_shap.main:main',  # Update module path
        ],
    },
)
