from setuptools import setup, find_packages

setup(
    name="model_tuner",
    version="0.0.13a",
    author="UCLA CTSI ML Team: Leonid Shpaner, Arthur Funnell, Panayiotis Petousis",
    author_email="pp89@ucla.edu",
    description="A Python library for tuning machine learning models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # Type of the long description
    package_dir={"": "src"},  # Directory where your package files are located
    # Automatically find packages in the specified directory
    packages=find_packages(where="src"),
    project_urls={  # Optional
        # "Author Website": "https://www.leonshpaner.com",
        "Documentation": "https://uclamii.github.io/model_tuner",
        "Zenodo Archive": "https://zenodo.org/doi/10.5281/zenodo.12727322",
        "Source Code": "https://github.com/uclamii/model_tuner/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: Apache License, Version 2.0 (Apache-2.0)",
        "Operating System :: OS Independent",
    ],  # Classifiers for the package
    python_requires=">=3.7",  # Minimum version of Python required
    install_requires=[
        "joblib>=1.3.2",
        "numpy>=1.21.6",
        "pandas>=1.3.5",
        "scikit-learn>=1.0.2",
        "scipy>=1.7.3",
        "tqdm>=4.66.4",
    ],
)
