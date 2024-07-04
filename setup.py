from setuptools import setup, find_packages

setup(
    name="model_tuner",
    version="0.0.08a",
    author="UCLA CTSI ML Team: Leonid Shpaner, Arthur Funnell, Panayiotis Petousis",
    author_email="pp89@ucla.edu",
    description="A Python library for tuning Machine Learning models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # Type of the long description
    package_dir={"": "src"},  # Directory where your package files are located
    # Automatically find packages in the specified directory
    packages=find_packages(where="src"),
    project_urls={  # Optional
        # "Author Website": "https://www.leonshpaner.com",
        # "Documentation": "https://lshpaner.github.io/kfre_docs/",
        # "Zenodo Archive": "https://zenodo.org/records/11100222",
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
