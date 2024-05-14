from setuptools import setup, find_packages

setup(
    name="model_tuner",
    version="0.0.1a",
    author="CTSI ML Team",
    author_email="pp89@ucla.edu",
    description="Python library for tuning ML models in healthcare.",
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
        # "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],  # Classifiers for the package
    python_requires=">=3.7",  # Minimum version of Python required
    install_requires=[
        "numpy>=1.18.5",  # Minimum version of numpy required
        "pandas>=1.0.5",  # Minimum version of pandas required
    ],
)