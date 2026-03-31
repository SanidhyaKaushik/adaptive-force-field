from setuptools import setup, find_packages

setup(
    name="adaptive-force-field",
    version="0.1.0",
    author="Sanidhya Kaushik",
    description="Adaptive Force Field Evaluation using Deep ResNets and Conformal Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SanidhyaKaushik/adaptive-force-field",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
)