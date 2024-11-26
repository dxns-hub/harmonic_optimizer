from setuptools import setup, find_packages

setup(
    name="harmonic-optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
    ],
    extras_require={
        'medical': ['pydicom>=2.2.0'],
        'mechanical': ['control>=0.9.0'],
        'electrical': ['scikit-dsp-comm>=2.0.0'],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A harmonic optimization algorithm for signal processing and system optimization",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/harmonic-optimizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires='>=3.8',
)
