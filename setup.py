from setuptools import find_packages, setup

setup(
    name="sygnals",
    version="0.1.0",
    description="A versatile CLI for signal and audio processing.",
    author="Araray Velho",
    author_email="araray@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "pandasql",
        "librosa",
        "soundfile",
        "matplotlib",
        "click",
        "tabulate",
        "pywavelets",
    ],
    entry_points={
        "console_scripts": [
            "sygnals=sygnals.cli:cli",
        ],
    },
)
