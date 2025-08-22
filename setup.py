from setuptools import setup, find_packages

setup(
    name='voice-dataset-evaluator',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A framework for evaluating and analyzing voice datasets for ASR suitability.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pydub',
        'librosa',
        'pyannote.audio',
        'torch',  # if using Whisper or similar models
        'transformers',  # if using zero-shot classification
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)