#!/usr/bin/env python3
"""Setup script for Turkish License Plate Reader."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = [
    'opencv-python>=4.8.0',
    'numpy>=1.24.0',
    'ultralytics>=8.0.0',
    'transformers>=4.30.0',
    'torch>=2.0.0',
    'colorlog>=6.7.0',
    'pyyaml>=6.0',
    'fastapi>=0.104.0',
    'uvicorn[standard]>=0.24.0',
    'pydantic>=2.0.0',
    'pydantic-settings>=2.0.0',
    'python-multipart>=0.0.6',
    'requests>=2.31.0',
]

dev_requirements = [
    'pytest>=7.4.0',
    'pytest-cov>=4.1.0',
    'black>=23.7.0',
    'flake8>=6.1.0',
    'mypy>=1.5.0',
]

setup(
    name='eplatereader',
    version='1.0.0',
    description='Turkish License Plate Recognition System with YOLO and Qwen3-VL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ePlateReader',
    license='MIT',
    
    # Package configuration
    packages=find_packages(exclude=['tests', 'tests.*', 'venv', 'venv.*']),
    include_package_data=True,
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'eplatereader=main:main',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    # Keywords
    keywords='license-plate-recognition ocr yolo computer-vision turkish-plates',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/ePlateReader/issues',
        'Source': 'https://github.com/yourusername/ePlateReader',
    },
)
