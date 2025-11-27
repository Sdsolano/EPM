"""
Setup script para el Sistema de Pronóstico Automatizado de Demanda Energética - EPM
Permite instalar el paquete con: pip install -e .
"""
from setuptools import setup, find_packages
from pathlib import Path

# Leer README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Leer requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').split('\n')
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="epm-forecast-system",
    version="1.0.0",
    author="EPM Development Team",
    author_email="desarrollo@epm.com.co",
    description="Sistema inteligente de pronóstico de demanda energética con autoaprendizaje",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/epm/forecast-system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "pydantic>=2.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
        ]
    },
    entry_points={
        "console_scripts": [
            "epm-pipeline=scripts.run_pipeline:main",
            "epm-train=scripts.train_models:main",
            "epm-predict=scripts.predict_30_days:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
)
