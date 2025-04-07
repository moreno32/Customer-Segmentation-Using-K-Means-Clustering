"""
Setup file for the E-commerce Recommendation System project.
"""

from setuptools import setup, find_packages

setup(
    name="ecommerce_recommendation_system",
    version="0.1.0",
    description="Customer segmentation and product recommendation system for e-commerce platforms",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.25.0",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "python-dateutil>=2.8.2",
        "joblib>=1.3.1",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
