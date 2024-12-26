from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
setup(
name="embedding-selection-framework",
version="0.1.0",
author="Your Name",
author_email="your.email@example.com",
description="A framework for evaluating and selecting optimal embedding models for various NLP tasks",
long_description=long_description,
long_description_content_type="text/markdown",
url="https://github.com/yourusername/embedding-selection-framework",
packages=find_packages(),
classifiers=[
"Development Status :: 3 - Alpha",
"Intended Audience :: Science/Research",
"License :: OSI Approved :: MIT License",
"Operating System :: OS Independent",
"Programming Language :: Python :: 3",
"Programming Language :: Python :: 3.8",
"Programming Language :: Python :: 3.9",
"Programming Language :: Python :: 3.10",
"Topic :: Scientific/Engineering :: Artificial Intelligence",
],
python_requires=">=3.8",
install_requires=requirements,
include_package_data=True,
keywords="embeddings, nlp, machine learning, deep learning, text analysis",
)
