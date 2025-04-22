from setuptools import setup, find_packages

setup(
    name="pii_service",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "presidio-analyzer",
        "presidio-anonymizer",
        "spacy",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pycountry",
        "langdetect",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A service for detecting and analyzing Personally Identifiable Information (PII)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pii-service",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 