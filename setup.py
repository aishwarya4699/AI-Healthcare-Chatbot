from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Aishwarya Pramod",
    author_email="aishwarya.1999@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
)