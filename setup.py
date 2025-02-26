from setuptools import find_packages, setup

setup(
    name="jaxvmas",
    version="alpha",
    url="https://github.com/jselvaraaj/JaxVMAS",
    license="GPLv3",
    packages=find_packages(),
    install_requires=["jax"],
)
