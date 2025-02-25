from setuptools import find_packages, setup

setup(
    name="jaxvmas",
    version="0.0.1",
    url="https://github.com/jselvaraaj/JaxVMAS",
    license="GPLv3",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxtyping",
        "flax",
        "chex",
        "beartype",
        "equinox",
        "pyglet<=1.5.27",
        "opencv-python",
    ],
)
