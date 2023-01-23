from setuptools import find_packages, setup

setup(
    name="yaslp",
    version="0.0.2",
    description="Yet Another Shepp-Logan Phantom, this time using jax.numpy",
    author="Renat Sibgatulin",
    author_email="sibgatulin@tuta.io",
    license="MIT",
    packages=find_packages(),
    install_requires=["jax"],
)
