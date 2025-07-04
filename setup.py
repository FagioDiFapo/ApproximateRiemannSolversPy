from setuptools import setup, find_packages

setup(
    name="approximate_riemann_solvers",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
    ],
)