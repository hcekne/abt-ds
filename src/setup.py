from setuptools import setup, find_packages

setup(
    name="ds_abm",
    packages=find_packages(
        include=["ds_abm"]
    ),
    include_package_data=True,
    python_requires=">=3.10",
)
