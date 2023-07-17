from setuptools import setup, find_packages

setup(
    name="abt_ds",
    packages=find_packages(
        include=["abt_ds"]
    ),
    include_package_data=True,
    python_requires=">=3.10",
)
