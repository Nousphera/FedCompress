"""FedCompress package metadata; install with requirements.txt."""

from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="fedcompress",
    version="0.2.0",
    description="Communication-efficient federated learning through adaptive weight clustering and server-side distillation",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.11,<3.14",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "torch==2.8.0",
        "torchvision==0.23.0",
        "flwr[simulation]==1.40.0",
        "numpy>=2,<3",
        "soundfile>=0.13,<1",
        "pillow>=11,<13",
        "tomli-w>=1,<2",
    ],
    license_files=["licenses/THIRD_PARTY.txt"],
    entry_points={"console_scripts": ["fedcompress=fedcompress.experiment:main"]},
)
