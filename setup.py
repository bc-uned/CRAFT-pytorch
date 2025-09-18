from setuptools import setup, find_packages

setup(
    name="craft-pytorch",
    version="1.0.0",
    description="CRAFT: Character Region Awareness for Text detection (PyTorch)",
    author="NAVER Corp.",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch==2.8.0",
        "torchvision==0.18.0",
        "opencv-python==4.12.0.0",
        "scikit-image==0.23.2",
        "scipy==1.13.0"
    ],
    python_requires=">=3.8",
)