from pathlib import Path

from setuptools import setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="normuon",
    version="0.1.0",
    description="NorMuon optimizers.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Zichong Li",
    url="https://github.com/zichongli5/NorMuon",
    py_modules=["normuon", "ns_triton"],
    install_requires=["torch"],
    extras_require={"triton": ["triton>=3.0.0"]},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
