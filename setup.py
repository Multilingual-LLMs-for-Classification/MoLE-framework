"""
pip-installable setup for the moe-classifier SDK and its core engine.

Install (editable, from the moe-classification-service/ directory)::

    pip install -e .

This installs both packages:
  - moe_classifier   — the clean public SDK
  - moe_router       — the underlying MOE engine (also usable directly)

The FastAPI web app (app/) is NOT included — it is a separate runtime concern.
"""

from setuptools import find_packages, setup

setup(
    name="moe-classifier",
    version="1.0.0",
    description="Multilingual Mixture-of-Experts text classification library",
    long_description=(
        "A Python SDK that wraps a hierarchical MOE classification pipeline: "
        "Language Detection → Domain Classification → Task Routing → Expert Execution. "
        "Supports 176 languages and multiple domain/task combinations via LoRA-adapted LLMs."
    ),
    python_requires=">=3.10",
    packages=find_packages(
        include=[
            "moe_classifier",
            "moe_classifier.*",
            "moe_router",
            "moe_router.*",
        ]
    ),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "fasttext-wheel>=0.9.2",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        # Install the full web service dependencies as well
        "server": [
            "fastapi>=0.109.0",
            "uvicorn[standard]>=0.27.0",
            "pydantic>=2.0.0",
            "pydantic-settings>=2.0.0",
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
            "python-multipart>=0.0.6",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
