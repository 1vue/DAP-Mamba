# Installation

This page describes a typical environment setup for DAP-Mamba.

## Requirements

The code is designed for a CUDA-enabled PyTorch environment. The following versions are recommended:

- Python 3.10+
- CUDA-compatible PyTorch
- DeepSpeed
- Mamba CUDA extensions

Other versions may also work if PyTorch, CUDA, and the extension packages are mutually compatible.

## Create Environment

```bash
conda create -n dap_mamba python=3.10 -y
conda activate dap_mamba
```

## Install Dependencies

Install the basic Python dependencies:

```bash
pip install -r requirements.txt
```

Install additional runtime dependencies used by the training and inference scripts:

```bash
pip install deepspeed einops pycocotools h5py pandas safetensors
```

Install the Mamba-related packages:

```bash
pip install mamba-ssm causal-conv1d
```

If these packages fail to build from source, install wheel files that match your Python, PyTorch, CUDA, and operating system versions.

## Verify Installation

From the repository root, run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import deepspeed; print('deepspeed ok')"
python -c "import mamba_ssm; print('mamba ok')"
```

If CUDA is unavailable, check that the installed PyTorch package matches your local CUDA runtime.
