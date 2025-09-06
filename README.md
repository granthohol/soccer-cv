# Install

## Prereqs
- Python 3.10-3.12
- A clean virtual environment (recommended)
```bash
# pick one
python -m venv .venv && source .venv/bin/activate      # venv
# OR
conda create -n soccer-cv python=3.12 -y && conda activate soccer-cv
```
- Model access
```
# either login (stores a token)
huggingface-cli login
# or set an env var (CI-friendly)
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### GPU CUDA (recommend if available)

1. Preinstall the matching CUDA wheels
``` bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
```

2. Install the soccer_cv library from GitHub
```bash
pip install "git+https://github.com/granthohol/soccer-cv.git@main"
```

### MPS (Apple Silicon)


### CPU

1. Preinstall the CPU PyTorch stack (avoiding large CUDA downloads)
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
```

2. Install the soccer_cv library from GitHub
```bash
pip install "git+https://github.com/granthohol/soccer-cv.git@main"
```



