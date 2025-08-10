# Install

### GPU (recommend if available)

1. Preinstall the matching CUDA wheels
`pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1`

2. Install the soccer_cv library from GitHub
`pip install "git+https://github.com/granthohol/soccer-cv.git@main" `

### CPU

1. Preinstall the CPU PyTorch stack (avoiding large CUDA downloads)
`pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1`

2. Install the soccer_cv library from GitHub
`pip install "git+https://github.com/granthohol/soccer-cv.git@main `



