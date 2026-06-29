# Installation Guide

This guide covers installing BananaForge on different operating systems and configurations.

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **GPU**: Optional but recommended for faster processing

### Recommended Requirements
- **Python**: 3.11 or higher
- **RAM**: 16GB or more
- **Storage**: 5GB free space
- **GPU**: NVIDIA GPU with CUDA support or Apple Silicon with MPS

## Quick Installation

### 1. Install BananaForge

```bash
pip install bananaforge
```

### 2. Verify Installation

```bash
bananaforge --version
# Should show: BananaForge Version: 0.1.0
```

### 3. Test Basic Functionality

```bash
bananaforge convert --help
```

## Detailed Installation

### Windows

#### Method 1: Using pip (Recommended)

1. **Install Python 3.9+**:
   - Download from [python.org](https://python.org)
   - Ensure "Add Python to PATH" is checked during installation

2. **Install BananaForge**:
   ```cmd
   pip install bananaforge
   ```

#### Method 2: Using Conda

```bash
conda create -n bananaforge python=3.11
conda activate bananaforge
pip install bananaforge
```

### macOS

#### Method 1: Using pip

1. **Install Python 3.9+**:
   ```bash
   brew install python@3.11
   ```

2. **Install BananaForge**:
   ```bash
   pip install bananaforge
   ```

#### Method 2: Using Conda

```bash
conda create -n bananaforge python=3.11
conda activate bananaforge
pip install bananaforge
```

### Linux (Ubuntu/Debian)

#### Method 1: Using pip

1. **Install Python 3.9+**:
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-pip python3.11-venv
   ```

2. **Install BananaForge**:
   ```bash
   pip install bananaforge
   ```

#### Method 2: Using Virtual Environment

```bash
python3.11 -m venv bananaforge_env
source bananaforge_env/bin/activate
pip install bananaforge
```

## Development Installation

For developers who want to modify BananaForge:

```bash
# Clone the repository
git clone https://github.com/eddieoz/BananaForge.git
cd bananaforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

## GPU Support

### NVIDIA CUDA

1. **Install CUDA Toolkit** (version 11.8 or higher)
2. **Install PyTorch with CUDA**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Install BananaForge**:
   ```bash
   pip install bananaforge
   ```

### Apple Silicon (M1/M2)

1. **Install PyTorch with MPS**:
   ```bash
   pip install torch torchvision
   ```
2. **Install BananaForge**:
   ```bash
   pip install bananaforge
   ```

### Verify GPU Support

```bash
# Test CUDA
bananaforge convert test.jpg --device cuda

# Test MPS (Apple Silicon)
bananaforge convert test.jpg --device mps

# Test CPU
bananaforge convert test.jpg --device cpu
```

## Dependencies

BananaForge automatically installs these key dependencies:

- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computing
- **Pillow**: Image handling
- **SciPy**: Scientific computing
- **Trimesh**: 3D mesh processing
- **Click**: Command-line interface
- **Rich**: Terminal formatting
- **Pydantic**: Data validation

## Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version

# Should be 3.9 or higher
```

#### Permission Issues
```bash
# Use user installation
pip install --user bananaforge

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install bananaforge
```

#### GPU Issues
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Use lower resolution
bananaforge convert image.jpg --resolution 128

# Use CPU instead of GPU
bananaforge convert image.jpg --device cpu
```

## Next Steps

After installation:

1. **[Quick Start](quickstart.md)** - Your first conversion
2. **[Configuration](configuration.md)** - Set up your preferences
3. **[Materials Guide](materials.md)** - Learn about material management

## Support

If you encounter issues:

- Check the **[Troubleshooting Guide](troubleshooting.md)**
- Search **[GitHub Issues](https://github.com/eddieoz/BananaForge/issues)**
- Ask in **[GitHub Discussions](https://github.com/eddieoz/BananaForge/discussions)**
