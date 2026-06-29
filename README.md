# BananaForge

🎨 **AI-powered multi-layer 3D printing optimization tool** that converts 2D images into optimized multi-layer 3D models for color printing with advanced transparency mixing.

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

## Current Status

BananaForge is currently an alpha-stage project. Core STL, alpha-aware STL, 3MF, Bambu-compatible 3MF, instruction, and cost-report export paths are under active test coverage, while filament TD/HueForge-library compatibility is intentionally out of scope for now.

## ✨ What Makes BananaForge Special

BananaForge uses AI optimization to create multi-color 3D prints with fewer material swaps and practical export artifacts:

- 🧠 **AI-Powered Optimization**: PyTorch-based differentiable optimization with Gumbel softmax sampling
- 🌈 **Advanced Transparency Mixing**: Create more colors with fewer materials through strategic layer transparency
- 🎯 **Intelligent Material Selection**: LAB color space optimization for perceptual accuracy
- ⚡ **GPU Acceleration**: CUDA and MPS support for fast processing
- 📊 **Print-Oriented Output**: STL files, 3MF/Bambu packages, HueForge projects, and detailed cost analysis

## 🚀 Quick Start

```bash
# Install BananaForge
pip install -e .

# Convert your first image
bananaforge convert photo.jpg --materials materials.csv

# With transparency mixing for fewer material swaps
bananaforge convert photo.jpg --enable-transparency --materials materials.csv --max-materials 6
```

## 🎨 Advanced Transparency Features

BananaForge introduces **transparency-based color mixing** that revolutionizes multi-color 3D printing:

### Three-Layer Opacity Model
- **33% opacity**: Light transparency for subtle color mixing
- **67% opacity**: Medium transparency for gradient effects  
- **100% opacity**: Full color for vibrant base layers

### Smart Material Savings
- **30%+ reduction** in material swaps
- **Intelligent base layer** optimization for maximum contrast
- **Gradient detection** for smooth color transitions
- **Cost analysis** with detailed savings reports

```bash
# Enable transparency features with full options
bananaforge convert image.jpg \
  --enable-transparency \
  --random-seed 123 \
  --opacity-levels "0.33,0.67,1.0" \
  --optimize-base-layers \
  --enable-gradients \
  --materials materials.csv \
  --max-materials 6 \
  --max-layers 25 \
  --mixed-precision \
  --export-format "stl,instructions,cost_report,transparency_analysis" \
  --output ./transparent_model/
```

## 🛠 Installation

### Development Installation (Current)
```bash
git clone https://github.com/eddieoz/BananaForge.git
cd BananaForge
pip install -e .[dev]
```

### Verify Installation
```bash
bananaforge version
bananaforge --help
```

### Example Configuration
```bash
bananaforge convert photo.jpg --config config.example.json
```

If no material file is provided, BananaForge uses its built-in Bambu Lab Basic PLA material set.

### Docker
```bash
docker build -t bananaforge .
docker run --rm -v "$PWD:/workspace" bananaforge convert photo.jpg --config config.example.json
```

## 🏗 Architecture Overview

### Core Components
- **Enhanced Optimization Engine**: Discrete validation, learning rate scheduling, mixed precision
- **Advanced Image Processing**: LAB color space, saturation enhancement, color-preserving resize
- **Intelligent Height Map System**: Two-stage K-means clustering, multi-threaded initialization
- **Transparency Mixing System**: Physics-based alpha compositing, gradient processing
- **Professional Output**: STL with alpha support, HueForge projects, detailed analytics

### Key Technologies
- **PyTorch**: Differentiable optimization with automatic mixed precision
- **LAB Color Space**: Perceptually uniform color calculations
- **Gumbel Softmax**: Discrete optimization with gradient flow
- **Multi-threading**: Parallel processing for complex operations

## 📚 Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get started in 5 minutes
- **[Materials Guide](docs/materials.md)** - Managing filaments and color matching
- **[CLI Reference](docs/cli-reference.md)** - Complete command reference
- **[API Reference](docs/api-reference.md)** - Python programming interface
- **[Configuration](docs/configuration.md)** - Advanced settings and workflows
- **[Examples](docs/examples/README.md)** - Real-world usage examples

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=bananaforge --cov-report=html

# Run specific feature tests
pytest tests/test_feature4_5_transparency_color_mixing.py -v
```

## 🤝 Contributing

We welcome contributions! This project follows TDD/BDD development practices.

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Write tests first**: Follow our BDD scenarios in `tests/`
4. **Implement features**: Make tests pass
5. **Submit a pull request**

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with ❤️ using PyTorch and modern AI techniques
- Inspired by the 3D printing and computer vision communities
- Special thanks to HueForge and Autoforge for pioneering multi-color 3D printing workflows

## Buy me a coffee
Did you like it? [Buy me a coffee](https://www.buymeacoffee.com/eddieoz)

[![Buy me a coffee](https://ipfs.io/ipfs/QmR6W4L3XiozMQc3EjfFeqSkcbu3cWnhZBn38z2W2FuTMZ?filename=buymeacoffee.webp)](https://www.buymeacoffee.com/eddieoz)

Or drop me a tip through Lightning Network: ⚡ [getalby.com/p/eddieoz](https://getalby.com/p/eddieoz)
