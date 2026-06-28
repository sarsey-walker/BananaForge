# 3MF Export Guide for BananaForge

## Overview

BananaForge now supports exporting optimized 3D models in the 3MF (3D Manufacturing Format) format, providing advanced multi-color printing capabilities with per-layer material assignment and full compatibility with modern slicers like Bambu Studio (EXPERIMENTAL).

## What is 3MF?

3MF (3D Manufacturing Format) is a modern 3D printing file format that addresses limitations of older formats like STL:

- **Lossless geometry**: Preserves exact model geometry without triangulation artifacts
- **Multi-material support**: Native support for multiple materials and colors
- **Layer information**: Can embed layer-specific material assignments
- **Metadata embedding**: Stores printing parameters, material properties, and manufacturing instructions
- **Industry standard**: Supported by major slicers (Bambu Studio (EXPERIMENTAL), PrusaSlicer, etc.)

## Key Features

BananaForge's 3MF export provides:

### ✅ Core Functionality
- **Per-layer material assignment**: Each layer can have different materials/colors
- **Material properties embedding**: Temperature, density, transparency settings
- **ZIP container structure**: Standard 3MF packaging with XML manifests
- **Validation and quality assurance**: Automatic file structure validation

### ✅ Advanced Features
- **Bambu Studio compatibility**: Optimized for Bambu Lab printers (EXPERIMENTAL)
- **Transparency support**: Alpha channel handling for transparent materials
- **Mesh optimization**: Efficient vertex and triangle generation from heightmaps
- **Progress reporting**: Real-time export progress feedback

### ✅ CLI Integration
- **Multiple export formats**: Export STL, 3MF, and instructions simultaneously
- **Configurable options**: Control compression, metadata, validation
- **Batch processing**: Process multiple images with consistent settings

## Usage Examples

### Basic 3MF Export

```bash
# Export to 3MF format
bananaforge convert input.jpg --export-format 3mf --output ./output/

# Export multiple formats including 3MF
bananaforge convert input.jpg --export-format stl,3mf,instructions --output ./output/
```

### Bambu Studio Compatible Export (EXPERIMENTAL)

```bash
# Export with Bambu Studio optimizations (EXPERIMENTAL)
bananaforge convert input.jpg \
  --export-format 3mf \
  --bambu-compatible \  # EXPERIMENTAL
  --include-3mf-metadata \
  --materials bambu_materials.csv \
  --output ./output/
```

### Advanced Configuration

```bash
# Full workflow with custom options
bananaforge convert input.jpg \
  --materials custom_filaments.csv \
  --max-layers 50 \
  --layer-height 0.2 \
  --export-format 3mf \
  --bambu-compatible \  # EXPERIMENTAL
  --include-3mf-metadata \
  --output ./output/
```

## CLI Options for 3MF Export

| Option | Description | Default |
|--------|-------------|---------|
| `--export-format` | Include `3mf` in format list | `stl,instructions` |
| `--bambu-compatible` | Enable Bambu Studio optimizations (EXPERIMENTAL) | `False` |
| `--include-3mf-metadata` | Include detailed metadata in 3MF | `True` |
| `--materials` | Material database CSV/JSON | Built-in materials |

## 3MF File Structure

BananaForge generates 3MF files with this structure:

```
model.3mf (ZIP archive)
├── [Content_Types].xml      # MIME type declarations
├── _rels/
│   └── .rels               # Package relationships
├── 3D/
│   └── 3dmodel.model       # Main 3D model with geometry and materials
└── Metadata/
    └── model_info.xml      # Export metadata and statistics
```

### Material Assignment

Materials are assigned per-layer in the 3D model:

```xml
<resources>
  <basematerials>
    <base name="Bambu PLA White" displaycolor="#FFFFFF"/>
    <base name="Bambu PLA Red" displaycolor="#FF0000"/>
  </basematerials>
  
  <!-- Geometry with per-layer material references -->
  <object id="1" type="model">
    <mesh>
      <vertices>...</vertices>
      <triangles>
        <!-- Layer 0: White material -->
        <triangle v1="0" v2="1" v3="2" pid="0"/>
        <!-- Layer 1: Red material -->
        <triangle v1="3" v2="4" v3="5" pid="1"/>
      </triangles>
    </mesh>
  </object>
</resources>
```

## Material Database Integration

### Using Built-in Materials

```bash
# Bambu Lab PLA materials
bananaforge convert input.jpg --export-format 3mf

# HueForge compatible materials  
bananaforge export-materials --format csv --preset hueforge --output hf_materials.csv
bananaforge convert input.jpg --materials hf_materials.csv --export-format 3mf
```

### Custom Material Properties

Create a custom materials CSV with these columns:

```csv
id,name,brand,color_hex,transparency,td,density,temperature,cost
pla_white,Basic PLA White,Generic,#FFFFFF,0.0,4.0,1.24,210,25.0
pla_red,Basic PLA Red,Generic,#FF0000,0.0,4.0,1.24,210,25.0
pla_clear,Clear PLA,Generic,#FFFFFF,0.8,4.0,1.24,210,30.0
```

## Python API Usage

```python
from bananaforge.output.threemf_exporter import ThreeMFExporter, ThreeMFExportConfig
from bananaforge.materials.database import DefaultMaterials

# Initialize exporter
exporter = ThreeMFExporter()
exporter.material_db = DefaultMaterials.create_bambu_basic_pla()

# Configure export
config = ThreeMFExportConfig(
    bambu_compatible=True,
    include_metadata=True,
    validate_output=True
)

# Export optimization results
result = exporter.export(
    optimization_results=optimization_results,
    output_path="output/model.3mf",
    config=config
)

print(f"Export success: {result['success']}")
print(f"File size: {result['file_size']} bytes")
print(f"Materials used: {result['materials_count']}")
```

## Slicer Compatibility

### Bambu Studio (EXPERIMENTAL)
- ✅ Full material assignment support
- ✅ Per-layer color changes
- ✅ Automatic material swap detection
- ✅ Transparency handling

### PrusaSlicer
- ✅ Basic 3MF import
- ✅ Multi-material support
- ⚠️ Limited per-layer material assignment

### Other Slicers
- ✅ Standard 3MF geometry import
- ⚠️ Material support varies by slicer

## Troubleshooting

### Common Issues

**Large file sizes**
```bash
# Reduce export mesh resolution
bananaforge convert input.jpg --max-triangles 2000000 --export-format 3mf
```

**Validation errors**
```bash
# Check file structure
unzip -l output/model.3mf

# Validate XML
xmllint --noout 3D/3dmodel.model
```

**Material assignment issues**
```bash
# Verify material database
bananaforge export-materials --output debug_materials.csv
```

### Performance Optimization

For large models:
- Use `--max-triangles` to limit mesh resolution
- Reduce `--max-layers` for simpler geometry
- Enable compression with larger images

## Advanced Features

### Transparency and Color Mixing

3MF format supports transparency for advanced color effects:

```python
# Materials with transparency
layer_materials = {
    0: {'material_id': 'base_black', 'transparency': 1.0},      # Opaque base
    1: {'material_id': 'accent_red', 'transparency': 0.7},      # Semi-transparent
    2: {'material_id': 'highlight_white', 'transparency': 0.3}  # Highly transparent
}
```

### Metadata Embedding

3MF files include comprehensive metadata:

```xml
<metadata>
  <generator>BananaForge</generator>
  <version>1.0</version>
  <vertices_count>10000</vertices_count>
  <triangles_count>5000</triangles_count>
  <materials_count>5</materials_count>
  <layers_count>20</layers_count>
</metadata>
```

## Best Practices

### Material Selection
1. **Start with base colors**: Use dark colors (black/brown) as base layers
2. **Limit material count**: 4-8 materials for optimal print time
3. **Consider transparency**: Use transparent materials for color mixing effects

### Optimization Settings
1. **Layer height**: 0.08-0.2mm for color transitions
2. **Max layers**: 15-30 for typical prints
3. **Physical size**: Scale to printer bed size

### Workflow Integration
1. **Test with small models**: Validate settings before large prints
2. **Export multiple formats**: Keep STL backup alongside 3MF
3. **Document material sources**: Track filament brands and colors used

## Technical Specifications

- **3MF Core Specification**: v1.3 compliant
- **Supported namespaces**: Core, Material, Slice extensions
- **File compression**: ZIP with DEFLATE compression
- **XML encoding**: UTF-8 with proper namespace handling
- **Validation**: Automated structure and content validation

## Future Enhancements

The 3MF export system is designed for extensibility:

- **Slice extension**: Layer-by-layer printing instructions
- **Production extension**: Manufacturing workflow integration  
- **Custom metadata**: Project-specific annotations
- **Advanced materials**: Multi-component material definitions

---

For more information, see:
- [3MF Consortium Specification](https://www.3mf.io/specification/)
- [BananaForge CLI Reference](cli_reference.md)
- [Material Database Guide](materials_guide.md)
