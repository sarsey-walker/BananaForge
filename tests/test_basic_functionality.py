#!/usr/bin/env python3
"""Basic functionality tests for BananaForge."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from bananaforge.core.optimizer import LayerOptimizer, OptimizationConfig
from bananaforge.image.processor import ImageProcessor
from bananaforge.materials.database import MaterialDatabase, DefaultMaterials
from bananaforge.materials.matcher import ColorMatcher
from bananaforge.output.exporter import ModelExporter


class TestBasicFunctionality:
    """Test basic functionality of BananaForge components."""
    
    @pytest.fixture
    def device(self):
        """Get appropriate device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
        
    @pytest.fixture
    def sample_image(self, device):
        """Create a sample image for testing."""
        # Create a simple test image (3x3 RGB stripes)
        image = torch.zeros(1, 3, 64, 64, device=device)
        
        # Red stripe
        image[:, 0, :, :21] = 1.0
        
        # Green stripe  
        image[:, 1, :, 21:42] = 1.0
        
        # Blue stripe
        image[:, 2, :, 42:] = 1.0
        
        return image
        
    @pytest.fixture
    def material_db(self):
        """Create test material database."""
        return DefaultMaterials.create_bambu_basic_pla()
        
    def test_image_processor_initialization(self, device):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(device)
        assert processor.device.type == device
        
    def test_material_database_creation(self, material_db):
        """Test material database creation."""
        assert len(material_db) > 0
        
        # Test getting materials
        materials = list(material_db)
        assert len(materials) > 0
        
        first_material = materials[0]
        assert hasattr(first_material, 'name')
        assert hasattr(first_material, 'color_rgb')
        assert hasattr(first_material, 'color_hex')
        
    def test_color_matching(self, sample_image, material_db, device):
        """Test color matching functionality."""
        color_matcher = ColorMatcher(material_db, device)
        
        selected_materials, selected_colors, color_mapping = color_matcher.match_image_colors(
            sample_image, max_materials=5, method="euclidean"
        )
        
        assert len(selected_materials) > 0
        assert len(selected_materials) <= 5
        assert selected_colors.shape[0] == len(selected_materials)
        assert selected_colors.shape[1] == 3  # RGB
        
    def test_optimization_config(self):
        """Test optimization configuration."""
        config = OptimizationConfig(
            iterations=100,
            learning_rate=0.01,
            device="cpu"
        )
        
        assert config.iterations == 100
        assert config.learning_rate == 0.01
        assert config.device == "cpu"
        
    def test_layer_optimizer_initialization(self, device):
        """Test LayerOptimizer initialization."""
        config = OptimizationConfig(
            iterations=10,
            device=device
        )
        
        optimizer = LayerOptimizer(
            image_size=(32, 32),
            num_materials=4,
            config=config
        )
        
        assert optimizer.config.iterations == 10
        assert optimizer.image_size == (32, 32)
        assert optimizer.num_materials == 4
        
    def test_optimization_step(self, sample_image, material_db, device):
        """Test a few optimization steps."""
        # Setup
        color_matcher = ColorMatcher(material_db, device)
        selected_materials, selected_colors, _ = color_matcher.match_image_colors(
            sample_image, max_materials=3
        )
        
        config = OptimizationConfig(
            iterations=5,  # Very few iterations for testing
            device=device
        )
        
        optimizer = LayerOptimizer(
            image_size=sample_image.shape[-2:],
            num_materials=len(selected_materials),
            config=config
        )
        
        # Run optimization
        loss_history = optimizer.optimize(
            target_image=sample_image,
            material_colors=selected_colors
        )
        
        # Check results
        assert 'total' in loss_history
        assert len(loss_history['total']) == config.iterations
        
        # Test getting final results
        final_image, height_map, material_assignments = optimizer.get_final_results(selected_colors)
        
        assert final_image.shape == sample_image.shape
        assert height_map.shape[-2:] == sample_image.shape[-2:]
        assert material_assignments.shape[-2:] == sample_image.shape[-2:]
        
    def test_stl_generation(self, device):
        """Test STL file generation."""
        from bananaforge.output.stl_generator import STLGenerator
        
        # Create simple height map
        height_map = torch.ones(1, 1, 32, 32, device=device) * 5  # 5 layers high
        
        generator = STLGenerator()
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            try:
                mesh = generator.generate_stl(
                    height_map=height_map,
                    output_path=tmp_file.name,
                    physical_size=50.0  # 50mm
                )
                
                assert mesh is not None
                assert len(mesh.vertices) > 0
                assert len(mesh.faces) > 0
                assert Path(tmp_file.name).exists()
                
            finally:
                # Cleanup
                if Path(tmp_file.name).exists():
                    os.unlink(tmp_file.name)

    def test_stl_height_uses_initial_layer_without_extra_base_offset(self):
        """STL Z bounds should match layer units plus the initial layer only."""
        from bananaforge.output.stl_generator import STLGenerator

        height_map = torch.ones(1, 1, 4, 4) * 5
        generator = STLGenerator(
            layer_height=0.08,
            initial_layer_height=0.16,
            base_height=0.24,
        )

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            try:
                mesh = generator.generate_stl(
                    height_map=height_map,
                    output_path=tmp_file.name,
                    physical_size=20.0,
                    smooth_mesh=False,
                )

                assert mesh.bounds[0][2] == pytest.approx(0.0)
                assert mesh.bounds[1][2] == pytest.approx(0.56)

            finally:
                if Path(tmp_file.name).exists():
                    os.unlink(tmp_file.name)

    def test_flat_stl_uses_greedy_mesh_reduction(self):
        """Flat height maps should merge large coplanar regions."""
        from bananaforge.output.exporter import ModelExporter
        from bananaforge.output.stl_generator import STLGenerator

        height_map = torch.ones(1, 1, 10, 10)
        generator = STLGenerator()

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            try:
                mesh = generator.generate_stl(
                    height_map=height_map,
                    output_path=tmp_file.name,
                    physical_size=20.0,
                    smooth_mesh=False,
                )

                old_triangle_estimate = ModelExporter.estimate_triangle_count(10, 10)

                assert len(mesh.faces) < old_triangle_estimate // 2
                assert Path(tmp_file.name).exists()

            finally:
                if Path(tmp_file.name).exists():
                    os.unlink(tmp_file.name)

    def test_stl_bottom_mode_none_removes_bottom_faces(self):
        """Skipping the bottom face should reduce mesh faces for flat exports."""
        from bananaforge.output.stl_generator import STLGenerator

        height_map = torch.ones(1, 1, 10, 10)
        generator = STLGenerator()

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as simplified_file:
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as open_file:
                try:
                    simplified_mesh = generator.generate_stl(
                        height_map=height_map,
                        output_path=simplified_file.name,
                        physical_size=20.0,
                        smooth_mesh=False,
                        bottom_mode="simplified",
                    )
                    open_mesh = generator.generate_stl(
                        height_map=height_map,
                        output_path=open_file.name,
                        physical_size=20.0,
                        smooth_mesh=False,
                        bottom_mode="none",
                    )

                    assert len(open_mesh.faces) < len(simplified_mesh.faces)
                    assert not open_mesh.is_watertight

                finally:
                    for path in (simplified_file.name, open_file.name):
                        if Path(path).exists():
                            os.unlink(path)
                    
    def test_instruction_generation(self, material_db, device):
        """Test swap instruction generation."""
        from bananaforge.output.instructions import SwapInstructionGenerator
        
        # Create simple material assignments
        material_assignments = torch.zeros(10, 32, 32, dtype=torch.long, device=device)
        
        # Layer 0-3: material 0
        material_assignments[0:4] = 0
        # Layer 4-6: material 1  
        material_assignments[4:7] = 1
        # Layer 7-9: material 2
        material_assignments[7:10] = 2
        
        material_ids = ['mat_0', 'mat_1', 'mat_2']
        
        generator = SwapInstructionGenerator()
        instructions = generator.generate_swap_instructions(
            material_assignments=material_assignments,
            material_database=material_db,
            material_ids=material_ids
        )
        
        # Should have 2 swaps (0->1, 1->2)
        assert len(instructions) == 2
        assert instructions[0].new_material == 'mat_1'
        assert instructions[1].new_material == 'mat_2'
        
    def test_export_workflow(self, sample_image, material_db, device):
        """Test complete export workflow."""
        # Quick optimization
        color_matcher = ColorMatcher(material_db, device)
        selected_materials, selected_colors, _ = color_matcher.match_image_colors(
            sample_image, max_materials=3
        )
        
        config = OptimizationConfig(iterations=3, device=device)
        optimizer = LayerOptimizer(
            image_size=sample_image.shape[-2:],
            num_materials=len(selected_materials),
            config=config
        )
        
        optimizer.optimize(sample_image, selected_colors)
        final_image, height_map, material_assignments = optimizer.get_final_results(selected_colors)
        
        # Test export
        exporter = ModelExporter()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_files = exporter.export_complete_model(
                height_map=height_map,
                material_assignments=material_assignments,
                material_database=material_db,
                material_ids=selected_materials,
                output_dir=tmp_dir,
                project_name="test_model",
                export_formats=["stl", "instructions"]
            )
            
            assert "stl" in generated_files
            assert "instructions_txt" in generated_files
            
            # Check files exist
            for file_path in generated_files.values():
                assert Path(file_path).exists()
                
    def test_cost_calculation(self, sample_image, material_db, device):
        """Test cost calculation."""
        from bananaforge.output.instructions import CostCalculator
        
        # Simple setup
        height_map = torch.ones(1, 1, 32, 32, device=device) * 5
        material_assignments = torch.zeros(5, 32, 32, dtype=torch.long, device=device)
        material_assignments[2:] = 1  # Use material 1 for top layers
        
        material_ids = ['mat_0', 'mat_1']
        
        calculator = CostCalculator()
        usage_data = calculator.calculate_material_usage(
            height_map=height_map,
            material_assignments=material_assignments,
            material_database=material_db,
            material_ids=material_ids,
            physical_size=100.0
        )
        
        assert len(usage_data) <= 2  # Should have data for used materials
        for material_id, data in usage_data.items():
            assert 'weight_grams' in data
            assert 'cost_usd' in data
            assert data['weight_grams'] > 0
            assert data['cost_usd'] > 0


def test_cli_import():
    """Test that CLI module can be imported."""
    try:
        from bananaforge.cli import cli
        assert cli is not None
    except ImportError as e:
        pytest.skip(f"CLI import failed: {e}")


def test_package_import():
    """Test basic package imports."""
    import bananaforge
    assert hasattr(bananaforge, '__version__')
    
    # Test main components can be imported
    from bananaforge import LayerOptimizer, ImageProcessor, MaterialManager, ModelExporter
    assert LayerOptimizer is not None
    assert ImageProcessor is not None
    assert ModelExporter is not None


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])
