#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 6.1: ImageProcessor Enhancement

Following BDD Given-When-Then scenarios from tasks1.md:
- ImageProcessor transparency method integration
- Memory efficient transparency analysis 
- Backward compatibility with RGB conversion
- Optional alpha channel preservation
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# Import the classes we're testing and enhancing
from bananaforge.image.processor import ImageProcessor
from bananaforge.image.transparency_detector import TransparencyDetector, TransparencyInfo


@pytest.fixture
def processor():
    """Create an ImageProcessor instance for testing."""
    return ImageProcessor(device="cpu")


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class TestImageProcessorHelpers:
    """Helper methods for creating test images."""
    
    def create_rgba_png(self, path: Path, width: int = 100, height: int = 100):
        """Helper: Create PNG with RGBA alpha channel."""
        image = Image.new('RGBA', (width, height), (255, 0, 0, 0))  # Transparent red
        # Add opaque center square
        for x in range(width // 4, 3 * width // 4):
            for y in range(height // 4, 3 * height // 4):
                image.putpixel((x, y), (0, 255, 0, 255))  # Opaque green
        image.save(path, 'PNG')
        return path
    
    def create_rgb_png(self, path: Path, width: int = 100, height: int = 100):
        """Helper: Create PNG without transparency."""
        image = Image.new('RGB', (width, height), (255, 0, 0))  # Red background
        # Add green center
        for x in range(width // 4, 3 * width // 4):
            for y in range(height // 4, 3 * height // 4):
                image.putpixel((x, y), (0, 255, 0))  # Green center
        image.save(path, 'PNG')
        return path


class TestScenario1_ImageProcessorTransparencyMethod:
    """
    BDD Scenario: ImageProcessor transparency method
    Given the ImageProcessor class exists
    When I add transparency detection capability
    Then it should integrate cleanly with load_image method
    And maintain backward compatibility with RGB conversion
    And provide optional alpha channel preservation
    """
    
    def test_processor_has_transparency_detector(self, processor):
        """Test that ImageProcessor includes transparency detection capability."""
        # Given: the ImageProcessor class exists
        assert isinstance(processor, ImageProcessor)
        
        # When: I check for transparency detection capability
        # Then: it should have a transparency detector integrated
        assert hasattr(processor, 'detect_transparency'), "Should have detect_transparency method"
        assert hasattr(processor, 'load_image_with_transparency'), "Should have enhanced load method"
        
    def test_backward_compatibility_rgb_conversion(self, processor, temp_dir):
        """Test that existing RGB conversion behavior is maintained."""
        # Given: a PNG image with alpha channel
        png_path = temp_dir / "rgba_test.png"
        TestImageProcessorHelpers().create_rgba_png(png_path)
        
        # When: I load the image using existing load_image method
        result = processor.load_image(str(png_path))
        
        # Then: it should maintain backward compatibility with RGB conversion
        assert result is not None, "Should return tensor"
        assert result.shape[1] == 3, "Should have 3 channels (RGB) for backward compatibility"
        assert result.dtype == torch.float32, "Should return float32 tensor"
        assert 0 <= result.min() and result.max() <= 1, "Should be normalized to [0,1]"
        
    def test_optional_alpha_channel_preservation(self, processor, temp_dir):
        """Test optional alpha channel preservation in load_image."""
        # Given: a PNG image with alpha channel
        png_path = temp_dir / "rgba_preserve.png"
        TestImageProcessorHelpers().create_rgba_png(png_path)
        
        # When: I load the image with alpha preservation enabled
        rgb_result, alpha_mask = processor.load_image_with_transparency(str(png_path))
        
        # Then: it should provide optional alpha channel preservation
        assert rgb_result is not None, "Should return RGB tensor"
        assert alpha_mask is not None, "Should return alpha mask"
        assert rgb_result.shape[1] == 3, "RGB result should have 3 channels"
        assert alpha_mask.shape[1] == 1, "Alpha mask should have 1 channel"
        assert rgb_result.shape[-2:] == alpha_mask.shape[-2:], "Should have matching spatial dimensions"
        
    def test_transparency_detection_integration(self, processor, temp_dir):
        """Test transparency detection integration with ImageProcessor."""
        # Given: an ImageProcessor with transparency detection
        rgba_path = temp_dir / "rgba_detection.png"
        TestImageProcessorHelpers().create_rgba_png(rgba_path)
        
        # When: I detect transparency using the processor
        transparency_info = processor.detect_transparency(str(rgba_path))
        
        # Then: it should integrate cleanly with existing methods
        assert isinstance(transparency_info, TransparencyInfo), "Should return TransparencyInfo"
        assert transparency_info.has_transparency is True, "Should detect transparency"
        assert transparency_info.format_type == "PNG", "Should identify format"


class TestScenario2_MemoryEfficientTransparencyAnalysis:
    """
    BDD Scenario: Memory efficient transparency analysis
    Given large images being processed for transparency
    When transparency detection runs
    Then it should not significantly increase memory usage
    And optimize for common cases (no transparency)
    And handle large files efficiently
    """
    
    def test_memory_efficient_processing(self, processor, temp_dir):
        """Test memory efficient transparency analysis."""
        # Given: large images being processed for transparency
        large_rgba_path = temp_dir / "large_rgba.png"
        TestImageProcessorHelpers().create_rgba_png(large_rgba_path, 512, 512)  # Larger image
        
        # When: transparency detection runs
        # Track memory usage (simplified - in real implementation would use memory profiler)
        import sys
        initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        transparency_info = processor.detect_transparency(str(large_rgba_path))
        
        final_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        # Then: it should not significantly increase memory usage
        assert transparency_info is not None, "Should successfully process large image"
        # Memory increase should be reasonable (this is a basic check)
        # In practice, you'd want more sophisticated memory profiling
        
    def test_optimize_for_no_transparency_case(self, processor, temp_dir):
        """Test optimization for common case of no transparency."""
        # Given: an RGB image without transparency (common case)
        rgb_path = temp_dir / "rgb_optimized.png"
        TestImageProcessorHelpers().create_rgb_png(rgb_path)
        
        # When: I detect transparency
        start_time = time.time()
        transparency_info = processor.detect_transparency(str(rgb_path))
        processing_time = time.time() - start_time
        
        # Then: it should optimize for common cases (no transparency)
        assert transparency_info.has_transparency is False, "Should detect no transparency"
        assert processing_time < 1.0, "Should process quickly for no-transparency case"
        assert transparency_info.transparent_pixel_count == 0, "Should have no transparent pixels"
        
    def test_handle_large_files_efficiently(self, processor, temp_dir):
        """Test efficient handling of large files."""
        # Given: a large image file
        large_path = temp_dir / "very_large.png"
        TestImageProcessorHelpers().create_rgba_png(large_path, 1024, 1024)  # Very large
        
        # When: I process the large file
        transparency_info = processor.detect_transparency(str(large_path))
        
        # Then: it should handle large files efficiently
        assert transparency_info is not None, "Should handle large files"
        assert transparency_info.total_pixels == 1024 * 1024, "Should count pixels correctly"
        # Should complete without memory errors or excessive time


class TestScenario3_CleanSeparationOfConcerns:
    """
    BDD Scenario: Clean separation of concerns
    Given ImageProcessor with integrated transparency detection
    When I use detection vs processing features
    Then detection should be separate from image processing
    And processing should be separate from transparency detection
    And both should work independently
    """
    
    def test_detection_separate_from_processing(self, processor, temp_dir):
        """Test that detection is separate from processing concerns."""
        # Given: ImageProcessor with integrated transparency detection
        png_path = temp_dir / "separation_test.png"
        TestImageProcessorHelpers().create_rgba_png(png_path)
        
        # When: I use detection vs processing features
        # Detection should not modify the image
        transparency_info = processor.detect_transparency(str(png_path))
        processed_image = processor.load_image(str(png_path))
        
        # Then: detection should be separate from image processing
        assert transparency_info.has_transparency is True, "Detection should work"
        assert processed_image.shape[1] == 3, "Processing should work independently"
        # Detection doesn't affect processing output
        
    def test_processing_separate_from_detection(self, processor, temp_dir):
        """Test that processing works independently of detection."""
        # Given: an image that can be processed
        png_path = temp_dir / "processing_independent.png"
        TestImageProcessorHelpers().create_rgb_png(png_path)
        
        # When: I use processing without detection
        processed_image = processor.load_image(str(png_path), target_size=(128, 128))
        
        # Then: processing should be separate from transparency detection
        assert processed_image is not None, "Processing should work independently"
        assert processed_image.shape[-2:] == (128, 128), "Should resize correctly"
        # Processing works without needing to call detection
        
    def test_both_features_work_independently(self, processor, temp_dir):
        """Test that detection and processing work completely independently."""
        # Given: images for testing independence
        rgba_path = temp_dir / "independent_rgba.png"
        rgb_path = temp_dir / "independent_rgb.png"
        TestImageProcessorHelpers().create_rgba_png(rgba_path)
        TestImageProcessorHelpers().create_rgb_png(rgb_path)
        
        # When: I use both features independently
        # Detection on RGBA image
        rgba_detection = processor.detect_transparency(str(rgba_path))
        # Processing on RGB image  
        rgb_processed = processor.load_image(str(rgb_path))
        # Processing on RGBA image (should still work)
        rgba_processed = processor.load_image(str(rgba_path))
        
        # Then: both should work independently
        assert rgba_detection.has_transparency is True, "Detection works on RGBA"
        assert rgb_processed.shape[1] == 3, "Processing works on RGB"
        assert rgba_processed.shape[1] == 3, "Processing works on RGBA (converts to RGB)"


class TestEnhancedImageProcessor:
    """Test the enhanced ImageProcessor interface and implementation."""
    
    def test_enhanced_processor_methods_exist(self, processor):
        """Test that enhanced methods exist with correct signatures."""
        # Test method existence
        assert hasattr(processor, 'detect_transparency')
        assert hasattr(processor, 'load_image_with_transparency')
        
        # Test method signatures
        import inspect
        
        detect_sig = inspect.signature(processor.detect_transparency)
        assert 'image_path' in detect_sig.parameters, "detect_transparency should accept image_path"
        
        load_sig = inspect.signature(processor.load_image_with_transparency)
        expected_params = ['image_path']
        for param in expected_params:
            assert param in load_sig.parameters, f"load_image_with_transparency should have {param} parameter"
    
    def test_enhanced_methods_return_correct_types(self, processor, temp_dir):
        """Test that enhanced methods return the correct types."""
        # Given: test image
        png_path = temp_dir / "type_test.png"
        TestImageProcessorHelpers().create_rgba_png(png_path)
        
        # When: calling enhanced methods
        transparency_result = processor.detect_transparency(str(png_path))
        load_result = processor.load_image_with_transparency(str(png_path))
        
        # Then: they should return correct types
        assert isinstance(transparency_result, TransparencyInfo), "detect_transparency should return TransparencyInfo"
        assert isinstance(load_result, tuple), "load_image_with_transparency should return tuple"
        assert len(load_result) == 2, "Should return (rgb_tensor, alpha_mask)"
        
        rgb_tensor, alpha_mask = load_result
        assert isinstance(rgb_tensor, torch.Tensor), "RGB should be tensor"
        assert isinstance(alpha_mask, torch.Tensor), "Alpha mask should be tensor"


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness of enhanced ImageProcessor."""
    
    def test_graceful_error_handling(self, processor, temp_dir):
        """Test graceful error handling in enhanced methods."""
        # Test non-existent file
        nonexistent_path = temp_dir / "does_not_exist.png"
        
        with pytest.raises((FileNotFoundError, IOError)):
            processor.detect_transparency(str(nonexistent_path))
            
        with pytest.raises((FileNotFoundError, IOError)):
            processor.load_image_with_transparency(str(nonexistent_path))
    
    def test_invalid_file_handling(self, processor, temp_dir):
        """Test handling of invalid image files."""
        # Create fake image file
        fake_path = temp_dir / "fake.png" 
        fake_path.write_text("This is not an image")
        
        with pytest.raises((IOError, ValueError)):
            processor.detect_transparency(str(fake_path))
            
        with pytest.raises((IOError, ValueError)):
            processor.load_image_with_transparency(str(fake_path))


# Import time for performance testing
import time
import gc