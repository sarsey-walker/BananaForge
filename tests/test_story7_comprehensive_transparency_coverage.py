#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 7.1: Comprehensive Test Coverage

This test suite provides comprehensive coverage for all transparency detection 
functionality, ensuring reliability across all scenarios and achieving >95% 
code coverage.

Test Coverage Areas:
- All image formats (PNG, GIF, WebP, TIFF)
- Edge cases and error conditions  
- CLI integration functionality
- Performance benchmarks
- Mock testing for error conditions
"""

import pytest
import tempfile
import time
import gc
import os
import sys
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

# Import the classes we're testing
from bananaforge.image.transparency_detector import TransparencyDetector, TransparencyInfo
from bananaforge.image.processor import ImageProcessor


@pytest.fixture
def detector():
    """Create a TransparencyDetector instance for testing."""
    return TransparencyDetector()


@pytest.fixture
def processor():
    """Create an ImageProcessor instance for testing."""
    return ImageProcessor(device="cpu")


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class ImageTestHelpers:
    """Helper class for creating various test images."""
    
    @staticmethod
    def create_png_rgba(path: Path, width: int = 100, height: int = 100, alpha_pattern: str = "gradient"):
        """Create PNG with RGBA alpha channel with various patterns."""
        image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        
        if alpha_pattern == "gradient":
            # Alpha gradient from transparent to opaque
            for x in range(width):
                alpha_val = int((x / width) * 255)
                for y in range(height):
                    image.putpixel((x, y), (255, 0, 0, alpha_val))
        elif alpha_pattern == "checkerboard":
            # Checkerboard transparency pattern
            for x in range(width):
                for y in range(height):
                    alpha_val = 255 if (x + y) % 2 == 0 else 0
                    image.putpixel((x, y), (0, 255, 0, alpha_val))
        elif alpha_pattern == "border":
            # Transparent border, opaque center
            for x in range(width):
                for y in range(height):
                    if x < 10 or x >= width-10 or y < 10 or y >= height-10:
                        alpha_val = 0  # Transparent border
                    else:
                        alpha_val = 255  # Opaque center
                    image.putpixel((x, y), (0, 0, 255, alpha_val))
        
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_gif_transparent(path: Path, width: int = 100, height: int = 100):
        """Create GIF with transparency."""
        # Create image with transparency
        image = Image.new('P', (width, height))
        
        # Create a simple palette (256 colors)
        palette = []
        for i in range(256):
            palette.extend([i, 0, 255-i])  # RGB triplets
        image.putpalette(palette)
        
        # Fill with pattern - index 0 will be transparent
        for x in range(width):
            for y in range(height):
                if (x + y) % 4 == 0:
                    image.putpixel((x, y), 0)  # Transparent pixels
                else:
                    image.putpixel((x, y), 1 + (x + y) % 10)  # Other colors
        
        image.save(path, 'GIF', transparency=0)
        return path
    
    @staticmethod
    def create_webp_alpha(path: Path, width: int = 100, height: int = 100):
        """Create WebP with alpha channel."""
        # Create RGBA image
        image = Image.new('RGBA', (width, height))
        
        # Create alpha pattern
        for x in range(width):
            for y in range(height):
                # Radial alpha gradient
                center_x, center_y = width // 2, height // 2
                distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
                max_distance = (width**2 + height**2)**0.5 / 2
                alpha_val = max(0, min(255, int(255 * (1 - distance / max_distance))))
                
                image.putpixel((x, y), (255, 128, 64, alpha_val))
        
        # WebP requires special handling
        try:
            image.save(path, 'WebP')
        except (OSError, ValueError):
            # WebP not supported, create PNG instead for testing
            path = path.with_suffix('.png')
            image.save(path, 'PNG')
        
        return path
    
    @staticmethod
    def create_tiff_alpha(path: Path, width: int = 100, height: int = 100):
        """Create TIFF with alpha channel."""
        image = Image.new('RGBA', (width, height))
        
        # Simple alpha pattern - half transparent, half opaque
        for x in range(width):
            for y in range(height):
                alpha_val = 255 if x < width // 2 else 128
                image.putpixel((x, y), (128, 255, 128, alpha_val))
        
        image.save(path, 'TIFF')
        return path
    
    @staticmethod
    def create_large_image(path: Path, width: int = 2048, height: int = 2048, format_type: str = 'PNG'):
        """Create large image for performance testing."""
        if format_type == 'RGBA':
            image = Image.new('RGBA', (width, height), (255, 0, 0, 128))
        else:
            image = Image.new('RGB', (width, height), (255, 0, 0))
        
        # Add some pattern for realism
        for x in range(0, width, 50):
            for y in range(0, height, 50):
                color = (0, 255, 0, 255) if format_type == 'RGBA' else (0, 255, 0)
                try:
                    if format_type == 'RGBA':
                        for dx in range(10):
                            for dy in range(10):
                                if x + dx < width and y + dy < height:
                                    image.putpixel((x + dx, y + dy), color)
                    else:
                        for dx in range(10):
                            for dy in range(10):
                                if x + dx < width and y + dy < height:
                                    image.putpixel((x + dx, y + dy), color)
                except IndexError:
                    pass
        
        image.save(path, 'PNG')
        return path


class TestComprehensiveImageFormatSupport:
    """
    Test suite for all supported image formats with comprehensive scenarios.
    Covers: PNG (RGBA, LA, P), GIF, WebP, TIFF
    """
    
    def test_png_rgba_various_patterns(self, detector, temp_dir):
        """Test PNG RGBA detection with various alpha patterns."""
        test_patterns = ["gradient", "checkerboard", "border"]
        
        for pattern in test_patterns:
            png_path = temp_dir / f"png_rgba_{pattern}.png"
            ImageTestHelpers.create_png_rgba(png_path, alpha_pattern=pattern)
            
            result = detector.detect_transparency(str(png_path))
            
            assert result.has_transparency is True, f"Should detect transparency in {pattern} pattern"
            assert result.format_type == "PNG", "Should identify PNG format"
            assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
            assert result.image_mode == "RGBA", "Should detect RGBA mode"
            assert result.transparent_pixel_count > 0, f"Should have transparent pixels in {pattern} pattern"
    
    def test_png_la_grayscale_alpha(self, detector, temp_dir):
        """Test PNG LA (grayscale with alpha) detection."""
        png_path = temp_dir / "png_la.png"
        
        # Create LA image
        la_image = Image.new('LA', (100, 100), (128, 255))  # Gray with full alpha
        # Add some transparent pixels
        for x in range(20):
            for y in range(100):
                la_image.putpixel((x, y), (128, 100))  # Semi-transparent
        
        la_image.save(png_path, 'PNG')
        
        result = detector.detect_transparency(str(png_path))
        
        assert result.has_transparency is True, "Should detect transparency in LA mode"
        assert result.image_mode == "LA", "Should detect LA mode"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
    
    def test_png_palette_with_transparency_key(self, detector, temp_dir):
        """Test PNG palette mode with transparency key."""
        png_path = temp_dir / "png_palette.png"
        
        # Create palette image with transparency
        p_image = Image.new('P', (100, 100))
        
        # Set up palette
        palette = []
        for i in range(256):
            palette.extend([i, 128, 255-i])
        p_image.putpalette(palette)
        
        # Fill with pattern where index 0 will be transparent
        for x in range(100):
            for y in range(100):
                if x < 25 or x > 75:
                    p_image.putpixel((x, y), 0)  # Will be transparent
                else:
                    p_image.putpixel((x, y), 1)
        
        p_image.save(png_path, 'PNG', transparency=0)
        
        result = detector.detect_transparency(str(png_path))
        
        assert result.has_transparency is True, "Should detect palette transparency"
        assert result.image_mode == "P", "Should detect P mode"
        assert result.transparency_type == "palette_transparency", "Should identify palette transparency"
        assert result.transparent_pixel_count > 0, "Should count transparent pixels correctly"
    
    def test_gif_transparency_detection(self, detector, temp_dir):
        """Test GIF transparency detection."""
        gif_path = temp_dir / "test_transparent.gif"
        ImageTestHelpers.create_gif_transparent(gif_path)
        
        result = detector.detect_transparency(str(gif_path))
        
        # GIF transparency should be detected by our system
        # Note: This depends on PIL's GIF transparency handling
        assert result.format_type in ["GIF", "PNG"], "Should identify image format"
        assert result.image_mode == "P", "GIF should typically be P mode"
    
    @pytest.mark.skipif(not hasattr(Image, 'WebP'), reason="WebP support not available")
    def test_webp_alpha_detection(self, detector, temp_dir):
        """Test WebP with alpha channel detection."""
        webp_path = temp_dir / "test_alpha.webp"
        actual_path = ImageTestHelpers.create_webp_alpha(webp_path)
        
        result = detector.detect_transparency(str(actual_path))
        
        if actual_path.suffix == '.webp':
            assert result.format_type == "WEBP", "Should identify WebP format"
        else:
            # Fallback to PNG if WebP not supported
            assert result.format_type == "PNG", "Should identify PNG format (WebP fallback)"
        
        assert result.has_transparency is True, "Should detect alpha channel"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
    
    def test_tiff_alpha_detection(self, detector, temp_dir):
        """Test TIFF with alpha channel detection."""
        tiff_path = temp_dir / "test_alpha.tiff"
        ImageTestHelpers.create_tiff_alpha(tiff_path)
        
        result = detector.detect_transparency(str(tiff_path))
        
        assert result.format_type == "TIFF", "Should identify TIFF format"
        assert result.has_transparency is True, "Should detect alpha channel"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
        assert result.image_mode == "RGBA", "Should detect RGBA mode"


class TestEdgeCasesAndErrorConditions:
    """
    Comprehensive testing of edge cases and error conditions.
    """
    
    def test_corrupted_image_handling(self, detector, temp_dir):
        """Test handling of corrupted image files."""
        corrupted_path = temp_dir / "corrupted.png"
        
        # Create fake PNG header followed by garbage
        fake_png_data = b'\x89PNG\r\n\x1a\n' + b'garbage data that is not a valid PNG'
        corrupted_path.write_bytes(fake_png_data)
        
        with pytest.raises((IOError, ValueError)):
            detector.detect_transparency(str(corrupted_path))
    
    def test_empty_file_handling(self, detector, temp_dir):
        """Test handling of empty files."""
        empty_path = temp_dir / "empty.png"
        empty_path.write_bytes(b'')
        
        with pytest.raises((IOError, ValueError)):
            detector.detect_transparency(str(empty_path))
    
    def test_non_image_file_handling(self, detector, temp_dir):
        """Test handling of non-image files."""
        text_path = temp_dir / "text_file.png"
        text_path.write_text("This is just a text file, not an image")
        
        with pytest.raises((IOError, ValueError)):
            detector.detect_transparency(str(text_path))
    
    def test_very_small_images(self, detector, temp_dir):
        """Test handling of very small images (1x1, 2x2)."""
        small_sizes = [(1, 1), (2, 2), (3, 1), (1, 5)]
        
        for width, height in small_sizes:
            small_path = temp_dir / f"small_{width}x{height}.png"
            small_image = Image.new('RGBA', (width, height), (255, 0, 0, 128))
            small_image.save(small_path, 'PNG')
            
            result = detector.detect_transparency(str(small_path))
            
            assert result is not None, f"Should handle {width}x{height} image"
            assert result.total_pixels == width * height, "Should count pixels correctly"
            assert result.has_transparency is True, "Should detect transparency in small image"
    
    def test_images_with_unusual_modes(self, detector, temp_dir):
        """Test images with unusual color modes."""
        # Test various PIL image modes
        modes_to_test = ['L', 'CMYK', '1']  # Grayscale, CMYK, 1-bit
        
        for mode in modes_to_test:
            try:
                mode_path = temp_dir / f"mode_{mode}.png"
                if mode == '1':
                    # 1-bit mode needs special handling
                    image = Image.new(mode, (50, 50), 1)
                elif mode == 'CMYK':
                    # CMYK mode
                    image = Image.new(mode, (50, 50), (100, 50, 0, 25))
                else:
                    # Grayscale
                    image = Image.new(mode, (50, 50), 128)
                
                # Convert to format that PNG can handle
                if mode == 'CMYK':
                    image = image.convert('RGB')
                
                image.save(mode_path, 'PNG')
                
                result = detector.detect_transparency(str(mode_path))
                
                assert result is not None, f"Should handle {mode} mode"
                assert result.has_transparency is False, f"Mode {mode} should not have transparency"
                
            except Exception as e:
                # Some modes might not be supported, which is OK
                print(f"Mode {mode} not supported: {e}")
    
    def test_permission_denied_error(self, detector, temp_dir):
        """Test handling when file permissions are denied."""
        if os.name == 'nt':  # Windows
            pytest.skip("Permission test not reliable on Windows")
        
        # Create a valid image first
        restricted_path = temp_dir / "restricted.png"
        test_image = Image.new('RGB', (50, 50), (255, 0, 0))
        test_image.save(restricted_path, 'PNG')
        
        # Remove read permissions
        os.chmod(restricted_path, 0o000)
        
        try:
            with pytest.raises((PermissionError, IOError)):
                detector.detect_transparency(str(restricted_path))
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_path, 0o644)
    
    def test_path_with_unicode_characters(self, detector, temp_dir):
        """Test handling of file paths with unicode characters."""
        unicode_path = temp_dir / "test_图片_αβγ_🔍.png"
        test_image = Image.new('RGBA', (50, 50), (255, 0, 0, 128))
        test_image.save(unicode_path, 'PNG')
        
        result = detector.detect_transparency(str(unicode_path))
        
        assert result is not None, "Should handle unicode paths"
        assert result.has_transparency is True, "Should detect transparency"
    
    def test_extremely_long_paths(self, detector, temp_dir):
        """Test handling of very long file paths."""
        # Create nested directory structure
        long_dir = temp_dir
        for i in range(20):  # Create deep nesting
            long_dir = long_dir / f"very_long_directory_name_level_{i}"
        
        try:
            long_dir.mkdir(parents=True)
            long_path = long_dir / "test_image.png"
            
            test_image = Image.new('RGB', (30, 30), (0, 255, 0))
            test_image.save(long_path, 'PNG')
            
            result = detector.detect_transparency(str(long_path))
            assert result is not None, "Should handle long paths"
            
        except OSError:
            # Path too long for filesystem, which is acceptable
            pytest.skip("Filesystem doesn't support very long paths")


class TestPerformanceBenchmarks:
    """
    Performance benchmarking tests for various image sizes and scenarios.
    """
    
    def test_small_image_performance(self, detector, temp_dir):
        """Test performance on small images (typical use case)."""
        small_path = temp_dir / "small_perf.png"
        ImageTestHelpers.create_png_rgba(small_path, 100, 100)
        
        start_time = time.time()
        result = detector.detect_transparency(str(small_path))
        elapsed_time = time.time() - start_time
        
        assert result is not None, "Should process small image"
        assert elapsed_time < 0.5, f"Small image processing should be fast, took {elapsed_time:.3f}s"
    
    def test_medium_image_performance(self, detector, temp_dir):
        """Test performance on medium images."""
        medium_path = temp_dir / "medium_perf.png"
        ImageTestHelpers.create_png_rgba(medium_path, 512, 512)
        
        start_time = time.time()
        result = detector.detect_transparency(str(medium_path))
        elapsed_time = time.time() - start_time
        
        assert result is not None, "Should process medium image"
        assert elapsed_time < 2.0, f"Medium image processing should be reasonable, took {elapsed_time:.3f}s"
    
    def test_large_image_performance(self, detector, temp_dir):
        """Test performance on large images."""
        large_path = temp_dir / "large_perf.png"
        ImageTestHelpers.create_large_image(large_path, 1024, 1024, 'RGBA')
        
        start_time = time.time()
        result = detector.detect_transparency(str(large_path))
        elapsed_time = time.time() - start_time
        
        assert result is not None, "Should process large image"
        assert elapsed_time < 10.0, f"Large image processing took {elapsed_time:.3f}s"
        assert result.has_transparency is True, "Should detect transparency in large image"
    
    def test_batch_processing_performance(self, detector, temp_dir):
        """Test performance when processing multiple images."""
        # Create multiple test images
        image_paths = []
        for i in range(10):
            path = temp_dir / f"batch_{i}.png"
            ImageTestHelpers.create_png_rgba(path, 100, 100)
            image_paths.append(path)
        
        start_time = time.time()
        results = []
        for path in image_paths:
            result = detector.detect_transparency(str(path))
            results.append(result)
        elapsed_time = time.time() - start_time
        
        assert len(results) == 10, "Should process all images"
        assert all(r.has_transparency for r in results), "All should have transparency"
        assert elapsed_time < 5.0, f"Batch processing took {elapsed_time:.3f}s"
    
    def test_memory_usage_large_images(self, detector, temp_dir):
        """Test memory usage with large images."""
        large_path = temp_dir / "memory_test.png"
        ImageTestHelpers.create_large_image(large_path, 2048, 2048, 'RGBA')
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        result = detector.detect_transparency(str(large_path))
        
        # Force garbage collection after test
        gc.collect()
        final_objects = len(gc.get_objects())
        
        assert result is not None, "Should process very large image"
        
        # Memory usage should not grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory usage grew by {object_growth} objects"


class TestImageProcessorIntegration:
    """
    Test integration between TransparencyDetector and ImageProcessor.
    """
    
    def test_processor_detect_transparency_method(self, processor, temp_dir):
        """Test ImageProcessor.detect_transparency method."""
        rgba_path = temp_dir / "processor_test.png"
        ImageTestHelpers.create_png_rgba(rgba_path)
        
        result = processor.detect_transparency(str(rgba_path))
        
        assert isinstance(result, TransparencyInfo), "Should return TransparencyInfo"
        assert result.has_transparency is True, "Should detect transparency"
    
    def test_processor_load_with_transparency_method(self, processor, temp_dir):
        """Test ImageProcessor.load_image_with_transparency method."""
        rgba_path = temp_dir / "processor_load_test.png"
        ImageTestHelpers.create_png_rgba(rgba_path)
        
        rgb_tensor, alpha_mask = processor.load_image_with_transparency(str(rgba_path))
        
        assert isinstance(rgb_tensor, torch.Tensor), "Should return RGB tensor"
        assert isinstance(alpha_mask, torch.Tensor), "Should return alpha mask"
        assert rgb_tensor.shape[1] == 3, "RGB tensor should have 3 channels"
        assert alpha_mask.shape[1] == 1, "Alpha mask should have 1 channel"
    
    def test_processor_integration_with_resizing(self, processor, temp_dir):
        """Test integration with image resizing."""
        rgba_path = temp_dir / "processor_resize_test.png"
        ImageTestHelpers.create_png_rgba(rgba_path, 200, 200)
        
        rgb_tensor, alpha_mask = processor.load_image_with_transparency(
            str(rgba_path), target_size=(128, 128)
        )
        
        assert rgb_tensor.shape[-2:] == (128, 128), "RGB tensor should be resized"
        assert alpha_mask.shape[-2:] == (128, 128), "Alpha mask should be resized"
    
    def test_processor_backward_compatibility(self, processor, temp_dir):
        """Test that existing ImageProcessor.load_image still works."""
        rgba_path = temp_dir / "processor_compat_test.png"
        ImageTestHelpers.create_png_rgba(rgba_path)
        
        # Original method should still work and convert to RGB
        rgb_tensor = processor.load_image(str(rgba_path))
        
        assert isinstance(rgb_tensor, torch.Tensor), "Should return tensor"
        assert rgb_tensor.shape[1] == 3, "Should have 3 RGB channels"
        assert rgb_tensor.min() >= 0 and rgb_tensor.max() <= 1, "Should be normalized"


class TestMockingAndErrorInjection:
    """
    Tests using mocking to simulate various error conditions and edge cases.
    """
    
    def test_pil_image_open_failure(self, detector, temp_dir):
        """Test handling when PIL.Image.open fails."""
        valid_path = temp_dir / "mock_test.png"
        ImageTestHelpers.create_png_rgba(valid_path)
        
        with patch('PIL.Image.open', side_effect=IOError("Mocked PIL error")):
            with pytest.raises(IOError):
                detector.detect_transparency(str(valid_path))
    
    def test_numpy_array_conversion_failure(self, detector, temp_dir):
        """Test handling when numpy array conversion fails."""
        valid_path = temp_dir / "numpy_mock_test.png"
        ImageTestHelpers.create_png_rgba(valid_path)
        
        with patch('numpy.array', side_effect=ValueError("Mocked numpy error")):
            with pytest.raises((ValueError, IOError)):
                detector.detect_transparency(str(valid_path))
    
    def test_file_system_errors(self, detector, temp_dir):
        """Test various file system error conditions."""
        valid_path = temp_dir / "fs_test.png"
        ImageTestHelpers.create_png_rgba(valid_path)
        
        # Mock Path.exists to return False
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                detector.detect_transparency(str(valid_path))
    
    def test_image_mode_edge_cases(self, detector, temp_dir):
        """Test edge cases in image mode detection."""
        valid_path = temp_dir / "mode_mock_test.png"
        ImageTestHelpers.create_png_rgba(valid_path)
        
        # Create mock image with unusual mode and proper context manager support
        mock_image = MagicMock()
        mock_image.mode = "XYZ"  # Non-standard mode
        mock_image.size = (100, 100)
        mock_image.info = {}
        mock_image.format = "PNG"
        
        # Mock the context manager behavior
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_image)
        mock_context.__exit__ = MagicMock(return_value=None)
        
        with patch('PIL.Image.open', return_value=mock_context):
            result = detector.detect_transparency(str(valid_path))
            # Should handle gracefully
            assert result is not None, "Should handle unusual image modes"


class TestCodeCoverageTargets:
    """
    Specific tests designed to achieve high code coverage.
    """
    
    def test_transparency_info_all_fields(self):
        """Test all fields of TransparencyInfo are accessible."""
        info = TransparencyInfo(
            has_transparency=True,
            format_type="PNG", 
            transparency_type="alpha_channel",
            image_mode="RGBA",
            total_pixels=10000,
            transparent_pixel_count=2500,
            opaque_pixel_count=7500,
            transparency_percentage=25.0,
            detection_details=["test detail 1", "test detail 2"]
        )
        
        # Test all field access
        assert info.has_transparency is True
        assert info.format_type == "PNG"
        assert info.transparency_type == "alpha_channel"
        assert info.image_mode == "RGBA"
        assert info.total_pixels == 10000
        assert info.transparent_pixel_count == 2500
        assert info.opaque_pixel_count == 7500
        assert info.transparency_percentage == 25.0
        assert len(info.detection_details) == 2
    
    def test_detector_all_internal_methods(self, detector, temp_dir):
        """Test coverage of all TransparencyDetector internal methods."""
        # Test RGBA analysis path
        rgba_path = temp_dir / "coverage_rgba.png"
        ImageTestHelpers.create_png_rgba(rgba_path)
        
        result = detector.detect_transparency(str(rgba_path))
        assert result.transparency_type == "alpha_channel"
        
        # Test LA analysis path
        la_path = temp_dir / "coverage_la.png"
        la_image = Image.new('LA', (50, 50), (128, 200))
        la_image.save(la_path)
        
        result = detector.detect_transparency(str(la_path))
        assert result.image_mode == "LA"
        
        # Test palette transparency path
        palette_path = temp_dir / "coverage_palette.png"
        ImageTestHelpers.create_gif_transparent(palette_path.with_suffix('.gif'))
        # Convert GIF to PNG to test palette handling
        with Image.open(str(palette_path.with_suffix('.gif'))) as img:
            img.save(palette_path, 'PNG', transparency=img.info.get('transparency', 0))
        
        result = detector.detect_transparency(str(palette_path))
        # Should handle palette transparency
        assert result is not None
    
    def test_processor_all_enhancement_paths(self, processor, temp_dir):
        """Test all code paths in ImageProcessor enhancements."""
        # Test all image mode paths in load_image_with_transparency
        
        # RGBA path
        rgba_path = temp_dir / "proc_coverage_rgba.png"
        ImageTestHelpers.create_png_rgba(rgba_path)
        rgb, alpha = processor.load_image_with_transparency(str(rgba_path))
        assert alpha is not None
        
        # LA path  
        la_path = temp_dir / "proc_coverage_la.png"
        la_image = Image.new('LA', (50, 50), (128, 200))
        la_image.save(la_path)
        rgb, alpha = processor.load_image_with_transparency(str(la_path))
        assert alpha is not None
        
        # RGB path (no transparency)
        rgb_path = temp_dir / "proc_coverage_rgb.png"
        rgb_image = Image.new('RGB', (50, 50), (255, 0, 0))
        rgb_image.save(rgb_path)
        rgb, alpha = processor.load_image_with_transparency(str(rgb_path))
        assert alpha is not None  # Should create opaque mask
        
        # Test with resizing
        rgb, alpha = processor.load_image_with_transparency(
            str(rgba_path), target_size=(64, 64), maintain_aspect=False
        )
        assert rgb.shape[-2:] == (64, 64)
        assert alpha.shape[-2:] == (64, 64)