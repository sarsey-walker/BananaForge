#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 2.2: WebP and Modern Format Support

Following BDD Given-When-Then scenarios from tasks1.md:
- WebP with alpha channel detection
- TIFF with alpha channel support
- Extensible architecture for future format support
- Consistent handling across all formats
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Import the classes we're testing
from bananaforge.image.transparency_detector import TransparencyDetector, TransparencyInfo


@pytest.fixture
def detector():
    """Create a TransparencyDetector instance for testing."""
    return TransparencyDetector()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class ModernFormatTestHelpers:
    """Helper class for creating various modern format test images."""
    
    @staticmethod
    def create_webp_with_alpha(path: Path, width: int = 100, height: int = 100, pattern: str = "gradient"):
        """Create WebP image with alpha channel."""
        # Create RGBA image
        image = Image.new('RGBA', (width, height))
        
        if pattern == "gradient":
            # Alpha gradient from transparent to opaque
            for x in range(width):
                for y in range(height):
                    alpha_val = int((x / width) * 255)
                    red_val = int((y / height) * 255)
                    image.putpixel((x, y), (red_val, 128, 64, alpha_val))
        
        elif pattern == "checkerboard":
            # Checkerboard transparency pattern
            for x in range(width):
                for y in range(height):
                    if (x + y) % 4 == 0:
                        alpha_val = 0  # Transparent
                    elif (x + y) % 4 == 1:
                        alpha_val = 85  # Semi-transparent
                    elif (x + y) % 4 == 2:
                        alpha_val = 170  # Semi-transparent
                    else:
                        alpha_val = 255  # Opaque
                    
                    color_val = (x + y) % 256
                    image.putpixel((x, y), (color_val, 255 - color_val, 128, alpha_val))
        
        elif pattern == "border":
            # Transparent border, opaque center
            for x in range(width):
                for y in range(height):
                    if x < 15 or x >= width-15 or y < 15 or y >= height-15:
                        alpha_val = 50  # Semi-transparent border
                    else:
                        alpha_val = 255  # Opaque center
                    
                    image.putpixel((x, y), (255, x % 256, y % 256, alpha_val))
        
        # Try to save as WebP, fallback to PNG if WebP not supported
        try:
            image.save(path, 'WebP', lossless=True)
            return path, 'WEBP'  # Use consistent format name
        except (OSError, ValueError, KeyError):
            # WebP not supported, save as PNG for testing
            png_path = path.with_suffix('.png')
            image.save(png_path, 'PNG')
            return png_path, 'PNG'
    
    @staticmethod
    def create_webp_lossy_with_alpha(path: Path, width: int = 100, height: int = 100):
        """Create lossy WebP with alpha channel."""
        image = Image.new('RGBA', (width, height))
        
        # Create complex pattern for lossy compression testing
        for x in range(width):
            for y in range(height):
                # Complex color pattern
                r = int(128 + 127 * np.sin(x * 0.1))
                g = int(128 + 127 * np.cos(y * 0.1))
                b = int(128 + 127 * np.sin((x + y) * 0.05))
                
                # Alpha pattern
                alpha = int(128 + 127 * np.sin((x - y) * 0.08))
                
                image.putpixel((x, y), (r, g, b, alpha))
        
        try:
            # Save as lossy WebP
            image.save(path, 'WebP', lossless=False, quality=80)
            return path, 'WEBP'  # Use consistent format name
        except (OSError, ValueError, KeyError):
            png_path = path.with_suffix('.png')
            image.save(png_path, 'PNG')
            return png_path, 'PNG'
    
    @staticmethod
    def create_tiff_with_alpha(path: Path, width: int = 100, height: int = 100, pattern: str = "half"):
        """Create TIFF image with alpha channel."""
        image = Image.new('RGBA', (width, height))
        
        if pattern == "half":
            # Half transparent, half opaque
            for x in range(width):
                for y in range(height):
                    if x < width // 2:
                        alpha_val = 128  # Semi-transparent
                    else:
                        alpha_val = 255  # Opaque
                    
                    image.putpixel((x, y), (x % 256, y % 256, (x + y) % 256, alpha_val))
        
        elif pattern == "radial":
            # Radial alpha gradient
            center_x, center_y = width // 2, height // 2
            max_distance = ((width // 2) ** 2 + (height // 2) ** 2) ** 0.5
            
            for x in range(width):
                for y in range(height):
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    alpha_val = max(0, min(255, int(255 * (1 - distance / max_distance))))
                    
                    image.putpixel((x, y), (255, 128, 64, alpha_val))
        
        image.save(path, 'TIFF')
        return path
    
    @staticmethod
    def create_opaque_webp(path: Path, width: int = 100, height: int = 100):
        """Create opaque WebP without transparency."""
        image = Image.new('RGB', (width, height))
        
        # Create colorful pattern
        for x in range(width):
            for y in range(height):
                r = (x * 3) % 256
                g = (y * 5) % 256
                b = ((x + y) * 2) % 256
                image.putpixel((x, y), (r, g, b))
        
        try:
            image.save(path, 'WebP')
            return path, 'WEBP'  # Use consistent format name
        except (OSError, ValueError, KeyError):
            png_path = path.with_suffix('.png')
            image.save(png_path, 'PNG')
            return png_path, 'PNG'
    
    @staticmethod
    def create_opaque_tiff(path: Path, width: int = 100, height: int = 100):
        """Create opaque TIFF without transparency."""
        image = Image.new('RGB', (width, height))
        
        for x in range(width):
            for y in range(height):
                image.putpixel((x, y), (x % 256, y % 256, 128))
        
        image.save(path, 'TIFF')
        return path


class TestScenario1_WebPWithAlphaChannel:
    """
    BDD Scenario: WebP with alpha channel
    Given a WebP image containing an alpha channel
    When I load the image for analysis
    Then the system should detect WebP transparency
    And report the alpha channel presence
    """
    
    def test_detect_webp_gradient_transparency(self, detector, temp_dir):
        """Test detection of WebP transparency with gradient pattern."""
        # Given: a WebP image containing an alpha channel
        webp_path = temp_dir / "webp_gradient.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_with_alpha(
            webp_path, pattern="gradient"
        )
        
        # When: I load the image for analysis
        result = detector.detect_transparency(str(actual_path))
        
        # Then: the system should detect WebP transparency
        assert result is not None, "Should return TransparencyInfo object"
        assert result.has_transparency is True, "Should detect transparency in WebP/PNG"
        assert result.format_type in ["WEBP", "PNG"], f"Should identify {actual_format} format"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
        
        # And: report the alpha channel presence
        assert result.transparent_pixel_count > 0, "Should count transparent pixels"
        # Gradient pattern naturally creates high transparency (alpha values 0-255)
        assert result.transparency_percentage > 0, "Should calculate transparency percentage"
        assert len(result.detection_details) > 0, "Should provide detection details"
        assert any("alpha" in detail.lower() for detail in result.detection_details), "Should mention alpha channel"
    
    def test_detect_webp_checkerboard_transparency(self, detector, temp_dir):
        """Test detection of WebP transparency with checkerboard pattern."""
        # Given: a WebP with complex alpha pattern
        webp_path = temp_dir / "webp_checkerboard.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_with_alpha(
            webp_path, pattern="checkerboard"
        )
        
        # When: I analyze the image for transparency
        result = detector.detect_transparency(str(actual_path))
        
        # Then: should detect complex transparency patterns
        assert result.has_transparency is True, "Should detect checkerboard transparency"
        assert result.format_type in ["WEBP", "PNG"], f"Should identify {actual_format} format"
        assert result.image_mode in ["RGBA", "LA"], "Should detect alpha-capable mode"
        
        # Checkerboard should have approximately 25% fully transparent pixels
        # (plus semi-transparent pixels)
        assert result.transparent_pixel_count > 0, "Should have transparent pixels"
        assert result.transparency_percentage > 15, "Should have significant transparency"
    
    def test_detect_webp_border_transparency(self, detector, temp_dir):
        """Test detection of WebP transparency with border pattern."""
        # Given: a WebP with transparent border
        webp_path = temp_dir / "webp_border.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_with_alpha(
            webp_path, pattern="border"
        )
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(actual_path))
        
        # Then: should detect border transparency
        assert result.has_transparency is True, "Should detect border transparency"
        assert result.format_type in ["WEBP", "PNG"], f"Should identify {actual_format} format"
        
        # Calculate expected transparent border area
        total_pixels = 100 * 100  # Default size
        border_pixels = total_pixels - (70 * 70)  # 15px border on each side
        expected_percentage = (border_pixels / total_pixels) * 100
        
        # Should be close to expected (with some tolerance for semi-transparency)
        actual_percentage = result.transparency_percentage
        assert abs(actual_percentage - expected_percentage) < 10, f"Expected ~{expected_percentage}% transparency, got {actual_percentage}%"
    
    def test_webp_transparency_statistics_accuracy(self, detector, temp_dir):
        """Test accuracy of WebP transparency statistics."""
        # Given: a WebP with known transparency distribution
        webp_path = temp_dir / "webp_stats.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_with_alpha(
            webp_path, 200, 200, pattern="gradient"
        )
        
        # When: analyzing transparency statistics
        result = detector.detect_transparency(str(actual_path))
        
        # Then: statistics should be accurate
        assert result.total_pixels == 200 * 200, "Should count total pixels correctly"
        assert result.transparent_pixel_count > 0, "Should have transparent pixels"
        # Gradient pattern creates alpha values 0-255, so most pixels have alpha < 255 (transparent)
        assert result.transparent_pixel_count + result.opaque_pixel_count == result.total_pixels, "Pixel counts should sum to total"
        
        # Gradient pattern creates pixels with alpha < 255, which counts as transparency
        assert result.transparency_percentage > 50, "Gradient should have significant transparency"


class TestScenario2_WebPLossyVsLossless:
    """
    Test WebP lossless vs lossy transparency handling.
    """
    
    def test_webp_lossless_transparency_preservation(self, detector, temp_dir):
        """Test that lossless WebP preserves transparency accurately."""
        # Given: a lossless WebP with alpha
        webp_path = temp_dir / "webp_lossless.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_with_alpha(
            webp_path, pattern="checkerboard"
        )
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(actual_path))
        
        # Then: should preserve transparency accurately
        if actual_format == "WEBP":
            assert result.format_type == "WEBP", "Should identify as WebP"
            assert result.has_transparency is True, "Lossless WebP should preserve transparency"
        else:
            # Fallback to PNG - still should work
            assert result.format_type == "PNG", "Should identify PNG fallback"
            assert result.has_transparency is True, "PNG fallback should have transparency"
    
    def test_webp_lossy_transparency_handling(self, detector, temp_dir):
        """Test transparency handling in lossy WebP."""
        # Given: a lossy WebP with alpha
        webp_path = temp_dir / "webp_lossy.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_lossy_with_alpha(webp_path)
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(actual_path))
        
        # Then: should handle lossy transparency
        assert result is not None, "Should handle lossy WebP/PNG"
        if actual_format == "WEBP":
            assert result.format_type == "WEBP", "Should identify as WebP"
        else:
            assert result.format_type == "PNG", "Should identify PNG fallback"
        
        # Lossy compression might affect transparency, but should still be detected
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel"


class TestScenario3_TIFFWithAlphaChannel:
    """
    BDD Scenario: TIFF with alpha channel
    Given a TIFF image with RGBA configuration
    When I process the image
    Then the system should identify TIFF transparency
    And handle it consistently with other formats
    """
    
    def test_detect_tiff_half_transparency(self, detector, temp_dir):
        """Test detection of TIFF transparency with half pattern."""
        # Given: a TIFF image with RGBA configuration
        tiff_path = temp_dir / "tiff_half.tiff"
        ModernFormatTestHelpers.create_tiff_with_alpha(tiff_path, pattern="half")
        
        # When: I process the image
        result = detector.detect_transparency(str(tiff_path))
        
        # Then: the system should identify TIFF transparency
        assert result is not None, "Should return TransparencyInfo object"
        assert result.has_transparency is True, "Should detect transparency in TIFF"
        assert result.format_type == "TIFF", "Should identify TIFF format"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
        assert result.image_mode == "RGBA", "Should detect RGBA mode"
        
        # And: handle it consistently with other formats
        assert result.transparent_pixel_count > 0, "Should count transparent pixels"
        assert 0 < result.transparency_percentage < 100, "Should calculate transparency percentage"
        assert len(result.detection_details) > 0, "Should provide detection details"
        
        # Half pattern should have approximately 50% transparency
        expected_percentage = 50.0
        actual_percentage = result.transparency_percentage
        assert abs(actual_percentage - expected_percentage) < 10, f"Expected ~50% transparency, got {actual_percentage}%"
    
    def test_detect_tiff_radial_transparency(self, detector, temp_dir):
        """Test detection of TIFF transparency with radial pattern."""
        # Given: a TIFF with radial alpha gradient
        tiff_path = temp_dir / "tiff_radial.tiff"
        ModernFormatTestHelpers.create_tiff_with_alpha(tiff_path, pattern="radial")
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(tiff_path))
        
        # Then: should detect radial transparency
        assert result.has_transparency is True, "Should detect radial transparency"
        assert result.format_type == "TIFF", "Should identify TIFF format"
        assert result.image_mode == "RGBA", "Should detect RGBA mode"
        
        # Radial pattern should have some transparency
        assert result.transparency_percentage > 10, "Radial pattern should have some transparency"
    
    def test_tiff_no_transparency_detection(self, detector, temp_dir):
        """Test TIFF without transparency."""
        # Given: a TIFF without transparency
        tiff_path = temp_dir / "tiff_opaque.tiff"
        ModernFormatTestHelpers.create_opaque_tiff(tiff_path)
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(tiff_path))
        
        # Then: should detect no transparency
        assert result.has_transparency is False, "Should detect no transparency in opaque TIFF"
        assert result.format_type == "TIFF", "Should identify TIFF format"
        assert result.transparency_type == "none", "Should identify no transparency"
        assert result.transparent_pixel_count == 0, "Should have no transparent pixels"
        assert result.transparency_percentage == 0.0, "Should have 0% transparency"


class TestScenario4_WebPNoTransparency:
    """Test WebP files without transparency."""
    
    def test_webp_opaque_detection(self, detector, temp_dir):
        """Test detection when WebP has no transparency."""
        # Given: a WebP without transparency
        webp_path = temp_dir / "webp_opaque.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_opaque_webp(webp_path)
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(actual_path))
        
        # Then: should detect no transparency
        assert result.has_transparency is False, "Should detect no transparency in opaque WebP/PNG"
        assert result.format_type in ["WEBP", "PNG"], f"Should identify {actual_format} format"
        assert result.transparency_type == "none", "Should identify no transparency"
        assert result.transparent_pixel_count == 0, "Should have no transparent pixels"
        assert result.transparency_percentage == 0.0, "Should have 0% transparency"


class TestExtensibleArchitecture:
    """
    Test extensible architecture for future format support.
    """
    
    def test_format_specific_routing(self, detector, temp_dir):
        """Test that different formats are routed to appropriate handlers."""
        # Create images in different formats
        formats_to_test = []
        
        # PNG
        png_path = temp_dir / "test.png"
        png_image = Image.new('RGBA', (50, 50), (255, 0, 0, 128))
        png_image.save(png_path, 'PNG')
        formats_to_test.append((png_path, "PNG"))
        
        # GIF
        gif_path = temp_dir / "test.gif"
        gif_image = Image.new('P', (50, 50))
        palette = [i for i in range(256) for _ in range(3)]
        gif_image.putpalette(palette)
        for x in range(50):
            for y in range(50):
                gif_image.putpixel((x, y), (x + y) % 2)
        gif_image.save(gif_path, 'GIF', transparency=0)
        formats_to_test.append((gif_path, "GIF"))
        
        # TIFF
        tiff_path = temp_dir / "test.tiff"
        tiff_image = Image.new('RGBA', (50, 50), (0, 255, 0, 200))
        tiff_image.save(tiff_path, 'TIFF')
        formats_to_test.append((tiff_path, "TIFF"))
        
        # WebP (or PNG fallback)
        webp_path = temp_dir / "test.webp"
        actual_webp_path, webp_format = ModernFormatTestHelpers.create_webp_with_alpha(webp_path)
        formats_to_test.append((actual_webp_path, webp_format))
        
        # Test each format
        for file_path, expected_format in formats_to_test:
            result = detector.detect_transparency(str(file_path))
            assert result.format_type == expected_format, f"Should identify {expected_format} format"
            assert result.has_transparency is True, f"Should detect transparency in {expected_format}"
    
    def test_consistent_transparency_info_structure(self, detector, temp_dir):
        """Test that all formats return consistent TransparencyInfo structure."""
        # Create transparent images in different formats
        test_images = []
        
        # PNG RGBA
        png_path = temp_dir / "consistency_png.png"
        png_image = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        png_image.save(png_path, 'PNG')
        test_images.append(png_path)
        
        # TIFF RGBA
        tiff_path = temp_dir / "consistency_tiff.tiff"
        ModernFormatTestHelpers.create_tiff_with_alpha(tiff_path)
        test_images.append(tiff_path)
        
        # WebP/PNG
        webp_path = temp_dir / "consistency_webp.webp"
        actual_webp_path, _ = ModernFormatTestHelpers.create_webp_with_alpha(webp_path)
        test_images.append(actual_webp_path)
        
        # Test consistency across all formats
        results = []
        for image_path in test_images:
            result = detector.detect_transparency(str(image_path))
            results.append(result)
        
        # All should have consistent structure
        required_fields = [
            'has_transparency', 'format_type', 'transparency_type', 'image_mode',
            'total_pixels', 'transparent_pixel_count', 'opaque_pixel_count',
            'transparency_percentage', 'detection_details'
        ]
        
        for result in results:
            for field in required_fields:
                assert hasattr(result, field), f"TransparencyInfo should have {field} field"
                assert getattr(result, field) is not None, f"{field} should not be None"
        
        # All should detect transparency
        for result in results:
            assert result.has_transparency is True, "All test images should have transparency"
            assert result.transparency_type == "alpha_channel", "All should use alpha channel transparency"


class TestFormatFallbackHandling:
    """Test fallback handling for unsupported formats."""
    
    def test_webp_fallback_to_png(self, detector, temp_dir):
        """Test fallback when WebP is not supported."""
        webp_path = temp_dir / "fallback_test.webp"
        
        # This will create PNG if WebP is not supported
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_with_alpha(webp_path)
        
        result = detector.detect_transparency(str(actual_path))
        
        # Should work regardless of whether it's WebP or PNG
        assert result is not None, "Should handle WebP or PNG fallback"
        assert result.format_type in ["WEBP", "PNG"], "Should identify correct format"
        assert result.has_transparency is True, "Should detect transparency in fallback format"
    
    def test_unknown_format_graceful_handling(self, detector, temp_dir):
        """Test graceful handling of unknown/unsupported formats."""
        # Create a file with unknown extension but valid PNG content
        unknown_path = temp_dir / "test.xyz"
        test_image = Image.new('RGBA', (50, 50), (255, 0, 0, 128))
        test_image.save(unknown_path, 'PNG')  # PNG content but .xyz extension
        
        # Should still work based on content, not extension
        result = detector.detect_transparency(str(unknown_path))
        assert result is not None, "Should handle files based on content"
        assert result.has_transparency is True, "Should detect transparency regardless of extension"


class TestModernFormatIntegration:
    """Test integration with the broader transparency detection system."""
    
    def test_modern_formats_with_imageprocessor(self, temp_dir):
        """Test that modern formats work with ImageProcessor integration."""
        from bananaforge.image.processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Test TIFF
        tiff_path = temp_dir / "integration_tiff.tiff"
        ModernFormatTestHelpers.create_tiff_with_alpha(tiff_path)
        
        # Should work with ImageProcessor
        tiff_result = processor.detect_transparency(str(tiff_path))
        assert tiff_result.format_type == "TIFF", "ImageProcessor should handle TIFF"
        assert tiff_result.has_transparency is True, "Should detect TIFF transparency"
        
        # Should work with load_image_with_transparency
        rgb_tensor, alpha_mask = processor.load_image_with_transparency(str(tiff_path))
        assert rgb_tensor.shape[1] == 3, "Should return RGB tensor"
        assert alpha_mask.shape[1] == 1, "Should return alpha mask"
        
        # Test WebP/PNG fallback
        webp_path = temp_dir / "integration_webp.webp"
        actual_path, _ = ModernFormatTestHelpers.create_webp_with_alpha(webp_path)
        
        webp_result = processor.detect_transparency(str(actual_path))
        assert webp_result.has_transparency is True, "Should detect WebP/PNG transparency"
        
        rgb_tensor, alpha_mask = processor.load_image_with_transparency(str(actual_path))
        assert rgb_tensor is not None, "Should process WebP/PNG with ImageProcessor"
        assert alpha_mask is not None, "Should extract alpha from WebP/PNG"
    
    def test_format_detection_accuracy(self, detector, temp_dir):
        """Test that format detection is accurate across all supported formats."""
        format_tests = []
        
        # Create test images
        formats = ["PNG", "GIF", "TIFF"]
        
        for fmt in formats:
            if fmt == "PNG":
                path = temp_dir / f"format_test.{fmt.lower()}"
                image = Image.new('RGBA', (50, 50), (255, 0, 0, 128))
                image.save(path, fmt)
                format_tests.append((path, fmt))
            
            elif fmt == "GIF":
                path = temp_dir / f"format_test.{fmt.lower()}"
                image = Image.new('P', (50, 50))
                palette = [i for i in range(256) for _ in range(3)]
                image.putpalette(palette)
                image.save(path, fmt, transparency=0)
                format_tests.append((path, fmt))
            
            elif fmt == "TIFF":
                path = temp_dir / f"format_test.{fmt.lower()}"
                ModernFormatTestHelpers.create_tiff_with_alpha(path)
                format_tests.append((path, fmt))
        
        # WebP (with fallback)
        webp_path = temp_dir / "format_test.webp"
        actual_path, actual_format = ModernFormatTestHelpers.create_webp_with_alpha(webp_path)
        format_tests.append((actual_path, actual_format))
        
        # Test format detection
        for file_path, expected_format in format_tests:
            result = detector.detect_transparency(str(file_path))
            assert result.format_type == expected_format, f"Should correctly identify {expected_format} format"