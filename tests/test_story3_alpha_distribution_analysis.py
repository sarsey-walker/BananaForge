#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 3.1: Alpha Value Distribution Analysis

Following BDD Given-When-Then scenarios from tasks1.md:
- Image with partial transparency (alpha values 1-254)
- Image with minimal transparency (<1% transparent pixels)
- Image with edge feathering (anti-aliased transparent edges)
- Performance optimization for large images
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

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


class AlphaDistributionTestHelpers:
    """Helper class for creating images with specific alpha distributions."""
    
    @staticmethod
    def create_partial_transparency_image(path: Path, width: int = 100, height: int = 100):
        """Create image with alpha values between 1-254 (partial transparency)."""
        image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        
        # Create gradient with alpha values from 1 to 254
        for x in range(width):
            for y in range(height):
                # Calculate alpha value from 1-254 based on position
                alpha_val = int(1 + ((x * y) / (width * height)) * 253)
                alpha_val = max(1, min(254, alpha_val))  # Ensure range 1-254
                
                # Vary colors for visual distinction
                r = (x * 2) % 256
                g = (y * 3) % 256  
                b = 128
                
                image.putpixel((x, y), (r, g, b, alpha_val))
        
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_minimal_transparency_image(path: Path, width: int = 200, height: int = 200):
        """Create image with <1% transparent pixels."""
        image = Image.new('RGBA', (width, height), (255, 0, 0, 255))  # Opaque red
        
        total_pixels = width * height
        transparent_pixels_count = int(total_pixels * 0.005)  # 0.5% transparent
        
        # Add scattered transparent pixels
        np.random.seed(42)  # For reproducible results
        for _ in range(transparent_pixels_count):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            # Semi-transparent pixel
            image.putpixel((x, y), (0, 255, 0, 128))
        
        image.save(path, 'PNG')
        return path, transparent_pixels_count, total_pixels
    
    @staticmethod
    def create_edge_feathering_image(path: Path, width: int = 100, height: int = 100):
        """Create image with anti-aliased transparent edges (feathering)."""
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # Transparent background
        
        # Create a circle with feathered edges
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        for x in range(width):
            for y in range(height):
                # Calculate distance from center
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                if distance <= radius:
                    # Inside circle - opaque
                    alpha_val = 255
                elif distance <= radius + 10:  # Feathering zone
                    # Feathered edge - gradient alpha from 255 to 0
                    fade_factor = (radius + 10 - distance) / 10
                    alpha_val = int(255 * fade_factor)
                else:
                    # Outside - transparent
                    alpha_val = 0
                
                # Color varies with position for visual effect
                r = int(128 + 127 * (x / width))
                g = int(128 + 127 * (y / height))
                b = 64
                
                image.putpixel((x, y), (r, g, b, alpha_val))
        
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_alpha_histogram_test_image(path: Path, width: int = 100, height: int = 100):
        """Create image with known alpha value distribution for histogram testing."""
        image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        
        # Create specific alpha distribution:
        # 25% fully transparent (0)
        # 25% quarter transparent (64) 
        # 25% half transparent (128)
        # 25% three-quarter transparent (192)
        
        quarter_pixels = (width * height) // 4
        pixel_count = 0
        
        for x in range(width):
            for y in range(height):
                if pixel_count < quarter_pixels:
                    alpha = 0  # Fully transparent
                elif pixel_count < 2 * quarter_pixels:
                    alpha = 64  # Quarter opacity
                elif pixel_count < 3 * quarter_pixels:
                    alpha = 128  # Half opacity
                else:
                    alpha = 192  # Three-quarter opacity
                
                image.putpixel((x, y), (255, 128, 64, alpha))
                pixel_count += 1
        
        image.save(path, 'PNG')
        return path, {0: quarter_pixels, 64: quarter_pixels, 128: quarter_pixels, 192: quarter_pixels}


class TestScenario1_PartialTransparency:
    """
    BDD Scenario: Image with partial transparency
    Given an image with alpha values between 1-254
    When I analyze transparency distribution
    Then the system should detect semi-transparent pixels
    And calculate transparency statistics (percentage, distribution)
    """
    
    def test_detect_partial_transparency_pixels(self, detector, temp_dir):
        """Test detection of pixels with alpha values 1-254."""
        # Given: an image with alpha values between 1-254
        partial_path = temp_dir / "partial_transparency.png"
        AlphaDistributionTestHelpers.create_partial_transparency_image(partial_path)
        
        # When: I analyze transparency distribution
        result = detector.detect_transparency(str(partial_path))
        
        # Then: the system should detect semi-transparent pixels
        assert result.has_transparency is True, "Should detect partial transparency"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
        assert result.transparent_pixel_count > 0, "Should count semi-transparent pixels"
        
        # And: calculate transparency statistics
        expected_transparent = 10000  # All pixels have alpha 1-254 (not 255)
        assert result.transparent_pixel_count == expected_transparent, f"Should count all {expected_transparent} semi-transparent pixels"
        assert result.transparency_percentage == 100.0, "Should report 100% transparency for alpha 1-254"
    
    def test_partial_transparency_statistics_accuracy(self, detector, temp_dir):
        """Test accuracy of statistics for partial transparency."""
        # Given: image with known partial transparency distribution
        partial_path = temp_dir / "partial_stats.png"
        AlphaDistributionTestHelpers.create_partial_transparency_image(partial_path, 50, 50)
        
        # When: analyzing transparency statistics  
        result = detector.detect_transparency(str(partial_path))
        
        # Then: statistics should be accurate
        assert result.total_pixels == 50 * 50, "Should count total pixels correctly"
        assert result.transparent_pixel_count + result.opaque_pixel_count == result.total_pixels, "Counts should sum to total"
        assert result.transparency_percentage >= 0 and result.transparency_percentage <= 100, "Percentage should be valid"
        
        # For alpha values 1-254, all pixels are considered transparent
        assert result.opaque_pixel_count == 0, "No pixels should be fully opaque (alpha 255)"
        assert result.transparent_pixel_count == 2500, "All 2500 pixels should be semi-transparent"


class TestScenario2_MinimalTransparency:
    """
    BDD Scenario: Image with minimal transparency
    Given an image with <1% transparent pixels
    When I analyze the transparency
    Then the system should still detect and report the transparency
    And provide statistics about the transparent region
    """
    
    def test_detect_minimal_transparency(self, detector, temp_dir):
        """Test detection of <1% transparent pixels."""
        # Given: an image with <1% transparent pixels
        minimal_path = temp_dir / "minimal_transparency.png"
        path, transparent_count, total_pixels = AlphaDistributionTestHelpers.create_minimal_transparency_image(minimal_path)
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(path))
        
        # Then: the system should still detect and report the transparency
        assert result.has_transparency is True, "Should detect even minimal transparency"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
        
        # And: provide statistics about the transparent region
        assert result.transparent_pixel_count >= transparent_count, f"Should detect at least {transparent_count} transparent pixels"
        expected_percentage = (transparent_count / total_pixels) * 100
        assert result.transparency_percentage >= expected_percentage * 0.5, f"Should report approximately {expected_percentage:.2f}% transparency"
        assert result.transparency_percentage < 1.0, "Should be less than 1% transparency"
    
    def test_minimal_transparency_precision(self, detector, temp_dir):
        """Test precision of minimal transparency detection."""
        # Given: image with very small amount of transparency
        minimal_path = temp_dir / "precision_test.png"
        
        # Create image with exactly 1 transparent pixel out of 10000
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 255))  # Opaque red
        image.putpixel((50, 50), (0, 255, 0, 128))  # Single semi-transparent pixel
        image.save(minimal_path, 'PNG')
        
        # When: analyzing the transparency
        result = detector.detect_transparency(str(minimal_path))
        
        # Then: should detect the single transparent pixel
        assert result.has_transparency is True, "Should detect single transparent pixel"
        assert result.transparent_pixel_count >= 1, "Should count at least 1 transparent pixel"
        assert result.transparency_percentage >= 0.01, "Should report at least 0.01% transparency"
        assert result.transparency_percentage < 0.1, "Should be less than 0.1% transparency"


class TestScenario3_EdgeFeathering:
    """
    BDD Scenario: Image with edge feathering
    Given an image with anti-aliased transparent edges
    When I analyze alpha values
    Then the system should detect the feathering effect
    And report it as containing transparency requiring attention
    """
    
    def test_detect_edge_feathering(self, detector, temp_dir):
        """Test detection of anti-aliased transparent edges."""
        # Given: an image with anti-aliased transparent edges
        feather_path = temp_dir / "edge_feathering.png"
        AlphaDistributionTestHelpers.create_edge_feathering_image(feather_path)
        
        # When: I analyze alpha values
        result = detector.detect_transparency(str(feather_path))
        
        # Then: the system should detect the feathering effect
        assert result.has_transparency is True, "Should detect feathering transparency"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
        
        # And: report it as containing transparency requiring attention
        assert result.transparent_pixel_count > 0, "Should count feathered edge pixels as transparent"
        assert 10 < result.transparency_percentage < 90, "Feathered image should have moderate transparency"
        assert len([d for d in result.detection_details if "alpha" in d.lower()]) > 0, "Should mention alpha channel in details"
    
    def test_feathering_gradient_detection(self, detector, temp_dir):
        """Test that gradient feathering is properly detected."""
        # Given: image with smooth alpha gradient (feathering)
        gradient_path = temp_dir / "gradient_feather.png"
        
        # Create smooth gradient from opaque to transparent
        image = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
        for x in range(100):
            alpha_val = int(255 * (1 - x / 100))  # Gradient from 255 to 0
            for y in range(100):
                image.putpixel((x, y), (255, 128, 64, alpha_val))
        image.save(gradient_path, 'PNG')
        
        # When: analyzing the feathered gradient
        result = detector.detect_transparency(str(gradient_path))
        
        # Then: should detect the gradient as transparency
        assert result.has_transparency is True, "Should detect gradient transparency"
        assert result.transparent_pixel_count > 5000, "Should detect most pixels as having some transparency"
        assert result.transparency_percentage > 50, "Gradient should show significant transparency"


class TestAlphaDistributionAnalysis:
    """
    Test advanced alpha distribution analysis capabilities.
    """
    
    def test_alpha_histogram_analysis(self, detector, temp_dir):
        """Test analysis of alpha value distribution."""
        # Given: image with known alpha distribution
        histogram_path = temp_dir / "alpha_histogram.png"
        path, expected_distribution = AlphaDistributionTestHelpers.create_alpha_histogram_test_image(histogram_path)
        
        # When: analyzing alpha distribution
        result = detector.detect_transparency(str(path))
        
        # Then: should detect transparency accurately
        assert result.has_transparency is True, "Should detect transparency in histogram test image"
        assert result.total_pixels == 10000, "Should count all pixels correctly"
        
        # Should count pixels with alpha < 255 as transparent
        expected_transparent = expected_distribution[0] + expected_distribution[64] + expected_distribution[128] + expected_distribution[192]
        assert result.transparent_pixel_count == expected_transparent, f"Should count {expected_transparent} transparent pixels"
    
    def test_edge_case_alpha_values(self, detector, temp_dir):
        """Test edge cases in alpha value detection."""
        edge_cases = [
            (1, "minimal_alpha"),      # Barely transparent
            (127, "half_alpha"),       # Half transparent  
            (254, "almost_opaque"),    # Almost opaque
        ]
        
        for alpha_val, test_name in edge_cases:
            # Given: image with specific alpha value
            edge_path = temp_dir / f"{test_name}.png"
            image = Image.new('RGBA', (50, 50), (255, 0, 0, alpha_val))
            image.save(edge_path, 'PNG')
            
            # When: analyzing the alpha value
            result = detector.detect_transparency(str(edge_path))
            
            # Then: should detect correctly based on alpha < 255 threshold
            if alpha_val < 255:
                assert result.has_transparency is True, f"Should detect transparency for alpha {alpha_val}"
                assert result.transparent_pixel_count == 2500, f"All pixels should be transparent for alpha {alpha_val}"
            else:
                assert result.has_transparency is False, f"Should not detect transparency for alpha {alpha_val}"
                assert result.transparent_pixel_count == 0, f"No pixels should be transparent for alpha {alpha_val}"


class TestPerformanceOptimization:
    """
    Test performance optimization for large images.
    """
    
    def test_large_image_performance(self, detector, temp_dir):
        """Test that large images are processed efficiently."""
        # Given: large image with transparency
        large_path = temp_dir / "large_performance.png"
        
        # Create 512x512 image (262,144 pixels)
        large_image = Image.new('RGBA', (512, 512), (255, 255, 255, 255))
        
        # Add some transparency pattern for realism
        for x in range(0, 512, 10):
            for y in range(0, 512, 10):
                # Create transparent patches
                for dx in range(5):
                    for dy in range(5):
                        if x + dx < 512 and y + dy < 512:
                            large_image.putpixel((x + dx, y + dy), (255, 0, 0, 128))
        
        large_image.save(large_path, 'PNG')
        
        # When: analyzing large image
        import time
        start_time = time.time()
        result = detector.detect_transparency(str(large_path))
        elapsed_time = time.time() - start_time
        
        # Then: should process efficiently
        assert result is not None, "Should process large image successfully"
        assert elapsed_time < 5.0, f"Large image processing should be fast, took {elapsed_time:.2f}s"
        assert result.has_transparency is True, "Should detect transparency in large image"
        assert result.total_pixels == 512 * 512, "Should count all pixels in large image"
    
    def test_memory_efficiency_large_images(self, detector, temp_dir):
        """Test memory efficiency with large images."""
        # Given: very large image
        large_path = temp_dir / "memory_test.png"
        
        # Create 1024x1024 image with alpha gradient
        large_image = Image.new('RGBA', (256, 256), (255, 255, 255, 255))  # Smaller for CI
        
        # Add alpha gradient
        for x in range(256):
            alpha_val = int((x / 256) * 255)
            for y in range(256):
                large_image.putpixel((x, y), (255, 0, 0, alpha_val))
        
        large_image.save(large_path, 'PNG')
        
        # When: processing large image
        result = detector.detect_transparency(str(large_path))
        
        # Then: should handle memory efficiently
        assert result is not None, "Should process large image without memory issues"
        assert result.has_transparency is True, "Should detect transparency"
        # Memory efficiency is tested by successful completion without errors


class TestIntegrationWithImageProcessor:
    """
    Test integration with enhanced ImageProcessor for alpha distribution analysis.
    """
    
    def test_processor_alpha_distribution_integration(self, temp_dir):
        """Test that ImageProcessor can work with alpha distribution analysis."""
        from bananaforge.image.processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Given: image with complex alpha distribution
        complex_path = temp_dir / "complex_alpha.png"
        AlphaDistributionTestHelpers.create_partial_transparency_image(complex_path)
        
        # When: processing with ImageProcessor
        transparency_info = processor.detect_transparency(str(complex_path))
        rgb_tensor, alpha_mask = processor.load_image_with_transparency(str(complex_path))
        
        # Then: should handle complex transparency
        assert transparency_info.has_transparency is True, "Processor should detect transparency"
        assert rgb_tensor is not None, "Should return RGB tensor"
        assert alpha_mask is not None, "Should return alpha mask"
        assert alpha_mask.shape[1] == 1, "Alpha mask should have 1 channel"
    
    def test_backward_compatibility_with_distribution_analysis(self, temp_dir):
        """Test that existing ImageProcessor methods work with new analysis."""
        from bananaforge.image.processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Given: image with transparency
        test_path = temp_dir / "compat_test.png"
        AlphaDistributionTestHelpers.create_minimal_transparency_image(test_path)
        
        # When: using existing load_image method
        rgb_tensor = processor.load_image(str(test_path))
        
        # Then: should work without errors (backward compatibility)
        assert rgb_tensor is not None, "Should load image successfully"
        assert rgb_tensor.shape[1] == 3, "Should return RGB tensor"
        # No transparency preserved in RGB mode - expected behavior


class TestErrorHandlingAndEdgeCases:
    """
    Test error handling and edge cases in alpha distribution analysis.
    """
    
    def test_zero_alpha_image(self, detector, temp_dir):
        """Test image with all pixels having alpha = 0."""
        # Given: fully transparent image
        zero_path = temp_dir / "zero_alpha.png"
        zero_image = Image.new('RGBA', (50, 50), (255, 0, 0, 0))  # All transparent
        zero_image.save(zero_path, 'PNG')
        
        # When: analyzing fully transparent image
        result = detector.detect_transparency(str(zero_path))
        
        # Then: should detect transparency correctly
        assert result.has_transparency is True, "Should detect full transparency"
        assert result.transparent_pixel_count == 2500, "All pixels should be transparent"
        assert result.transparency_percentage == 100.0, "Should report 100% transparency"
    
    def test_single_opaque_pixel(self, detector, temp_dir):
        """Test image with single opaque pixel among transparent ones."""
        # Given: mostly transparent image with one opaque pixel
        single_path = temp_dir / "single_opaque.png"
        single_image = Image.new('RGBA', (50, 50), (255, 0, 0, 0))  # All transparent
        single_image.putpixel((25, 25), (0, 255, 0, 255))  # Single opaque pixel
        single_image.save(single_path, 'PNG')
        
        # When: analyzing the image
        result = detector.detect_transparency(str(single_path))
        
        # Then: should detect transparency and count correctly
        assert result.has_transparency is True, "Should detect transparency"
        assert result.transparent_pixel_count == 2499, "Should count 2499 transparent pixels"
        assert result.opaque_pixel_count == 1, "Should count 1 opaque pixel"
        assert abs(result.transparency_percentage - 99.96) < 0.1, "Should report ~99.96% transparency"