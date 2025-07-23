#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 4.2: Transparency Statistics Reporting

Following BDD Given-When-Then scenarios from tasks1.md:
- Detailed transparency report
- Transparency visualization guidance
- Calculate and display transparency percentage
- Show alpha value distribution statistics
- Provide contextual recommendations based on analysis
"""

import pytest
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Import the classes we're testing
from bananaforge.image.transparency_detector import TransparencyDetector, TransparencyInfo
from bananaforge.image.transparency_reporter import TransparencyReporter, TransparencyStatistics


@pytest.fixture
def reporter():
    """Create a TransparencyReporter instance for testing."""
    return TransparencyReporter()


@pytest.fixture
def detector():
    """Create a TransparencyDetector instance for testing."""
    return TransparencyDetector()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class StatisticsTestHelpers:
    """Helper class for creating test images with specific transparency patterns."""
    
    @staticmethod
    def create_complex_transparency_image(path: Path):
        """Create image with complex transparency patterns for testing."""
        image = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
        
        # Create multiple transparency patterns
        # 1. Gradient area (top-left)
        for x in range(25):
            for y in range(25):
                alpha_val = int(255 * (x + y) / 50)
                image.putpixel((x, y), (255, 0, 0, alpha_val))
        
        # 2. Solid transparent area (top-right)
        for x in range(75, 100):
            for y in range(25):
                image.putpixel((x, y), (0, 255, 0, 0))
        
        # 3. Semi-transparent area (bottom-left)
        for x in range(25):
            for y in range(75, 100):
                image.putpixel((x, y), (0, 0, 255, 128))
        
        # 4. Mixed transparency (bottom-right) 
        for x in range(75, 100):
            for y in range(75, 100):
                alpha_val = 64 if (x + y) % 3 == 0 else 192
                image.putpixel((x, y), (255, 255, 0, alpha_val))
        
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_edge_transparency_image(path: Path):
        """Create image with transparency focused on edges."""
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 255))  # Opaque center
        
        # Create transparent border
        border_width = 10
        for x in range(100):
            for y in range(100):
                # Distance from edge
                edge_distance = min(x, y, 99-x, 99-y)
                if edge_distance < border_width:
                    # Gradient from transparent at edge to opaque
                    alpha_val = int(255 * (edge_distance / border_width))
                    image.putpixel((x, y), (255, 0, 0, alpha_val))
        
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_scattered_transparency_image(path: Path):
        """Create image with scattered transparency pattern."""
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 255))  # Opaque base
        
        # Add scattered transparent pixels in a pattern
        np.random.seed(42)  # For reproducible results
        for _ in range(500):  # 5% of pixels
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            alpha_val = np.random.randint(0, 128)  # Random transparency
            image.putpixel((x, y), (0, 255, 0, alpha_val))
        
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_uniform_transparency_image(path: Path):
        """Create image with uniform transparency."""
        # All pixels have same alpha value
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 180))  # Uniform semi-transparent
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_minimal_alpha_variation_image(path: Path):
        """Create image with minimal alpha variation for statistics testing."""
        image = Image.new('RGBA', (50, 50), (255, 255, 255, 255))
        
        # Create known alpha distribution
        # 25% pixels with alpha 255 (opaque)
        # 25% pixels with alpha 200
        # 25% pixels with alpha 100  
        # 25% pixels with alpha 50
        
        pixel_count = 0
        total_pixels = 50 * 50
        quarter = total_pixels // 4
        
        alpha_values = [255, 200, 100, 50]
        
        for x in range(50):
            for y in range(50):
                alpha_index = pixel_count // quarter
                if alpha_index >= len(alpha_values):
                    alpha_index = len(alpha_values) - 1
                
                alpha_val = alpha_values[alpha_index]
                image.putpixel((x, y), (255, 128, 64, alpha_val))
                pixel_count += 1
        
        image.save(path, 'PNG')
        return path, {255: quarter, 200: quarter, 100: quarter, 50: quarter}


class TestScenario1_DetailedTransparencyReport:
    """
    BDD Scenario: Detailed transparency report
    Given my image has been analyzed for transparency
    When transparency statistics are generated
    Then I should see percentage of transparent pixels
    And understand the distribution of alpha values
    And receive recommendations based on transparency extent
    """
    
    def test_detailed_report_includes_all_statistics(self, reporter, detector, temp_dir):
        """Test that detailed report includes comprehensive statistics."""
        # Given: my image has been analyzed for transparency
        complex_path = temp_dir / "detailed_report.png"
        StatisticsTestHelpers.create_complex_transparency_image(complex_path)
        transparency_info = detector.detect_transparency(str(complex_path))
        
        # When: transparency statistics are generated
        stats = reporter.generate_detailed_report(transparency_info, str(complex_path))
        
        # Then: I should see percentage of transparent pixels
        assert isinstance(stats.transparency_percentage, float), "Should have transparency percentage"
        assert stats.transparency_percentage > 0, "Should have some transparency"
        assert stats.total_pixels == 10000, "Should count all pixels"
        
        # And: understand the distribution of alpha values
        assert isinstance(stats.alpha_histogram, dict), "Should have alpha histogram"
        assert len(stats.alpha_histogram) > 1, "Should have multiple alpha values"
        assert isinstance(stats.alpha_distribution_summary, dict), "Should have distribution summary"
        
        required_stats = ["mean", "std", "min", "max"]
        for stat in required_stats:
            assert stat in stats.alpha_distribution_summary, f"Should include {stat} statistic"
        
        # And: receive recommendations based on transparency extent
        assert stats.processing_recommendation in ["proceed", "review", "edit_required"], "Should have valid recommendation"
        assert isinstance(stats.specific_issues, list), "Should have issues list"
    
    def test_transparency_percentage_calculation_accuracy(self, reporter, detector, temp_dir):
        """Test accuracy of transparency percentage calculations."""
        # Given: image with known transparency distribution
        test_cases = [
            ("uniform_50", lambda p: Image.new('RGBA', (100, 100), (255, 0, 0, 127)).save(p, 'PNG'), 100.0),
            ("half_transparent", self._create_half_transparent_image, 50.0),
            ("quarter_transparent", self._create_quarter_transparent_image, 25.0)
        ]
        
        for test_name, creator_func, expected_percentage in test_cases:
            # Create test image
            test_path = temp_dir / f"{test_name}.png"
            creator_func(test_path)
            
            # When: generating statistics
            transparency_info = detector.detect_transparency(str(test_path))
            stats = reporter.generate_detailed_report(transparency_info, str(test_path))
            
            # Then: percentage should be accurate
            assert abs(stats.transparency_percentage - expected_percentage) < 5.0, \
                f"Expected ~{expected_percentage}% for {test_name}, got {stats.transparency_percentage}%"
    
    def _create_half_transparent_image(self, path: Path):
        """Create image where exactly half the pixels are transparent."""
        image = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
        # Make left half transparent
        for x in range(50):
            for y in range(100):
                image.putpixel((x, y), (255, 0, 0, 128))
        image.save(path, 'PNG')
    
    def _create_quarter_transparent_image(self, path: Path):
        """Create image where exactly quarter of pixels are transparent."""
        image = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
        # Make top-left quarter transparent
        for x in range(50):
            for y in range(50):
                image.putpixel((x, y), (255, 0, 0, 128))
        image.save(path, 'PNG')
    
    def test_alpha_value_distribution_analysis(self, reporter, detector, temp_dir):
        """Test detailed alpha value distribution analysis."""
        # Given: image with known alpha distribution
        alpha_path = temp_dir / "alpha_distribution.png"
        expected_distribution = StatisticsTestHelpers.create_minimal_alpha_variation_image(alpha_path)[1]
        transparency_info = detector.detect_transparency(str(alpha_path))
        
        # When: analyzing alpha distribution
        stats = reporter.generate_detailed_report(transparency_info, str(alpha_path))
        
        # Then: should accurately represent alpha histogram
        assert len(stats.alpha_histogram) == 4, "Should have 4 different alpha values"
        
        # Verify distribution matches expected (with some tolerance for edge effects)
        total_pixels = sum(stats.alpha_histogram.values())
        for alpha_val in [255, 200, 100, 50]:
            if alpha_val in stats.alpha_histogram:
                actual_count = stats.alpha_histogram[alpha_val]
                expected_count = expected_distribution.get(alpha_val, 0)
                tolerance = expected_count * 0.1  # 10% tolerance
                assert abs(actual_count - expected_count) <= tolerance, \
                    f"Alpha {alpha_val}: expected ~{expected_count}, got {actual_count}"


class TestScenario2_TransparencyVisualizationGuidance:
    """
    BDD Scenario: Transparency visualization guidance
    Given an image with complex transparency
    When detailed analysis is complete
    Then I should receive suggestions for visualization
    And understand which areas need attention
    And get guidance on preparation techniques
    """
    
    def test_complex_transparency_guidance(self, reporter, detector, temp_dir):
        """Test guidance for complex transparency patterns."""
        # Given: an image with complex transparency
        complex_path = temp_dir / "complex_guidance.png"
        StatisticsTestHelpers.create_complex_transparency_image(complex_path)
        transparency_info = detector.detect_transparency(str(complex_path))
        
        # When: detailed analysis is complete
        stats = reporter.generate_detailed_report(transparency_info, str(complex_path))
        
        # Then: I should receive suggestions for visualization
        assert stats.complexity_score > 0.3, "Should detect complexity in complex image"
        assert stats.processing_recommendation in ["review", "edit_required"], "Should recommend review for complex transparency"
        
        # And: understand which areas need attention
        assert len(stats.transparency_regions) > 0, "Should identify transparency regions"
        assert stats.largest_transparent_region_size > 0, "Should identify largest region"
        
        # And: get guidance on preparation techniques
        assert len(stats.specific_issues) > 0, "Should identify specific issues"
        guidance_indicators = ["variation", "pattern", "complexity", "review", "transparency"]
        has_guidance = any(any(indicator in issue.lower() for indicator in guidance_indicators) 
                          for issue in stats.specific_issues)
        assert has_guidance, "Should provide guidance about transparency issues"
    
    def test_edge_transparency_detection_and_guidance(self, reporter, detector, temp_dir):
        """Test detection and guidance for edge transparency patterns."""
        # Given: image with edge-focused transparency
        edge_path = temp_dir / "edge_transparency.png"  
        StatisticsTestHelpers.create_edge_transparency_image(edge_path)
        transparency_info = detector.detect_transparency(str(edge_path))
        
        # When: analyzing edge transparency
        stats = reporter.generate_detailed_report(transparency_info, str(edge_path))
        
        # Then: should detect edge pattern
        assert stats.transparency_distribution == "edge-focused", "Should detect edge-focused pattern"
        
        # And: provide appropriate guidance
        edge_guidance_found = any("edge" in issue.lower() or "anti-aliasing" in issue.lower() 
                                 for issue in stats.specific_issues)
        assert edge_guidance_found, "Should provide edge-specific guidance"
    
    def test_scattered_transparency_detection(self, reporter, detector, temp_dir):
        """Test detection of scattered transparency patterns."""
        # Given: image with scattered transparency
        scattered_path = temp_dir / "scattered_transparency.png"
        StatisticsTestHelpers.create_scattered_transparency_image(scattered_path)  
        transparency_info = detector.detect_transparency(str(scattered_path))
        
        # When: analyzing scattered pattern
        stats = reporter.generate_detailed_report(transparency_info, str(scattered_path))
        
        # Then: should detect scattered pattern
        assert stats.transparency_distribution == "scattered", "Should detect scattered pattern"
        
        # And: identify multiple regions
        assert len(stats.transparency_regions) > 3, "Should identify multiple scattered regions"


class TestRegionalAnalysis:
    """Test regional transparency analysis capabilities."""
    
    def test_transparency_region_identification(self, reporter, detector, temp_dir):
        """Test identification of transparency regions."""
        # Given: image with distinct transparency regions
        regions_path = temp_dir / "distinct_regions.png"
        
        # Create image with 3 distinct transparent regions
        image = Image.new('RGBA', (150, 100), (255, 255, 255, 255))
        
        # Region 1: Large rectangle (left)
        for x in range(10, 40):
            for y in range(10, 40):
                image.putpixel((x, y), (255, 0, 0, 0))
        
        # Region 2: Medium rectangle (center)
        for x in range(60, 80):
            for y in range(20, 50):
                image.putpixel((x, y), (0, 255, 0, 128))
        
        # Region 3: Small rectangle (right)
        for x in range(120, 140):
            for y in range(30, 45):
                image.putpixel((x, y), (0, 0, 255, 64))
        
        image.save(regions_path, 'PNG')
        
        # When: analyzing regions
        transparency_info = detector.detect_transparency(str(regions_path))
        stats = reporter.generate_detailed_report(transparency_info, str(regions_path))
        
        # Then: should identify multiple regions
        assert len(stats.transparency_regions) >= 3, "Should identify at least 3 regions"
        assert stats.largest_transparent_region_size > 0, "Should identify largest region"
        
        # Regions should be sorted by size (largest first)
        if len(stats.transparency_regions) > 1:
            for i in range(len(stats.transparency_regions) - 1):
                assert stats.transparency_regions[i]['size'] >= stats.transparency_regions[i+1]['size'], \
                    "Regions should be sorted by size"
    
    def test_complexity_score_calculation(self, reporter, detector, temp_dir):
        """Test complexity score calculation for different patterns."""
        test_cases = [
            ("simple_uniform", StatisticsTestHelpers.create_uniform_transparency_image, 0.0, 0.3),
            ("edge_focused", StatisticsTestHelpers.create_edge_transparency_image, 0.2, 0.6),
            ("complex_mixed", StatisticsTestHelpers.create_complex_transparency_image, 0.4, 1.0)
        ]
        
        for test_name, creator_func, min_complexity, max_complexity in test_cases:
            # Given: image with specific complexity pattern
            test_path = temp_dir / f"{test_name}.png"
            creator_func(test_path)
            transparency_info = detector.detect_transparency(str(test_path))
            
            # When: calculating complexity
            stats = reporter.generate_detailed_report(transparency_info, str(test_path))
            
            # Then: complexity should be in expected range
            assert min_complexity <= stats.complexity_score <= max_complexity, \
                f"{test_name}: expected complexity {min_complexity}-{max_complexity}, got {stats.complexity_score}"


class TestReportFormatting:
    """Test report formatting and output generation."""
    
    def test_human_readable_report_formatting(self, reporter, detector, temp_dir):
        """Test human-readable report formatting."""
        # Given: transparency statistics
        stats_path = temp_dir / "formatting_test.png"
        StatisticsTestHelpers.create_complex_transparency_image(stats_path)
        transparency_info = detector.detect_transparency(str(stats_path))
        stats = reporter.generate_detailed_report(transparency_info, str(stats_path))
        
        # When: formatting report
        report = reporter.format_statistics_report(stats, verbose=False)
        verbose_report = reporter.format_statistics_report(stats, verbose=True)
        
        # Then: should include all required sections
        required_sections = [
            "📊 Transparency Analysis Report",
            "🔢 Basic Statistics:",
            "📈 Alpha Channel Analysis:",
            "🎯 Analysis Summary:"
        ]
        
        for section in required_sections:
            assert section in report, f"Should include {section} section"
        
        # Verbose report should have additional details
        assert len(verbose_report) > len(report), "Verbose report should be longer"
        assert "🔬 Detailed Technical Information:" in verbose_report, "Should include detailed section"
    
    def test_json_export_functionality(self, reporter, detector, temp_dir):
        """Test JSON export of statistics."""
        # Given: transparency statistics
        json_path = temp_dir / "json_export.png"
        self.create_moderate_transparency_image(json_path)
        transparency_info = detector.detect_transparency(str(json_path))
        stats = reporter.generate_detailed_report(transparency_info, str(json_path))
        
        # When: exporting to JSON
        json_data = reporter.export_statistics_json(stats)
        
        # Then: should be valid JSON structure
        assert isinstance(json_data, dict), "Should return dictionary"
        
        required_sections = ["basic_statistics", "alpha_analysis", "regional_analysis", "assessment"]
        for section in required_sections:
            assert section in json_data, f"Should include {section} section"
        
        # Should be JSON serializable
        json_str = json.dumps(json_data)
        assert isinstance(json_str, str), "Should be JSON serializable"
        
        # Verify data integrity
        assert json_data["basic_statistics"]["total_pixels"] == stats.total_pixels, "Should preserve total pixels"
        assert json_data["assessment"]["processing_recommendation"] == stats.processing_recommendation, "Should preserve recommendation"
    
    def create_moderate_transparency_image(self, path: Path):
        """Helper to create moderate transparency image."""
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        # Make 25% of pixels semi-transparent
        for x in range(50, 100):
            for y in range(50, 100):
                image.putpixel((x, y), (0, 255, 0, 128))
        image.save(path, 'PNG')


class TestRecommendationSystem:
    """Test recommendation generation system."""
    
    def test_processing_recommendations_accuracy(self, reporter, detector, temp_dir):
        """Test accuracy of processing recommendations."""
        test_cases = [
            # (image_creator, expected_recommendation_level)
            (StatisticsTestHelpers.create_uniform_transparency_image, "proceed_or_review"),
            (StatisticsTestHelpers.create_edge_transparency_image, "proceed_or_review"),
            (StatisticsTestHelpers.create_complex_transparency_image, "review_or_edit"),
            (StatisticsTestHelpers.create_scattered_transparency_image, "review_or_edit")
        ]
        
        for creator_func, expected_level in test_cases:
            # Given: image with specific transparency pattern
            test_path = temp_dir / f"recommendation_{creator_func.__name__}.png"
            creator_func(test_path)
            transparency_info = detector.detect_transparency(str(test_path))
            
            # When: generating recommendations
            stats = reporter.generate_detailed_report(transparency_info, str(test_path))
            
            # Then: should provide appropriate recommendation
            if expected_level == "proceed_or_review":
                assert stats.processing_recommendation in ["proceed", "review"], \
                    f"Expected proceed/review for {creator_func.__name__}, got {stats.processing_recommendation}"
            elif expected_level == "review_or_edit":
                assert stats.processing_recommendation in ["review", "edit_required"], \
                    f"Expected review/edit for {creator_func.__name__}, got {stats.processing_recommendation}"
    
    def test_specific_issue_identification(self, reporter, detector, temp_dir):
        """Test identification of specific transparency issues."""
        # Given: image with high transparency
        high_trans_path = temp_dir / "high_transparency_issues.png"
        
        # Create image with 90% transparency
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 0))  # Fully transparent
        # Add small opaque area (10% of pixels)
        for x in range(10):
            for y in range(100):
                image.putpixel((x, y), (255, 0, 0, 255))
        image.save(high_trans_path, 'PNG')
        
        # When: analyzing issues
        transparency_info = detector.detect_transparency(str(high_trans_path))
        stats = reporter.generate_detailed_report(transparency_info, str(high_trans_path))
        
        # Then: should identify high transparency issue
        high_transparency_issue = any("high transparency" in issue.lower() for issue in stats.specific_issues)
        assert high_transparency_issue, "Should identify high transparency as issue"
        
        # Should recommend action
        assert stats.processing_recommendation in ["review", "edit_required"], "Should recommend review or edit"


class TestNoTransparencyHandling:
    """Test handling of images without transparency."""
    
    def test_no_transparency_statistics_report(self, reporter, detector, temp_dir):
        """Test statistics report for images without transparency."""
        # Given: opaque image
        opaque_path = temp_dir / "opaque_stats.png"
        opaque_image = Image.new('RGB', (100, 100), (255, 0, 0))
        opaque_image.save(opaque_path, 'PNG')
        
        # When: generating report
        transparency_info = detector.detect_transparency(str(opaque_path))
        stats = reporter.generate_detailed_report(transparency_info, str(opaque_path))
        
        # Then: should handle no transparency gracefully
        assert stats.transparency_percentage == 0.0, "Should report 0% transparency"
        assert stats.transparent_pixel_count == 0, "Should have no transparent pixels"
        assert stats.complexity_score == 0.0, "Should have 0 complexity"
        assert stats.processing_recommendation == "proceed", "Should recommend proceeding"
        assert len(stats.specific_issues) == 0, "Should have no issues"
        assert stats.transparency_distribution == "none", "Should indicate no transparency"
    
    def test_no_transparency_report_formatting(self, reporter, detector, temp_dir):
        """Test report formatting for non-transparent images."""
        # Given: opaque image statistics
        opaque_path = temp_dir / "opaque_format.png"
        Image.new('RGB', (50, 50), (0, 255, 0)).save(opaque_path, 'PNG')
        transparency_info = detector.detect_transparency(str(opaque_path))
        stats = reporter.generate_detailed_report(transparency_info, str(opaque_path))
        
        # When: formatting report
        report = reporter.format_statistics_report(stats)
        
        # Then: should format appropriately for no transparency
        assert "0.0%" in report, "Should show 0% transparency"
        assert "Proceed" in report, "Should show proceed recommendation"
        assert "📊 Transparency Analysis Report" in report, "Should have report header"
        
        # Should not include transparency-specific sections when not relevant
        transparency_sections = ["🗺️ Transparency Regions:", "⚠️ Issues Identified:"]
        for section in transparency_sections:
            if section in report:
                # If section is present, it should indicate no issues/regions
                section_start = report.find(section)
                section_content = report[section_start:section_start + 200]
                assert "0" in section_content or "none" in section_content.lower(), \
                    f"Section {section} should indicate no transparency features"