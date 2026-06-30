"""Gradient processing for advanced shading and transparency effects.

This module implements gradient detection and processing for creating realistic
shading and gradient effects using transparency mixing.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GradientConfig:
    """Configuration for gradient processing."""

    gradient_detection_threshold: float = 0.1
    smoothness_preservation_weight: float = 0.8
    detail_preservation_weight: float = 0.2
    max_gradient_layers: int = 3
    enable_enhancement: bool = True


class GradientProcessor:
    """Processor for detecting and optimizing gradients using transparency.

    This class implements Story 4.5.5: Advanced Shading and Gradient Effects,
    focusing on creating realistic gradients using transparency mixing.
    """

    def __init__(
        self,
        gradient_detection_threshold: float = 0.1,
        smoothness_preservation_weight: float = 0.8,
        detail_preservation_weight: float = 0.2,
        device: str = "cuda",
    ):
        """Initialize gradient processor.

        Args:
            gradient_detection_threshold: Threshold for gradient detection
            smoothness_preservation_weight: Weight for preserving smoothness
            detail_preservation_weight: Weight for preserving details
            device: Device for computations
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gradient_threshold = gradient_detection_threshold
        self.smoothness_weight = smoothness_preservation_weight
        self.detail_weight = detail_preservation_weight

        # Validate weights
        total_weight = smoothness_preservation_weight + detail_preservation_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def detect_gradient_regions(
        self,
        image: torch.Tensor,
        min_gradient_length: int = 10,
        gradient_smoothness_threshold: float = 0.8,
    ) -> Dict:
        """Detect gradient regions suitable for transparency mixing.

        Args:
            image: Input image tensor (1, 3, H, W)
            min_gradient_length: Minimum length for gradient detection
            gradient_smoothness_threshold: Threshold for gradient smoothness

        Returns:
            Dictionary with detected gradient regions and analysis
        """
        # Convert to grayscale for initial gradient detection
        gray = torch.mean(image, dim=1, keepdim=True)

        # Calculate gradients using Sobel operators
        gradient_x, gradient_y = self._calculate_image_gradients(gray)
        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)

        # Detect gradient directions
        gradient_direction = torch.atan2(gradient_y, gradient_x)

        # Find regions with consistent gradients
        gradient_regions = self._find_consistent_gradient_regions(
            gradient_magnitude, gradient_direction, min_gradient_length
        )

        # Analyze each region for transparency suitability
        analyzed_regions = []
        for region in gradient_regions:
            analysis = self._analyze_gradient_region(
                image, region, gradient_smoothness_threshold
            )
            analyzed_regions.append(analysis)

        # Classify gradient types
        region_types = self._classify_gradient_types(analyzed_regions)

        # Calculate suitability scores
        suitability_scores = self._calculate_transparency_suitability(analyzed_regions)

        return {
            "gradient_regions": analyzed_regions,
            "region_types": region_types,
            "suitability_scores": suitability_scores,
            "gradient_magnitude_map": gradient_magnitude,
            "gradient_direction_map": gradient_direction,
        }

    def create_layer_combinations_for_gradients(
        self,
        image: torch.Tensor,
        max_layers_per_region: int = 3,
        smoothness_target: float = 0.9,
    ) -> Dict:
        """Create layer combinations for smooth gradient transitions.

        Args:
            image: Input image tensor (1, 3, H, W)
            max_layers_per_region: Maximum layers per gradient region
            smoothness_target: Target smoothness score

        Returns:
            Dictionary with layer assignments and quality metrics
        """
        # Detect gradient regions
        gradient_analysis = self.detect_gradient_regions(image)
        gradient_regions = gradient_analysis["gradient_regions"]

        # Create layer assignments
        height, width = image.shape[-2:]
        layer_assignments = torch.zeros(
            max_layers_per_region, height, width, dtype=torch.long
        )

        # Process each gradient region
        region_assignments = {}
        for i, region in enumerate(gradient_regions):
            if (
                region["transparency_suitability"] > 0.5
            ):  # Only process suitable regions
                region_layers = self._create_gradient_layers(
                    image, region, max_layers_per_region, smoothness_target
                )
                region_assignments[f"region_{i}"] = region_layers

                # Apply to layer assignments
                bounds = region["region_bounds"]
                self._apply_region_assignments(layer_assignments, region_layers, bounds)

        # Calculate transition quality
        transition_quality = self._calculate_transition_quality(
            layer_assignments, gradient_regions
        )

        # Calculate smoothness metrics
        smoothness_metrics = self._calculate_smoothness_metrics(
            layer_assignments, image, smoothness_target
        )

        return {
            "layer_assignments": layer_assignments,
            "region_assignments": region_assignments,
            "transition_quality": transition_quality,
            "smoothness_metrics": smoothness_metrics,
        }

    def analyze_layer_usage(self, layer_assignments: torch.Tensor) -> Dict:
        """Analyze layer usage patterns."""
        num_layers, height, width = layer_assignments.shape

        # Count single vs multi-layer regions
        single_layer_count = 0
        multi_layer_count = 0

        for i in range(height):
            for j in range(width):
                pixel_layers = layer_assignments[:, i, j]
                unique_layers = torch.unique(pixel_layers)

                if len(unique_layers) == 1:
                    single_layer_count += 1
                else:
                    multi_layer_count += 1

        total_pixels = height * width

        return {
            "single_layer_regions": single_layer_count,
            "multi_layer_regions": multi_layer_count,
            "single_layer_ratio": single_layer_count / total_pixels,
            "multi_layer_ratio": multi_layer_count / total_pixels,
            "average_layers_per_pixel": torch.mean(
                torch.sum(layer_assignments > 0, dim=0).float()
            ).item(),
        }

    def analyze_layer_boundary_smoothness(
        self,
        image: torch.Tensor,
        layer_height: float = 0.08,
        smoothness_kernel_size: int = 5,
    ) -> Dict:
        """Analyze smoothness across layer boundaries.

        Args:
            image: Input image tensor (1, 3, H, W)
            layer_height: Physical layer height in mm
            smoothness_kernel_size: Kernel size for smoothness analysis

        Returns:
            Dictionary with boundary smoothness analysis
        """
        # Detect gradient regions for boundary analysis
        gradient_analysis = self.detect_gradient_regions(image)
        gradient_regions = gradient_analysis["gradient_regions"]

        boundary_scores = []
        problematic_boundaries = []
        smoothness_corrections = []

        for i, region in enumerate(gradient_regions):
            # Analyze boundary smoothness for this region
            boundary_analysis = self._analyze_region_boundary_smoothness(
                image, region, smoothness_kernel_size
            )

            boundary_score = boundary_analysis["smoothness_score"]
            boundary_scores.append(boundary_score)

            # Identify problematic boundaries
            if boundary_score < 0.7:
                problematic_boundaries.append(
                    {
                        "region_id": i,
                        "boundary_score": boundary_score,
                        "issues": boundary_analysis["issues"],
                    }
                )

                # Generate correction suggestions
                correction = self._generate_smoothness_correction(
                    region, boundary_analysis, layer_height
                )
                smoothness_corrections.append(correction)

        return {
            "boundary_smoothness_scores": boundary_scores,
            "average_boundary_smoothness": (
                np.mean(boundary_scores) if boundary_scores else 0.0
            ),
            "problematic_boundaries": problematic_boundaries,
            "smoothness_corrections": smoothness_corrections,
        }

    def preserve_gradient_details(
        self,
        image: torch.Tensor,
        detail_preservation_threshold: float = 0.02,
        detail_enhancement_factor: float = 1.2,
    ) -> Dict:
        """Preserve fine details in gradient regions.

        Args:
            image: Input image tensor (1, 3, H, W)
            detail_preservation_threshold: Threshold for detail detection
            detail_enhancement_factor: Factor for detail enhancement

        Returns:
            Dictionary with detail preservation results
        """
        # Detect fine details in the image
        detail_map = self._detect_fine_details(image, detail_preservation_threshold)

        # Identify detail regions within gradients
        gradient_analysis = self.detect_gradient_regions(image)
        gradient_regions = gradient_analysis["gradient_regions"]

        detail_regions = self._identify_detail_regions(
            detail_map, gradient_regions, detail_preservation_threshold
        )

        # Calculate preservation accuracy
        preservation_accuracy = self._calculate_detail_preservation_accuracy(
            image, detail_regions, detail_enhancement_factor
        )

        # Create enhancement map
        enhancement_map = self._create_detail_enhancement_map(
            image, detail_regions, detail_enhancement_factor
        )

        # Calculate detail quality score
        detail_quality_score = self._calculate_detail_quality_score(
            detail_regions, enhancement_map, preservation_accuracy
        )

        return {
            "preserved_details": {
                "detail_regions": detail_regions,
                "preservation_accuracy": preservation_accuracy,
            },
            "detail_quality_score": detail_quality_score,
            "enhancement_map": enhancement_map,
            "detail_statistics": {
                "total_detail_regions": len(detail_regions),
                "average_detail_strength": (
                    np.mean([dr["detail_strength"] for dr in detail_regions])
                    if detail_regions
                    else 0.0
                ),
            },
        }

    def process_complex_gradients(
        self,
        image: torch.Tensor,
        max_layers: int = 3,
        quality_target: float = 0.9,
        processing_mode: str = "detailed",
    ) -> Dict:
        """Process complex multi-color gradients.

        Args:
            image: Complex gradient image tensor (1, 3, H, W)
            max_layers: Maximum layers for processing
            quality_target: Target quality score
            processing_mode: Processing mode ('fast', 'standard', 'detailed')

        Returns:
            Dictionary with complex gradient processing results
        """
        # Analyze gradient complexity
        complexity_analysis = self._analyze_gradient_complexity(image)

        # Adapt processing strategy based on complexity
        if complexity_analysis["complexity_level"] == "high":
            strategy = self._get_high_complexity_strategy(processing_mode)
        elif complexity_analysis["complexity_level"] == "medium":
            strategy = self._get_medium_complexity_strategy(processing_mode)
        else:
            strategy = self._get_low_complexity_strategy(processing_mode)

        # Apply processing strategy
        processing_result = self._apply_gradient_processing_strategy(
            image, strategy, max_layers, quality_target
        )

        # Calculate quality metrics
        quality_metrics = self._calculate_complex_gradient_quality(
            image, processing_result, quality_target
        )

        # Create layer strategy
        layer_strategy = self._create_complex_gradient_layer_strategy(
            processing_result, max_layers
        )

        return {
            "processing_success": processing_result["success"],
            "gradient_analysis": complexity_analysis,
            "layer_strategy": layer_strategy,
            "quality_metrics": quality_metrics,
            "processing_details": processing_result,
        }

    def process_gradients_optimized(
        self,
        image: torch.Tensor,
        max_layers: int = 3,
        use_gpu_acceleration: bool = True,
        chunk_processing: bool = True,
        chunk_size: int = 128,
    ) -> Dict:
        """Process gradients with performance optimization.

        Args:
            image: Input image tensor (1, 3, H, W)
            max_layers: Maximum layers for processing
            use_gpu_acceleration: Whether to use GPU acceleration
            chunk_processing: Whether to use chunk processing
            chunk_size: Size of chunks for processing

        Returns:
            Dictionary with optimized processing results
        """
        start_time = (
            torch.cuda.Event(enable_timing=True) if use_gpu_acceleration else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if use_gpu_acceleration else None
        )

        if use_gpu_acceleration and torch.cuda.is_available():
            if start_time:
                start_time.record()

        # Move to appropriate device
        device = self.device if use_gpu_acceleration else torch.device("cpu")
        image = image.to(device)

        if chunk_processing and image.shape[-1] > chunk_size:
            # Process in chunks
            result = self._process_gradients_in_chunks(
                image, max_layers, chunk_size, device
            )
        else:
            # Process normally
            result = self._process_gradients_standard(image, max_layers, device)

        if use_gpu_acceleration and torch.cuda.is_available() and end_time:
            end_time.record()
            torch.cuda.synchronize()
            processing_time = (
                start_time.elapsed_time(end_time) / 1000.0
            )  # Convert to seconds
        else:
            processing_time = 0.0

        # Calculate quality score
        quality_score = self._calculate_processing_quality_score(result)

        # Estimate memory usage
        memory_usage = self._estimate_memory_usage(image, max_layers)

        return {
            "processing_result": result,
            "processing_time_seconds": processing_time,
            "quality_score": quality_score,
            "memory_usage": memory_usage,
            "optimization_used": {
                "gpu_acceleration": use_gpu_acceleration and torch.cuda.is_available(),
                "chunk_processing": chunk_processing,
                "chunk_size": chunk_size if chunk_processing else None,
            },
        }

    # Private helper methods

    def _calculate_image_gradients(
        self, gray_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate image gradients using Sobel operators."""
        sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        gradient_x = F.conv2d(gray_image, sobel_x, padding=1)
        gradient_y = F.conv2d(gray_image, sobel_y, padding=1)

        return gradient_x.squeeze(0), gradient_y.squeeze(0)

    def _find_consistent_gradient_regions(
        self,
        gradient_magnitude: torch.Tensor,
        gradient_direction: torch.Tensor,
        min_length: int,
    ) -> List[Dict]:
        """Find regions with consistent gradient patterns."""
        # Threshold gradient magnitude
        strong_gradients = gradient_magnitude > self.gradient_threshold

        # Find connected components of strong gradients
        regions = []
        visited = torch.zeros_like(strong_gradients, dtype=torch.bool)

        height, width = strong_gradients.shape[-2:]

        for i in range(height):
            for j in range(width):
                if strong_gradients[0, i, j] and not visited[i, j]:
                    # Start flood fill to find connected region
                    region_pixels = self._flood_fill_gradient_region(
                        strong_gradients[0], gradient_direction[0], visited, i, j
                    )

                    if len(region_pixels) >= min_length:
                        region_bounds = self._calculate_region_bounds(region_pixels)
                        regions.append(
                            {
                                "pixels": region_pixels,
                                "region_bounds": region_bounds,
                                "size": len(region_pixels),
                            }
                        )

        return regions

    def _analyze_gradient_region(
        self, image: torch.Tensor, region: Dict, smoothness_threshold: float
    ) -> Dict:
        """Analyze a gradient region for transparency suitability."""
        pixels = region["pixels"]
        bounds = region["region_bounds"]

        # Extract region colors
        region_colors = []
        for i, j in pixels:
            color = image[0, :, i, j]
            region_colors.append(color)

        if not region_colors:
            return {
                "region_id": region.get("region_id", 0),
                "gradient_type": "unknown",
                "start_color": torch.zeros(3),
                "end_color": torch.zeros(3),
                "transparency_suitability": 0.0,
                "region_bounds": bounds,
            }

        region_color_tensor = torch.stack(region_colors)

        # Identify start and end colors
        start_color, end_color = self._identify_gradient_endpoints(
            region_color_tensor, pixels
        )

        # Calculate gradient smoothness
        smoothness_score = self._calculate_gradient_smoothness(region_color_tensor)

        # Determine gradient type
        gradient_type = self._determine_gradient_type(start_color, end_color, pixels)

        # Calculate transparency suitability
        transparency_suitability = self._calculate_region_transparency_suitability(
            start_color, end_color, smoothness_score, smoothness_threshold
        )

        return {
            "region_id": region.get("region_id", 0),
            "gradient_type": gradient_type,
            "start_color": start_color,
            "end_color": end_color,
            "transparency_suitability": transparency_suitability,
            "region_bounds": bounds,
            "smoothness_score": smoothness_score,
        }

    def _classify_gradient_types(self, regions: List[Dict]) -> Dict:
        """Classify gradient regions by type."""
        type_counts = {"linear": 0, "radial": 0, "diagonal": 0, "complex": 0}

        for region in regions:
            gradient_type = region["gradient_type"]
            if gradient_type in type_counts:
                type_counts[gradient_type] += 1
            else:
                type_counts["complex"] += 1

        return type_counts

    def _calculate_transparency_suitability(self, regions: List[Dict]) -> List[float]:
        """Calculate transparency suitability scores for regions."""
        return [region["transparency_suitability"] for region in regions]

    def _create_gradient_layers(
        self,
        image: torch.Tensor,
        region: Dict,
        max_layers: int,
        smoothness_target: float,
    ) -> Dict:
        """Create layer assignments for gradient region."""
        start_color = region["start_color"]
        end_color = region["end_color"]
        pixels = region["pixels"]

        # Create smooth progression between start and end colors
        layer_progression = []
        for layer in range(max_layers):
            alpha = layer / (max_layers - 1) if max_layers > 1 else 0
            layer_color = start_color * (1 - alpha) + end_color * alpha
            layer_progression.append(layer_color)

        # Assign pixels to layers based on their position in gradient
        pixel_assignments = {}
        for pixel_idx, (i, j) in enumerate(pixels):
            pixel_color = image[0, :, i, j]

            # Find closest layer color
            distances = [
                torch.norm(pixel_color - layer_color)
                for layer_color in layer_progression
            ]
            closest_layer = np.argmin(distances)

            pixel_assignments[(i, j)] = closest_layer

        return {
            "pixel_assignments": pixel_assignments,
            "layer_colors": layer_progression,
            "num_layers_used": max_layers,
        }

    def _apply_region_assignments(
        self, layer_assignments: torch.Tensor, region_layers: Dict, bounds: Tuple
    ):
        """Apply region layer assignments to global layer tensor."""
        pixel_assignments = region_layers["pixel_assignments"]

        for (i, j), layer_idx in pixel_assignments.items():
            if (
                0 <= i < layer_assignments.shape[1]
                and 0 <= j < layer_assignments.shape[2]
            ):
                layer_assignments[layer_idx, i, j] = 1  # Mark as assigned

    def _calculate_transition_quality(
        self, layer_assignments: torch.Tensor, gradient_regions: List[Dict]
    ) -> Dict:
        """Calculate quality of layer transitions."""
        # Simplified transition quality calculation
        total_transitions = 0
        smooth_transitions = 0

        num_layers = layer_assignments.shape[0]

        for layer in range(num_layers - 1):
            current_layer = layer_assignments[layer]
            next_layer = layer_assignments[layer + 1]

            # Count transitions
            transitions = torch.sum(current_layer != next_layer).item()
            total_transitions += transitions

            # Estimate smooth transitions (simplified)
            smooth_transitions += int(transitions * 0.8)  # Assume 80% are smooth

        quality_score = smooth_transitions / max(total_transitions, 1)

        return {
            "overall_quality": quality_score,
            "total_transitions": total_transitions,
            "smooth_transitions": smooth_transitions,
        }

    def _calculate_smoothness_metrics(
        self, layer_assignments: torch.Tensor, image: torch.Tensor, target: float
    ) -> Dict:
        """Calculate detailed smoothness metrics."""
        # Calculate gradient smoothness
        gradient_smoothness = self._measure_gradient_smoothness(layer_assignments)

        # Calculate layer transition quality
        layer_transition_quality = self._measure_layer_transition_quality(
            layer_assignments
        )

        # Calculate color continuity
        color_continuity = self._measure_color_continuity(layer_assignments, image)

        return {
            "gradient_smoothness": gradient_smoothness,
            "layer_transition_quality": layer_transition_quality,
            "color_continuity": color_continuity,
            "meets_target": gradient_smoothness >= target,
        }

    def _flood_fill_gradient_region(
        self,
        strong_gradients: torch.Tensor,
        gradient_direction: torch.Tensor,
        visited: torch.Tensor,
        start_i: int,
        start_j: int,
    ) -> List[Tuple[int, int]]:
        """Flood fill to find connected gradient region."""
        stack = [(start_i, start_j)]
        region_pixels = []
        height, width = strong_gradients.shape

        while stack:
            i, j = stack.pop()

            if (
                i < 0
                or i >= height
                or j < 0
                or j >= width
                or visited[i, j]
                or not strong_gradients[i, j]
            ):
                continue

            visited[i, j] = True
            region_pixels.append((i, j))

            # Add neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((i + di, j + dj))

        return region_pixels

    def _calculate_region_bounds(
        self, pixels: List[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        """Calculate bounding box for region pixels."""
        if not pixels:
            return (0, 0, 0, 0)

        min_i = min(p[0] for p in pixels)
        max_i = max(p[0] for p in pixels)
        min_j = min(p[1] for p in pixels)
        max_j = max(p[1] for p in pixels)

        return (min_i, min_j, max_i, max_j)

    def _identify_gradient_endpoints(
        self, region_colors: torch.Tensor, pixels: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify start and end colors of gradient."""
        if len(region_colors) == 0:
            return torch.zeros(3), torch.zeros(3)
        elif len(region_colors) == 1:
            return region_colors[0], region_colors[0]

        # Simple approach: use colors with maximum distance
        max_distance = 0
        start_color = region_colors[0]
        end_color = region_colors[0]

        for i in range(len(region_colors)):
            for j in range(i + 1, len(region_colors)):
                distance = torch.norm(region_colors[i] - region_colors[j])
                if distance > max_distance:
                    max_distance = distance
                    start_color = region_colors[i]
                    end_color = region_colors[j]

        return start_color, end_color

    def _calculate_gradient_smoothness(self, region_colors: torch.Tensor) -> float:
        """Calculate smoothness score for gradient colors."""
        if len(region_colors) <= 1:
            return 1.0

        # Calculate variance in color changes
        color_diffs = []
        for i in range(len(region_colors) - 1):
            diff = torch.norm(region_colors[i + 1] - region_colors[i])
            color_diffs.append(diff.item())

        # Smoothness is inverse of variance in differences
        if not color_diffs:
            return 1.0

        variance = np.var(color_diffs)
        smoothness = 1.0 / (1.0 + variance)

        return smoothness

    def _determine_gradient_type(
        self,
        start_color: torch.Tensor,
        end_color: torch.Tensor,
        pixels: List[Tuple[int, int]],
    ) -> str:
        """Determine type of gradient based on geometry and colors."""
        if len(pixels) < 4:
            return "simple"

        # Analyze pixel arrangement
        pixel_array = np.array(pixels)

        # Check if pixels form a line (linear gradient)
        if self._is_linear_arrangement(pixel_array):
            return "linear"

        # Check if pixels form a circle (radial gradient)
        if self._is_radial_arrangement(pixel_array):
            return "radial"

        # Check if pixels form a diagonal (diagonal gradient)
        if self._is_diagonal_arrangement(pixel_array):
            return "diagonal"

        return "complex"

    def _calculate_region_transparency_suitability(
        self,
        start_color: torch.Tensor,
        end_color: torch.Tensor,
        smoothness_score: float,
        smoothness_threshold: float,
    ) -> float:
        """Calculate how suitable a region is for transparency mixing."""
        # Color contrast factor
        color_distance = torch.norm(end_color - start_color).item()
        contrast_factor = min(color_distance / 2.0, 1.0)  # Normalize to 0-1

        # Smoothness factor
        smoothness_factor = smoothness_score

        # Brightness factor (darker colors work better with transparency)
        avg_brightness = torch.mean((start_color + end_color) / 2).item()
        brightness_factor = 1.0 - avg_brightness  # Darker = better

        # Combined suitability score
        suitability = (
            0.4 * contrast_factor + 0.4 * smoothness_factor + 0.2 * brightness_factor
        )

        # Apply smoothness threshold
        if smoothness_score < smoothness_threshold:
            suitability *= 0.5  # Reduce suitability for non-smooth gradients

        return min(suitability, 1.0)

    def _is_linear_arrangement(self, pixels: np.ndarray) -> bool:
        """Check if pixels form a linear arrangement."""
        if len(pixels) < 3:
            return True

        # Simple linear regression to check linearity
        x = pixels[:, 0]
        y = pixels[:, 1]

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

        return abs(correlation) > 0.8  # High correlation indicates linear arrangement

    def _is_radial_arrangement(self, pixels: np.ndarray) -> bool:
        """Check if pixels form a radial arrangement."""
        if len(pixels) < 5:
            return False

        # Calculate center of pixels
        center = np.mean(pixels, axis=0)

        # Calculate distances from center
        distances = np.linalg.norm(pixels - center, axis=1)

        # Check if distances form a pattern
        distance_variance = np.var(distances)

        return distance_variance < 100  # Low variance indicates radial pattern

    def _is_diagonal_arrangement(self, pixels: np.ndarray) -> bool:
        """Check if pixels form a diagonal arrangement."""
        if len(pixels) < 3:
            return False

        # Check if the arrangement is roughly diagonal
        x = pixels[:, 0]
        y = pixels[:, 1]

        # Calculate slope
        if np.var(x) > 0:
            slope = np.cov(x, y)[0, 1] / np.var(x)
            return abs(abs(slope) - 1.0) < 0.3  # Close to 45-degree angle

        return False

    # Additional helper methods for remaining functionality
    def _analyze_region_boundary_smoothness(
        self, image: torch.Tensor, region: Dict, kernel_size: int
    ) -> Dict:
        """Analyze smoothness of region boundaries."""
        # Simplified boundary analysis
        smoothness_score = region.get("smoothness_score", 0.5)

        issues = []
        if smoothness_score < 0.7:
            issues.append("low_smoothness")
        if region.get("size", 0) < 10:
            issues.append("too_small")

        return {"smoothness_score": smoothness_score, "issues": issues}

    def _generate_smoothness_correction(
        self, region: Dict, boundary_analysis: Dict, layer_height: float
    ) -> Dict:
        """Generate correction suggestions for smoothness issues."""
        correction_type = "interpolation"
        if "low_smoothness" in boundary_analysis["issues"]:
            correction_type = "gaussian_blur"
        elif "too_small" in boundary_analysis["issues"]:
            correction_type = "region_merge"

        return {
            "boundary_id": region.get("region_id", 0),
            "correction_type": correction_type,
            "expected_improvement": 0.2,
        }

    def _detect_fine_details(
        self, image: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """Detect fine details in image."""
        # Use Laplacian to detect details
        laplacian_kernel = (
            torch.tensor(
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        gray = torch.mean(image, dim=1, keepdim=True)
        detail_response = F.conv2d(gray, laplacian_kernel, padding=1)
        detail_map = torch.abs(detail_response) > threshold

        return detail_map

    def _identify_detail_regions(
        self, detail_map: torch.Tensor, gradient_regions: List[Dict], threshold: float
    ) -> List[Dict]:
        """Identify detail regions within gradients."""
        detail_regions = []

        for i, region in enumerate(gradient_regions):
            bounds = region["region_bounds"]
            min_i, min_j, max_i, max_j = bounds

            # Extract detail map for this region
            region_details = detail_map[0, 0, min_i:max_i, min_j:max_j]
            detail_count = torch.sum(region_details).item()

            if detail_count > 5:  # Minimum detail count
                detail_regions.append(
                    {
                        "region_id": i,
                        "detail_count": detail_count,
                        "detail_strength": torch.mean(region_details.float()).item(),
                        "bounds": bounds,
                    }
                )

        return detail_regions

    def _calculate_detail_preservation_accuracy(
        self, image: torch.Tensor, detail_regions: List[Dict], enhancement_factor: float
    ) -> float:
        """Calculate accuracy of detail preservation."""
        if not detail_regions:
            return 1.0

        # Simplified accuracy calculation
        total_accuracy = 0.0

        for region in detail_regions:
            region_accuracy = min(region["detail_strength"] * enhancement_factor, 1.0)
            total_accuracy += region_accuracy

        return total_accuracy / len(detail_regions)

    def _create_detail_enhancement_map(
        self, image: torch.Tensor, detail_regions: List[Dict], enhancement_factor: float
    ) -> torch.Tensor:
        """Create enhancement map for detail preservation."""
        enhancement_map = torch.zeros_like(image)

        for region in detail_regions:
            bounds = region["bounds"]
            min_i, min_j, max_i, max_j = bounds

            # Apply enhancement to region
            region_enhancement = (
                torch.ones(1, 3, max_i - min_i, max_j - min_j) * enhancement_factor
            )
            enhancement_map[:, :, min_i:max_i, min_j:max_j] = region_enhancement

        return enhancement_map

    def _calculate_detail_quality_score(
        self,
        detail_regions: List[Dict],
        enhancement_map: torch.Tensor,
        preservation_accuracy: float,
    ) -> float:
        """Calculate overall detail quality score."""
        if not detail_regions:
            return 0.5

        # Combine factors
        region_count_factor = min(len(detail_regions) / 10.0, 1.0)
        enhancement_factor = torch.mean(enhancement_map).item()
        accuracy_factor = preservation_accuracy

        quality_score = (
            0.3 * region_count_factor + 0.3 * enhancement_factor + 0.4 * accuracy_factor
        )

        return min(quality_score, 1.0)

    def _analyze_gradient_complexity(self, image: torch.Tensor) -> Dict:
        """Analyze complexity of gradients in image."""
        # Calculate gradient statistics
        gray = torch.mean(image, dim=1, keepdim=True)
        grad_x, grad_y = self._calculate_image_gradients(gray)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        # Complexity metrics
        gradient_variance = torch.var(gradient_magnitude).item()
        gradient_mean = torch.mean(gradient_magnitude).item()
        color_variance = torch.var(image).item()

        # Determine complexity level
        if gradient_variance > 0.1 and color_variance > 0.2:
            complexity_level = "high"
        elif gradient_variance > 0.05 or color_variance > 0.1:
            complexity_level = "medium"
        else:
            complexity_level = "low"

        return {
            "complexity_level": complexity_level,
            "gradient_variance": gradient_variance,
            "gradient_mean": gradient_mean,
            "color_variance": color_variance,
            "complexity_score": min((gradient_variance + color_variance) / 0.3, 1.0),
        }

    def _get_high_complexity_strategy(self, mode: str) -> Dict:
        """Get strategy for high complexity gradients."""
        if mode == "detailed":
            return {
                "subdivision_levels": 3,
                "smoothing_iterations": 5,
                "detail_preservation": True,
                "adaptive_layers": True,
            }
        else:
            return {
                "subdivision_levels": 2,
                "smoothing_iterations": 3,
                "detail_preservation": False,
                "adaptive_layers": True,
            }

    def _get_medium_complexity_strategy(self, mode: str) -> Dict:
        """Get strategy for medium complexity gradients."""
        return {
            "subdivision_levels": 2,
            "smoothing_iterations": 3,
            "detail_preservation": mode == "detailed",
            "adaptive_layers": True,
        }

    def _get_low_complexity_strategy(self, mode: str) -> Dict:
        """Get strategy for low complexity gradients."""
        return {
            "subdivision_levels": 1,
            "smoothing_iterations": 2,
            "detail_preservation": False,
            "adaptive_layers": False,
        }

    def _apply_gradient_processing_strategy(
        self,
        image: torch.Tensor,
        strategy: Dict,
        max_layers: int,
        quality_target: float,
    ) -> Dict:
        """Apply gradient processing strategy."""
        # Simplified strategy application
        success = True
        processed_regions = []

        try:
            # Detect gradients
            gradient_analysis = self.detect_gradient_regions(image)
            gradient_regions = gradient_analysis["gradient_regions"]

            # Process each region according to strategy
            for region in gradient_regions:
                if strategy["adaptive_layers"]:
                    region_layers = min(max_layers, 3)
                else:
                    region_layers = 2

                processed_region = {
                    "region_id": region.get("region_id", 0),
                    "layers_used": region_layers,
                    "quality_achieved": region.get("transparency_suitability", 0.5),
                }
                processed_regions.append(processed_region)

        except Exception:
            success = False

        return {
            "success": success,
            "processed_regions": processed_regions,
            "strategy_applied": strategy,
        }

    def _calculate_complex_gradient_quality(
        self, image: torch.Tensor, processing_result: Dict, quality_target: float
    ) -> Dict:
        """Calculate quality metrics for complex gradient processing."""
        if not processing_result["success"]:
            return {
                "overall_quality": 0.0,
                "gradient_preservation": 0.0,
                "color_accuracy": 0.0,
            }

        # Simplified quality calculation
        processed_regions = processing_result["processed_regions"]

        if not processed_regions:
            return {
                "overall_quality": 0.5,
                "gradient_preservation": 0.5,
                "color_accuracy": 0.5,
            }

        avg_quality = np.mean([r["quality_achieved"] for r in processed_regions])

        return {
            "overall_quality": min(avg_quality, 1.0),
            "gradient_preservation": min(avg_quality * 1.1, 1.0),
            "color_accuracy": min(avg_quality * 0.9, 1.0),
        }

    def _create_complex_gradient_layer_strategy(
        self, processing_result: Dict, max_layers: int
    ) -> Dict:
        """Create layer strategy for complex gradients."""
        processed_regions = processing_result.get("processed_regions", [])

        if not processed_regions:
            return {
                "total_layers": 1,
                "layer_distribution": [1],
                "complexity_handling": "simple",
            }

        # Calculate layer distribution
        total_layers = sum(r["layers_used"] for r in processed_regions)
        layer_distribution = [r["layers_used"] for r in processed_regions]

        return {
            "total_layers": min(total_layers, max_layers),
            "layer_distribution": layer_distribution,
            "complexity_handling": "adaptive",
            "regions_processed": len(processed_regions),
        }

    def _process_gradients_in_chunks(
        self,
        image: torch.Tensor,
        max_layers: int,
        chunk_size: int,
        device: torch.device,
    ) -> Dict:
        """Process gradients in chunks for large images."""
        _, _, height, width = image.shape
        chunk_results = []

        for i in range(0, height, chunk_size):
            for j in range(0, width, chunk_size):
                end_i = min(i + chunk_size, height)
                end_j = min(j + chunk_size, width)

                chunk = image[:, :, i:end_i, j:end_j]
                chunk_result = self._process_gradients_standard(
                    chunk, max_layers, device
                )

                chunk_results.append(
                    {"bounds": (i, j, end_i, end_j), "result": chunk_result}
                )

        return {
            "chunk_results": chunk_results,
            "total_chunks": len(chunk_results),
            "processing_method": "chunked",
        }

    def _process_gradients_standard(
        self, image: torch.Tensor, max_layers: int, device: torch.device
    ) -> Dict:
        """Standard gradient processing."""
        # Move to device
        image = image.to(device)

        # Detect gradients
        gradient_analysis = self.detect_gradient_regions(image)

        # Create layer combinations
        layer_combinations = self.create_layer_combinations_for_gradients(
            image, max_layers, smoothness_target=0.8
        )

        return {
            "gradient_analysis": gradient_analysis,
            "layer_combinations": layer_combinations,
            "processing_method": "standard",
        }

    def _calculate_processing_quality_score(self, result: Dict) -> float:
        """Calculate quality score for processing result."""
        if result.get("processing_method") == "chunked":
            # Average quality across chunks
            chunk_results = result.get("chunk_results", [])
            if not chunk_results:
                return 0.0

            quality_scores = []
            for chunk_data in chunk_results:
                chunk_result = chunk_data["result"]
                chunk_quality = self._calculate_single_result_quality(chunk_result)
                quality_scores.append(chunk_quality)

            return np.mean(quality_scores)
        else:
            return self._calculate_single_result_quality(result)

    def _calculate_single_result_quality(self, result: Dict) -> float:
        """Calculate quality score for single processing result."""
        gradient_analysis = result.get("gradient_analysis", {})
        layer_combinations = result.get("layer_combinations", {})

        # Quality based on detected regions and smoothness
        regions = gradient_analysis.get("gradient_regions", [])
        smoothness_metrics = layer_combinations.get("smoothness_metrics", {})

        region_quality = min(len(regions) / 5.0, 1.0)  # Normalize by expected regions
        smoothness_quality = smoothness_metrics.get("gradient_smoothness", 0.5)

        return (region_quality + smoothness_quality) / 2.0

    def _estimate_memory_usage(self, image: torch.Tensor, max_layers: int) -> Dict:
        """Estimate memory usage for processing."""
        # Calculate tensor sizes
        image_size = image.numel() * image.element_size()
        layer_size = image.shape[-2] * image.shape[-1] * max_layers * 4  # Float32

        total_memory = image_size + layer_size * 3  # Multiple intermediate tensors

        return {
            "image_memory_mb": image_size / (1024 * 1024),
            "layer_memory_mb": layer_size / (1024 * 1024),
            "total_estimated_mb": total_memory / (1024 * 1024),
        }

    def _measure_gradient_smoothness(self, layer_assignments: torch.Tensor) -> float:
        """Measure smoothness of gradients in layer assignments."""
        # Simple smoothness measurement
        if layer_assignments.shape[0] <= 1:
            return 1.0

        smoothness_scores = []
        num_layers, height, width = layer_assignments.shape

        for layer in range(num_layers):
            layer_data = layer_assignments[layer].float()

            # Calculate local variance as inverse of smoothness
            kernel = torch.ones(3, 3, device=self.device) / 9.0
            smoothed = F.conv2d(
                layer_data.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1,
            )

            variance = torch.var(layer_data - smoothed.squeeze()).item()
            smoothness = 1.0 / (1.0 + variance)
            smoothness_scores.append(smoothness)

        return np.mean(smoothness_scores)

    def _measure_layer_transition_quality(
        self, layer_assignments: torch.Tensor
    ) -> float:
        """Measure quality of transitions between layers."""
        if layer_assignments.shape[0] <= 1:
            return 1.0

        transition_scores = []
        num_layers = layer_assignments.shape[0]

        for layer in range(num_layers - 1):
            current = layer_assignments[layer]
            next_layer = layer_assignments[layer + 1]

            # Calculate similarity between adjacent layers
            similarity = torch.mean((current == next_layer).float()).item()
            transition_scores.append(similarity)

        return np.mean(transition_scores)

    def _measure_color_continuity(
        self, layer_assignments: torch.Tensor, image: torch.Tensor
    ) -> float:
        """Measure color continuity in layer assignments."""
        # Simplified color continuity measurement
        # This would require reconstructing colors from assignments

        # For now, return a reasonable default
        return 0.8
