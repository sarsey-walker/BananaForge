"""Base layer strategy optimization for transparency-based color mixing.

This module implements base layer color selection optimization to maximize
contrast and color palette expansion through strategic transparency usage.
"""

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import numpy as np
import torch

from ..utils.color import rgb_to_lab


@dataclass
class BaseLayerConfig:
    """Configuration for base layer optimization."""

    contrast_weight: float = 0.7
    palette_expansion_weight: float = 0.3
    max_base_colors: int = 3
    analysis_depth: str = "standard"  # 'fast', 'standard', 'detailed'
    enable_parallel_processing: bool = True


class BaseLayerOptimizer:
    """Optimizer for base layer color selection in transparency mixing.

    This class implements Story 4.5.3: Base Layer Strategy Optimization,
    focusing on selecting optimal dark base colors for maximum contrast
    and color palette expansion.
    """

    def __init__(
        self,
        contrast_weight: float = 0.7,
        palette_expansion_weight: float = 0.3,
        device: str = "cuda",
    ):
        """Initialize base layer optimizer.

        Args:
            contrast_weight: Weight for contrast optimization (0-1)
            palette_expansion_weight: Weight for palette expansion (0-1)
            device: Device for computations
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.contrast_weight = contrast_weight
        self.palette_expansion_weight = palette_expansion_weight

        # Validate weights
        if abs(contrast_weight + palette_expansion_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self._analysis_cache = {}
        self._cache_lock = Lock()

    def analyze_optimal_base_colors(
        self,
        image: torch.Tensor,
        available_base_colors: List[torch.Tensor],
        num_base_colors: int = 3,
    ) -> Dict:
        """Analyze image to identify optimal dark base colors.

        Args:
            image: Input image tensor (1, 3, H, W)
            available_base_colors: List of available base color tensors
            num_base_colors: Number of base colors to recommend

        Returns:
            Dictionary with analysis results and recommendations
        """
        # Extract image statistics
        image_stats = self._analyze_image_characteristics(image)

        # Analyze brightness of available colors
        brightness_analysis = self._analyze_color_brightness(available_base_colors)

        # Calculate contrast scores for each base color
        contrast_scores = self._calculate_contrast_scores(image, available_base_colors)

        # Rank colors by optimization criteria
        rankings = self._rank_base_colors_by_criteria(
            available_base_colors, contrast_scores, brightness_analysis, num_base_colors
        )

        return {
            "recommended_base_colors": rankings[:num_base_colors],
            "contrast_analysis": contrast_scores,
            "brightness_analysis": brightness_analysis,
            "image_characteristics": image_stats,
        }

    def optimize_for_contrast(
        self,
        image: torch.Tensor,
        available_base_colors: List[torch.Tensor],
        prioritize_dark: bool = True,
    ) -> Dict:
        """Optimize base color selection for maximum contrast.

        Args:
            image: Input image tensor (1, 3, H, W)
            available_base_colors: List of available base colors
            prioritize_dark: Whether to prioritize dark colors

        Returns:
            Dictionary with optimization results
        """
        # Calculate image brightness statistics
        image_brightness = torch.mean(image).item()

        # Calculate contrast scores for each base color
        contrast_scores = []
        for i, base_color in enumerate(available_base_colors):
            base_brightness = torch.mean(base_color).item()

            # Calculate contrast based on brightness difference
            contrast = abs(image_brightness - base_brightness)

            # Apply dark color prioritization
            if prioritize_dark and base_brightness < 0.3:
                contrast *= 1.5  # Boost dark colors

            contrast_scores.append(contrast)

        # Find optimal base color
        optimal_index = np.argmax(contrast_scores)

        return {
            "optimal_base_color": optimal_index,
            "contrast_scores": contrast_scores,
            "image_brightness": image_brightness,
            "base_color_brightness": [
                torch.mean(color).item() for color in available_base_colors
            ],
        }

    def optimize_base_colors_by_region(
        self,
        image: torch.Tensor,
        available_base_colors: List[torch.Tensor],
        region_size: int = 32,
    ) -> Dict:
        """Optimize base colors for each image region separately.

        Args:
            image: Input image tensor (1, 3, H, W)
            available_base_colors: List of available base colors
            region_size: Size of each region for analysis

        Returns:
            Dictionary with region-specific optimization results
        """
        _, _, height, width = image.shape

        # Divide image into regions
        regions = self._divide_image_into_regions(image, region_size)

        region_assignments = {}
        region_analysis = {}

        for region_id, region_data in regions.items():
            region_image = region_data["image"]
            region_bounds = region_data["bounds"]

            # Optimize base color for this region
            region_optimization = self.optimize_for_contrast(
                image=region_image,
                available_base_colors=available_base_colors,
                prioritize_dark=True,
            )

            # Calculate confidence score based on region characteristics
            confidence = self._calculate_region_confidence(
                region_image, region_optimization
            )

            # Calculate contrast improvement
            baseline_contrast = self._calculate_baseline_contrast(region_image)
            optimized_contrast = max(region_optimization["contrast_scores"])
            contrast_improvement = (optimized_contrast - baseline_contrast) / (
                baseline_contrast + 1e-6
            )

            region_assignments[region_id] = {
                "base_color_index": region_optimization["optimal_base_color"],
                "confidence": confidence,
                "contrast_improvement": contrast_improvement,
                "region_bounds": region_bounds,
            }

            region_analysis[region_id] = {
                "region_characteristics": self._analyze_region_characteristics(
                    region_image
                ),
                "optimization_details": region_optimization,
            }

        return {
            "region_assignments": region_assignments,
            "region_analysis": region_analysis,
            "total_regions": len(regions),
        }

    def optimize_for_palette_expansion(
        self,
        image: torch.Tensor,
        available_base_colors: List[torch.Tensor],
        overlay_colors: List[torch.Tensor],
        max_base_colors: int = 2,
    ) -> Dict:
        """Optimize base colors to maximize achievable color palette.

        Args:
            image: Input image tensor (1, 3, H, W)
            available_base_colors: List of available base colors
            overlay_colors: List of available overlay colors
            max_base_colors: Maximum number of base colors to select

        Returns:
            Dictionary with palette expansion optimization results
        """
        from .transparency_mixer import TransparencyColorMixer

        # Initialize transparency mixer for palette analysis
        mixer = TransparencyColorMixer(device=str(self.device))

        # Analyze each base color for palette expansion potential
        base_color_analysis = []
        for i, base_color in enumerate(available_base_colors):
            # Calculate achievable colors with this base
            achievable_colors = []
            for overlay_color in overlay_colors:
                mixed_colors = mixer.apply_three_layer_model(base_color, overlay_color)
                achievable_colors.extend(mixed_colors)

            # Calculate palette metrics
            unique_colors = self._find_unique_colors(achievable_colors)
            color_space_coverage = self._calculate_color_space_coverage(unique_colors)

            base_color_analysis.append(
                {
                    "base_color_index": i,
                    "achievable_colors": len(unique_colors),
                    "color_space_coverage": color_space_coverage,
                    "brightness": torch.mean(base_color).item(),
                }
            )

        # Select optimal combination of base colors
        selected_bases = self._select_optimal_base_combination(
            base_color_analysis, max_base_colors
        )

        # Calculate combined palette size
        combined_palette = self._calculate_combined_palette_size(
            selected_bases, available_base_colors, overlay_colors, mixer
        )

        # Compare with baseline (using brightest colors)
        baseline_palette = self._calculate_baseline_palette_size(
            available_base_colors, overlay_colors, mixer, max_base_colors
        )

        return {
            "selected_base_colors": selected_bases,
            "palette_size_comparison": {
                "combined_palette_size": combined_palette,
                "baseline_palette_size": baseline_palette,
                "improvement_factor": combined_palette / max(baseline_palette, 1),
            },
            "color_coverage_analysis": base_color_analysis,
        }

    def optimize_for_complex_image(
        self,
        image: torch.Tensor,
        available_base_colors: List[torch.Tensor],
        num_base_colors: int = 3,
        analysis_depth: str = "detailed",
    ) -> Dict:
        """Optimize base colors for complex images with detailed analysis.

        Args:
            image: Input image tensor (1, 3, H, W)
            available_base_colors: List of available base colors
            num_base_colors: Number of base colors to select
            analysis_depth: Depth of analysis ('fast', 'standard', 'detailed')

        Returns:
            Dictionary with comprehensive optimization results
        """
        # Analyze image complexity
        complexity_analysis = self._analyze_image_complexity(image)

        # Multi-criteria optimization
        criteria_results = {}

        # Contrast optimization
        contrast_result = self.optimize_for_contrast(
            image, available_base_colors, prioritize_dark=True
        )
        criteria_results["contrast"] = contrast_result

        # Region-based optimization if complex enough
        if complexity_analysis["complexity_score"] > 0.5:
            region_result = self.optimize_base_colors_by_region(
                image, available_base_colors, region_size=min(64, image.shape[-1] // 4)
            )
            criteria_results["region_based"] = region_result

        # Palette expansion optimization
        palette_result = self.optimize_for_palette_expansion(
            image, available_base_colors, available_base_colors, num_base_colors
        )
        criteria_results["palette_expansion"] = palette_result

        # Combine results using weighted scoring
        final_selection = self._combine_optimization_results(
            criteria_results, num_base_colors
        )

        # Calculate optimization quality metrics
        quality_metrics = self._calculate_optimization_quality(
            image, final_selection, available_base_colors
        )

        return {
            "selected_base_colors": final_selection,
            "complexity_analysis": complexity_analysis,
            "optimization_quality": quality_metrics,
            "criteria_breakdown": criteria_results,
        }

    def optimize_base_colors(
        self,
        image: torch.Tensor,
        available_base_colors: List[torch.Tensor],
        num_base_colors: int = 3,
        performance_mode: bool = False,
    ) -> Dict:
        """General base color optimization with performance options.

        Args:
            image: Input image tensor (1, 3, H, W)
            available_base_colors: List of available base colors
            num_base_colors: Number of base colors to select
            performance_mode: Whether to use faster algorithms

        Returns:
            Dictionary with optimization results
        """
        if performance_mode:
            # Fast optimization for large images
            return self._fast_optimization(
                image, available_base_colors, num_base_colors
            )
        else:
            # Standard optimization
            return self.analyze_optimal_base_colors(
                image, available_base_colors, num_base_colors
            )

    def _analyze_image_characteristics(self, image: torch.Tensor) -> Dict:
        """Analyze image characteristics for base color optimization."""
        # Calculate brightness statistics
        brightness = torch.mean(image).item()
        brightness_std = torch.std(image).item()

        # Calculate color diversity
        reshaped = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        color_std = torch.std(reshaped, dim=0)
        color_diversity = torch.mean(color_std).item()

        # Calculate contrast
        gray = torch.mean(image, dim=1, keepdim=True)
        contrast = torch.std(gray).item()

        return {
            "brightness": brightness,
            "brightness_std": brightness_std,
            "color_diversity": color_diversity,
            "contrast": contrast,
            "is_high_contrast": contrast > 0.2,
            "is_dark_image": brightness < 0.4,
        }

    def _analyze_color_brightness(self, colors: List[torch.Tensor]) -> List[Dict]:
        """Analyze brightness characteristics of available colors."""
        analysis = []

        for i, color in enumerate(colors):
            brightness = torch.mean(color).item()

            # Convert to LAB for better brightness analysis
            lab_color = rgb_to_lab(color.unsqueeze(0)).squeeze(0)
            lab_brightness = lab_color[0].item() / 100.0  # L component normalized

            analysis.append(
                {
                    "color_index": i,
                    "brightness": brightness,
                    "lab_brightness": lab_brightness,
                    "is_dark": brightness < 0.3,
                    "is_very_dark": brightness < 0.15,
                    "contrast_potential": 1.0
                    - brightness,  # Dark colors have high contrast potential
                }
            )

        return analysis

    def _calculate_contrast_scores(
        self, image: torch.Tensor, base_colors: List[torch.Tensor]
    ) -> List[float]:
        """Calculate contrast scores for each base color against the image."""
        # Extract dominant colors from image
        image_pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

        # Calculate histogram for dominant colors
        dominant_colors = self._extract_dominant_colors(image_pixels, num_colors=5)

        contrast_scores = []
        for base_color in base_colors:
            # Calculate contrast against dominant colors
            contrasts = []
            for dom_color in dominant_colors:
                # Use perceptual contrast in LAB space
                base_lab = rgb_to_lab(base_color.unsqueeze(0)).squeeze(0)
                dom_lab = rgb_to_lab(dom_color.unsqueeze(0)).squeeze(0)

                # Contrast is primarily difference in L (lightness)
                lightness_contrast = abs(base_lab[0] - dom_lab[0]) / 100.0

                # Add color contrast (a and b components)
                color_contrast = torch.norm(base_lab[1:] - dom_lab[1:]).item() / 200.0

                total_contrast = 0.8 * lightness_contrast + 0.2 * color_contrast
                contrasts.append(total_contrast)

            # Use average contrast as the score
            avg_contrast = np.mean(contrasts)
            contrast_scores.append(avg_contrast)

        return contrast_scores

    def _rank_base_colors_by_criteria(
        self,
        base_colors: List[torch.Tensor],
        contrast_scores: List[float],
        brightness_analysis: List[Dict],
        num_colors: int,
    ) -> List[Dict]:
        """Rank base colors using multiple criteria."""
        rankings = []

        for i, (color, contrast, brightness_info) in enumerate(
            zip(base_colors, contrast_scores, brightness_analysis)
        ):
            # Calculate composite score
            contrast_score = contrast
            darkness_bonus = 1.0 if brightness_info["is_dark"] else 0.5
            very_dark_bonus = 1.2 if brightness_info["is_very_dark"] else 1.0

            composite_score = (
                self.contrast_weight * contrast_score * darkness_bonus * very_dark_bonus
                + self.palette_expansion_weight * brightness_info["contrast_potential"]
            )

            rankings.append(
                {
                    "color_index": i,
                    "brightness": brightness_info["brightness"],
                    "contrast_score": contrast_score,
                    "composite_score": composite_score,
                    "is_dark": brightness_info["is_dark"],
                }
            )

        # Sort by composite score
        rankings.sort(key=lambda x: x["composite_score"], reverse=True)

        return rankings[:num_colors]

    def _divide_image_into_regions(
        self, image: torch.Tensor, region_size: int
    ) -> Dict[str, Dict]:
        """Divide image into regions for analysis."""
        _, _, height, width = image.shape
        regions = {}

        region_id = 0
        for i in range(0, height, region_size):
            for j in range(0, width, region_size):
                end_i = min(i + region_size, height)
                end_j = min(j + region_size, width)

                region_image = image[:, :, i:end_i, j:end_j]

                regions[f"region_{region_id}"] = {
                    "image": region_image,
                    "bounds": (i, j, end_i, end_j),
                }
                region_id += 1

        return regions

    def _calculate_region_confidence(
        self, region_image: torch.Tensor, optimization_result: Dict
    ) -> float:
        """Calculate confidence score for region optimization."""
        # Base confidence on contrast variation and optimization strength
        contrast_scores = optimization_result["contrast_scores"]
        max_contrast = max(contrast_scores)
        contrast_variance = np.var(contrast_scores)

        # High max contrast and low variance = high confidence
        confidence = max_contrast * (1.0 - min(contrast_variance, 1.0))

        return min(confidence, 1.0)

    def _calculate_baseline_contrast(self, image: torch.Tensor) -> float:
        """Calculate baseline contrast for an image region."""
        # Simple contrast calculation using standard deviation
        gray = torch.mean(image, dim=1, keepdim=True)
        return torch.std(gray).item()

    def _analyze_region_characteristics(self, region_image: torch.Tensor) -> Dict:
        """Analyze characteristics of an image region."""
        return {
            "brightness": torch.mean(region_image).item(),
            "contrast": torch.std(region_image).item(),
            "color_variation": torch.std(region_image, dim=(2, 3)).mean().item(),
            "size": region_image.shape[-2:],
        }

    def _find_unique_colors(
        self, colors: List[torch.Tensor], threshold: float = 0.05
    ) -> List[torch.Tensor]:
        """Find unique colors in a list with similarity threshold."""
        if not colors:
            return []

        unique_colors = [colors[0]]

        for color in colors[1:]:
            is_unique = True
            for unique_color in unique_colors:
                if torch.norm(color - unique_color).item() < threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_colors.append(color)

        return unique_colors

    def _calculate_color_space_coverage(self, colors: List[torch.Tensor]) -> float:
        """Calculate how much of the color space is covered by a color list."""
        if len(colors) <= 1:
            return 0.0

        color_tensor = torch.stack(colors)

        # Calculate volume of bounding box in RGB space
        min_values = torch.min(color_tensor, dim=0)[0]
        max_values = torch.max(color_tensor, dim=0)[0]
        coverage_volume = torch.prod(max_values - min_values).item()

        # Normalize by theoretical maximum volume (1.0 for RGB cube)
        return min(coverage_volume, 1.0)

    def _select_optimal_base_combination(
        self, base_analysis: List[Dict], max_bases: int
    ) -> List[Dict]:
        """Select optimal combination of base colors."""
        # Sort by combined score of achievable colors and coverage
        scored_bases = []
        for analysis in base_analysis:
            score = (
                0.6
                * analysis["achievable_colors"]
                / 10.0  # Normalize achievable colors
                + 0.4 * analysis["color_space_coverage"]
            )

            # Bonus for dark colors
            if analysis["brightness"] < 0.3:
                score *= 1.2

            scored_bases.append({**analysis, "selection_score": score})

        # Sort and select top bases
        scored_bases.sort(key=lambda x: x["selection_score"], reverse=True)
        return scored_bases[:max_bases]

    def _calculate_combined_palette_size(
        self,
        selected_bases: List[Dict],
        base_colors: List[torch.Tensor],
        overlay_colors: List[torch.Tensor],
        mixer,
    ) -> int:
        """Calculate combined palette size for selected base colors."""
        all_achievable = []

        for base_info in selected_bases:
            base_color = base_colors[base_info["base_color_index"]]
            for overlay_color in overlay_colors:
                mixed_colors = mixer.apply_three_layer_model(base_color, overlay_color)
                all_achievable.extend(mixed_colors)

        unique_colors = self._find_unique_colors(all_achievable)
        return len(unique_colors)

    def _calculate_baseline_palette_size(
        self,
        base_colors: List[torch.Tensor],
        overlay_colors: List[torch.Tensor],
        mixer,
        num_bases: int,
    ) -> int:
        """Calculate baseline palette size using brightest colors."""
        # Select brightest colors as baseline
        brightness_scores = [torch.mean(color).item() for color in base_colors]
        sorted_indices = sorted(
            range(len(base_colors)), key=lambda i: brightness_scores[i], reverse=True
        )

        baseline_bases = sorted_indices[:num_bases]

        all_achievable = []
        for base_idx in baseline_bases:
            base_color = base_colors[base_idx]
            for overlay_color in overlay_colors:
                mixed_colors = mixer.apply_three_layer_model(base_color, overlay_color)
                all_achievable.extend(mixed_colors)

        unique_colors = self._find_unique_colors(all_achievable)
        return len(unique_colors)

    def _analyze_image_complexity(self, image: torch.Tensor) -> Dict:
        """Analyze image complexity for optimization strategy selection."""
        # Edge detection for complexity
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=self.device,
        )

        gray = torch.mean(image, dim=1, keepdim=True)

        # Apply edge detection
        edges_x = torch.nn.functional.conv2d(
            gray, sobel_x.unsqueeze(0).unsqueeze(0), padding=1
        )
        edges_y = torch.nn.functional.conv2d(
            gray, sobel_y.unsqueeze(0).unsqueeze(0), padding=1
        )
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)

        complexity_score = torch.mean(edge_magnitude).item()

        # Color complexity
        color_variance = (
            torch.var(image.squeeze(0).permute(1, 2, 0).reshape(-1, 3), dim=0)
            .mean()
            .item()
        )

        return {
            "complexity_score": complexity_score,
            "color_variance": color_variance,
            "edge_density": torch.mean((edge_magnitude > 0.1).float()).item(),
            "requires_region_analysis": complexity_score > 0.5,
        }

    def _combine_optimization_results(
        self, criteria_results: Dict, num_base_colors: int
    ) -> List[Dict]:
        """Combine results from multiple optimization criteria."""
        # Weight different criteria results
        weights = {"contrast": 0.5, "region_based": 0.3, "palette_expansion": 0.2}

        # Collect all candidate base colors with scores
        candidates = {}

        # Add contrast results
        if "contrast" in criteria_results:
            contrast_result = criteria_results["contrast"]
            optimal_idx = contrast_result["optimal_base_color"]
            score = (
                contrast_result["contrast_scores"][optimal_idx] * weights["contrast"]
            )

            candidates[optimal_idx] = candidates.get(optimal_idx, 0) + score

        # Add region-based results
        if "region_based" in criteria_results:
            region_result = criteria_results["region_based"]
            for region_data in region_result["region_assignments"].values():
                base_idx = region_data["base_color_index"]
                score = region_data["confidence"] * weights["region_based"]
                candidates[base_idx] = candidates.get(base_idx, 0) + score

        # Add palette expansion results
        if "palette_expansion" in criteria_results:
            palette_result = criteria_results["palette_expansion"]
            for base_info in palette_result["selected_base_colors"]:
                base_idx = base_info["base_color_index"]
                score = base_info["selection_score"] * weights["palette_expansion"]
                candidates[base_idx] = candidates.get(base_idx, 0) + score

        # Sort and select top candidates
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        selected = []
        for base_idx, score in sorted_candidates[:num_base_colors]:
            selected.append(
                {
                    "base_color_index": base_idx,
                    "combined_score": score,
                    "selection_confidence": min(score, 1.0),
                }
            )

        return selected

    def _calculate_optimization_quality(
        self,
        image: torch.Tensor,
        final_selection: List[Dict],
        available_base_colors: List[torch.Tensor],
    ) -> Dict:
        """Calculate quality metrics for the optimization result."""
        if not final_selection:
            return {
                "color_coverage": 0.0,
                "contrast_improvement": 0.0,
                "palette_expansion": 1.0,
            }

        # Extract selected colors
        selected_colors = [
            available_base_colors[sel["base_color_index"]] for sel in final_selection
        ]

        # Calculate color coverage
        image_pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        coverage = self._calculate_color_coverage_quality(image_pixels, selected_colors)

        # Calculate contrast improvement
        contrast_improvement = self._calculate_contrast_improvement(
            image, selected_colors
        )

        # Calculate palette expansion factor
        baseline_colors = len(available_base_colors)
        # Estimate achievable colors (simplified)
        estimated_achievable = (
            len(selected_colors) * len(available_base_colors) * 3
        )  # 3 opacity levels
        expansion_factor = estimated_achievable / max(baseline_colors, 1)

        return {
            "color_coverage": coverage,
            "contrast_improvement": contrast_improvement,
            "palette_expansion": min(expansion_factor, 5.0),  # Cap at 5x expansion
        }

    def _fast_optimization(
        self,
        image: torch.Tensor,
        available_base_colors: List[torch.Tensor],
        num_base_colors: int,
    ) -> Dict:
        """Fast optimization for performance mode."""
        # Simplified analysis for speed
        brightness_scores = [
            torch.mean(color).item() for color in available_base_colors
        ]

        # Select darkest colors for good contrast
        dark_indices = sorted(
            range(len(brightness_scores)), key=lambda i: brightness_scores[i]
        )

        selected = []
        for i in range(min(num_base_colors, len(dark_indices))):
            idx = dark_indices[i]
            selected.append(
                {
                    "color_index": idx,
                    "brightness": brightness_scores[idx],
                    "contrast_score": 1.0
                    - brightness_scores[idx],  # Darker = higher contrast potential
                    "composite_score": 1.0 - brightness_scores[idx],
                }
            )

        return {
            "recommended_base_colors": selected,
            "optimization_mode": "fast",
            "processing_time_optimized": True,
        }

    def _extract_dominant_colors(
        self, pixels: torch.Tensor, num_colors: int = 5
    ) -> List[torch.Tensor]:
        """Extract dominant colors from pixel data using simplified clustering."""
        # Simple uniform sampling for speed
        num_samples = min(1000, len(pixels))
        indices = torch.randperm(len(pixels))[:num_samples]
        sampled_pixels = pixels[indices]

        # K-means-like clustering (simplified)
        if len(sampled_pixels) < num_colors:
            return [sampled_pixels[i] for i in range(len(sampled_pixels))]

        # Initialize clusters randomly
        cluster_centers = sampled_pixels[
            torch.randperm(len(sampled_pixels))[:num_colors]
        ]

        # Simple iterative refinement (few iterations for speed)
        for _ in range(3):
            # Assign pixels to closest cluster
            distances = torch.cdist(sampled_pixels, cluster_centers)
            assignments = torch.argmin(distances, dim=1)

            # Update cluster centers
            for i in range(num_colors):
                mask = assignments == i
                if mask.sum() > 0:
                    cluster_centers[i] = torch.mean(sampled_pixels[mask], dim=0)

        return [cluster_centers[i] for i in range(num_colors)]

    def _calculate_color_coverage_quality(
        self, image_pixels: torch.Tensor, selected_colors: List[torch.Tensor]
    ) -> float:
        """Calculate how well selected colors cover the image color space."""
        if not selected_colors:
            return 0.0

        # Calculate color space bounds
        image_min = torch.min(image_pixels, dim=0)[0]
        image_max = torch.max(image_pixels, dim=0)[0]
        image_range = image_max - image_min

        selected_tensor = torch.stack(selected_colors)
        selected_min = torch.min(selected_tensor, dim=0)[0]
        selected_max = torch.max(selected_tensor, dim=0)[0]
        selected_range = selected_max - selected_min

        # Coverage is how much of the image color space is spanned
        coverage_ratio = selected_range / (image_range + 1e-6)
        coverage = torch.mean(torch.clamp(coverage_ratio, 0, 1)).item()

        return coverage

    def _calculate_contrast_improvement(
        self, image: torch.Tensor, selected_colors: List[torch.Tensor]
    ) -> float:
        """Calculate contrast improvement from selected base colors."""
        # Calculate baseline contrast
        baseline_contrast = torch.std(image).item()

        # Estimate improved contrast with dark base colors
        avg_selected_brightness = torch.mean(torch.stack(selected_colors)).item()
        image_brightness = torch.mean(image).item()

        # Contrast improvement based on brightness difference
        brightness_diff = abs(image_brightness - avg_selected_brightness)
        improvement = brightness_diff / (baseline_contrast + 1e-6)

        return min(improvement, 2.0)  # Cap at 2x improvement
