"""Transparency-based color mixing system for advanced 3D printing optimization.

This module implements the transparency color mixing system that enables creating
more colors with fewer materials through strategic layer transparency mixing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..utils.color import rgb_to_lab


@dataclass
class TransparencyConfig:
    """Configuration for transparency mixing."""

    opacity_levels: List[float] = None
    blending_method: str = "alpha_composite"
    max_layers: int = 3
    enable_optimization: bool = True
    quality_threshold: float = 0.85

    def __post_init__(self):
        if self.opacity_levels is None:
            self.opacity_levels = [0.33, 0.67, 1.0]


class TransparencyColorMixer:
    """Advanced color mixing through layer transparency.

    This class implements the three-layer opacity model and transparency-aware
    color calculations for Feature 4.5.
    """

    def __init__(
        self,
        opacity_levels: Optional[List[float]] = None,
        blending_method: str = "alpha_composite",
        max_layers: int = 3,
        device: str = "cuda",
    ):
        """Initialize transparency color mixer.

        Args:
            opacity_levels: Opacity levels for layer mixing (default: [0.33, 0.67, 1.0])
            blending_method: Method for color blending ('alpha_composite', 'linear')
            max_layers: Maximum number of layers for mixing
            device: Device for computations
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.opacity_levels = opacity_levels or [0.33, 0.67, 1.0]
        self.blending_method = blending_method
        self.max_layers = max_layers

        # Validate opacity levels
        if len(self.opacity_levels) != 3:
            raise ValueError("Opacity levels must contain exactly 3 values")
        if not all(0 < level <= 1.0 for level in self.opacity_levels):
            raise ValueError("All opacity levels must be between 0 and 1")
        if not all(
            self.opacity_levels[i] <= self.opacity_levels[i + 1] for i in range(2)
        ):
            raise ValueError("Opacity levels must be in ascending order")

    def calculate_layer_opacities(
        self,
        base_color: torch.Tensor,
        overlay_color: torch.Tensor,
        target_layers: int = 3,
    ) -> List[float]:
        """Calculate layer opacities for three-layer model.

        Args:
            base_color: Base color RGB tensor (3,)
            overlay_color: Overlay color RGB tensor (3,)
            target_layers: Target number of layers

        Returns:
            List of opacity values for each layer
        """
        if target_layers > len(self.opacity_levels):
            # Extend opacity levels if more layers requested
            extended_levels = self.opacity_levels.copy()
            step = (1.0 - self.opacity_levels[-1]) / (
                target_layers - len(self.opacity_levels)
            )
            for i in range(target_layers - len(self.opacity_levels)):
                extended_levels.append(
                    min(1.0, self.opacity_levels[-1] + (i + 1) * step)
                )
            return extended_levels[:target_layers]

        return self.opacity_levels[:target_layers]

    def mix_colors_with_transparency(
        self,
        base_color: torch.Tensor,
        overlay_color: torch.Tensor,
        opacity_levels: List[float],
    ) -> List[torch.Tensor]:
        """Mix colors using transparency alpha blending.

        Args:
            base_color: Base color RGB tensor (3,)
            overlay_color: Overlay color RGB tensor (3,)
            opacity_levels: List of opacity values for mixing

        Returns:
            List of mixed color tensors
        """
        mixed_colors = []

        for alpha in opacity_levels:
            if self.blending_method == "alpha_composite":
                # Standard alpha compositing: result = base * (1 - alpha) + overlay * alpha
                mixed_color = base_color * (1 - alpha) + overlay_color * alpha
            elif self.blending_method == "linear":
                # Linear interpolation
                mixed_color = torch.lerp(base_color, overlay_color, alpha)
            else:
                raise ValueError(f"Unknown blending method: {self.blending_method}")

            # Clamp to valid color range
            mixed_color = torch.clamp(mixed_color, 0.0, 1.0)
            mixed_colors.append(mixed_color)

        return mixed_colors

    def create_layer_sequence(
        self, base_color: torch.Tensor, overlay_color: torch.Tensor, num_layers: int
    ) -> List[torch.Tensor]:
        """Create smooth layer sequence with gradual opacity changes.

        Args:
            base_color: Base color RGB tensor (3,)
            overlay_color: Overlay color RGB tensor (3,)
            num_layers: Number of layers in sequence

        Returns:
            List of color tensors for each layer
        """
        if num_layers <= 1:
            return [base_color]

        # Generate smooth opacity progression
        opacity_progression = torch.linspace(0.0, 1.0, num_layers)

        layer_sequence = []
        for alpha in opacity_progression:
            mixed_color = base_color * (1 - alpha) + overlay_color * alpha
            mixed_color = torch.clamp(mixed_color, 0.0, 1.0)
            layer_sequence.append(mixed_color)

        return layer_sequence

    def calculate_opacity_progression(
        self, base_color: torch.Tensor, overlay_color: torch.Tensor, steps: int
    ) -> List[float]:
        """Calculate opacity progression for smooth transitions.

        Args:
            base_color: Base color RGB tensor (3,)
            overlay_color: Overlay color RGB tensor (3,)
            steps: Number of steps in progression

        Returns:
            List of opacity values
        """
        # Use perceptual color space for better progression
        base_lab = rgb_to_lab(base_color.unsqueeze(0)).squeeze(0)
        overlay_lab = rgb_to_lab(overlay_color.unsqueeze(0)).squeeze(0)

        # Calculate perceptual distance
        perceptual_distance = torch.norm(overlay_lab - base_lab).item()

        # Adjust opacity progression based on perceptual distance
        if perceptual_distance > 50:  # Large perceptual difference
            # Use more gradual progression for better visual result
            progression = torch.pow(torch.linspace(0, 1, steps), 0.8)
        else:
            # Use linear progression for similar colors
            progression = torch.linspace(0, 1, steps)

        return progression.tolist()

    def apply_three_layer_model(
        self, base_color: torch.Tensor, overlay_color: torch.Tensor
    ) -> List[torch.Tensor]:
        """Apply the three-layer opacity model (33%, 67%, 100%).

        Args:
            base_color: Base color RGB tensor (3,)
            overlay_color: Overlay color RGB tensor (3,)

        Returns:
            List of 3 color tensors for the three opacity levels
        """
        return self.mix_colors_with_transparency(
            base_color=base_color,
            overlay_color=overlay_color,
            opacity_levels=self.opacity_levels,
        )

    def apply_opacity_model_to_image(
        self,
        base_image: torch.Tensor,
        overlay_image: torch.Tensor,
        opacity_levels: List[float],
    ) -> torch.Tensor:
        """Apply opacity model to entire images.

        Args:
            base_image: Base image tensor (1, 3, H, W)
            overlay_image: Overlay image tensor (1, 3, H, W)
            opacity_levels: List of opacity values

        Returns:
            Stacked opacity layers tensor (num_levels, 3, H, W)
        """
        batch_size, channels, height, width = base_image.shape
        num_levels = len(opacity_levels)

        # Create output tensor
        opacity_layers = torch.zeros(
            num_levels,
            channels,
            height,
            width,
            device=self.device,
            dtype=base_image.dtype,
        )

        # Apply opacity mixing for each level
        for i, alpha in enumerate(opacity_levels):
            opacity_layers[i] = base_image[0] * (1 - alpha) + overlay_image[0] * alpha

        # Clamp to valid range
        opacity_layers = torch.clamp(opacity_layers, 0.0, 1.0)

        return opacity_layers

    def compute_achievable_colors(
        self,
        filament_colors: List[torch.Tensor],
        max_layers: int = 3,
        optimize_performance: bool = False,
        grayscale_only: bool = False,
    ) -> List[Dict]:
        """Compute all achievable colors through transparency mixing.

        Args:
            filament_colors: List of available filament color tensors
            max_layers: Maximum layers for mixing
            optimize_performance: Whether to optimize for performance
            grayscale_only: Whether to restrict to grayscale combinations only

        Returns:
            List of dictionaries containing color combinations and metadata
        """
        achievable_colors = []

        # Ensure all filament colors are on the correct device
        filament_colors = [color.to(self.device) for color in filament_colors]

        # Filter to grayscale colors if requested
        if grayscale_only:
            filament_colors = self._filter_grayscale_colors(filament_colors)

        # Add base colors (single layer)
        for i, base_color in enumerate(filament_colors):
            achievable_colors.append(
                {
                    "color": base_color,
                    "base_material": i,
                    "overlay_material": i,
                    "opacity": 1.0,
                    "layers": 1,
                    "accuracy_metrics": self._calculate_color_accuracy_metrics(
                        base_color, base_color
                    ),
                }
            )

        # Add mixed colors (multi-layer)
        for base_idx, base_color in enumerate(filament_colors):
            for overlay_idx, overlay_color in enumerate(filament_colors):
                if base_idx == overlay_idx:
                    continue  # Skip same color combinations

                # Try different opacity levels
                for opacity in self.opacity_levels:
                    mixed_color = base_color * (1 - opacity) + overlay_color * opacity
                    mixed_color = torch.clamp(mixed_color, 0.0, 1.0)

                    # Calculate how many layers this represents
                    layer_count = min(max_layers, int(opacity / (1.0 / max_layers)) + 1)

                    achievable_colors.append(
                        {
                            "color": mixed_color,
                            "base_material": base_idx,
                            "overlay_material": overlay_idx,
                            "opacity": opacity,
                            "layers": layer_count,
                            "accuracy_metrics": self._calculate_color_accuracy_metrics(
                                mixed_color, overlay_color
                            ),
                        }
                    )

        return achievable_colors

    def map_rgb_to_layer_combination(
        self,
        target_rgb: torch.Tensor,
        available_filaments: List[torch.Tensor],
        max_layers: int = 3,
    ) -> Dict:
        """Map target RGB to best transparency layer combination.

        Args:
            target_rgb: Target color RGB tensor (3,)
            available_filaments: List of available filament colors
            max_layers: Maximum layers for mixing

        Returns:
            Dictionary with best match and layer recipe
        """
        achievable_colors = self.compute_achievable_colors(
            available_filaments, max_layers
        )

        best_match = None
        best_error = float("inf")

        # Find closest achievable color
        for color_combo in achievable_colors:
            color_error = torch.norm(color_combo["color"] - target_rgb).item()

            if color_error < best_error:
                best_error = color_error
                best_match = color_combo.copy()
                best_match["achieved_color"] = color_combo["color"]

        return {
            "best_match": best_match,
            "color_error": best_error,
            "layer_recipe": (
                self._create_layer_recipe(best_match) if best_match else None
            ),
        }

    def analyze_palette_expansion(
        self, filament_colors: List[torch.Tensor], max_layers: int = 3
    ) -> Dict:
        """Analyze how transparency expands the color palette.

        Args:
            filament_colors: List of available filament colors
            max_layers: Maximum layers for analysis

        Returns:
            Dictionary with palette expansion analysis
        """
        # Calculate achievable colors
        achievable_colors = self.compute_achievable_colors(filament_colors, max_layers)

        # Analyze base color effectiveness
        base_color_rankings = []
        expansion_metrics = []

        for base_idx, base_color in enumerate(filament_colors):
            # Count unique colors achievable with this base
            unique_colors = []
            for combo in achievable_colors:
                if combo["base_material"] == base_idx:
                    # Check if color is unique (not too similar to existing)
                    is_unique = True
                    for existing_color in unique_colors:
                        if torch.norm(combo["color"] - existing_color) < 0.05:
                            is_unique = False
                            break
                    if is_unique:
                        unique_colors.append(combo["color"])

            # Calculate color space coverage
            if len(unique_colors) > 1:
                color_tensor = torch.stack(unique_colors)
                color_range = (
                    torch.max(color_tensor, dim=0)[0]
                    - torch.min(color_tensor, dim=0)[0]
                )
                coverage = torch.mean(color_range).item()
            else:
                coverage = 0.0

            base_color_rankings.append(
                {
                    "color_index": base_idx,
                    "unique_colors_generated": len(unique_colors),
                    "color_space_coverage": coverage,
                    "rank": 0,  # Will be set after sorting
                }
            )

            expansion_metrics.append(
                {
                    "color_index": base_idx,
                    "unique_colors_generated": len(unique_colors),
                    "color_space_coverage": coverage,
                }
            )

        # Rank base colors by expansion potential
        base_color_rankings.sort(
            key=lambda x: (x["unique_colors_generated"], x["color_space_coverage"]),
            reverse=True,
        )

        # Set ranks
        for i, ranking in enumerate(base_color_rankings):
            ranking["rank"] = i + 1

        return {
            "base_color_rankings": base_color_rankings,
            "expansion_metrics": expansion_metrics,
            "total_achievable_colors": len(achievable_colors),
            "expansion_factor": len(achievable_colors) / len(filament_colors),
        }

    def compute_combinations_with_accuracy(
        self,
        filament_colors: List[torch.Tensor],
        max_layers: int = 3,
        include_metrics: bool = True,
    ) -> List[Dict]:
        """Compute color combinations with detailed accuracy metrics.

        Args:
            filament_colors: List of available filament colors
            max_layers: Maximum layers for mixing
            include_metrics: Whether to include detailed accuracy metrics

        Returns:
            List of color combinations with accuracy metrics
        """
        combinations = self.compute_achievable_colors(filament_colors, max_layers)

        if not include_metrics:
            return combinations

        # Add detailed accuracy metrics to each combination
        for combo in combinations:
            overlay_color = filament_colors[combo["overlay_material"]]
            achieved_color = combo["color"]

            # Enhanced accuracy metrics
            combo["accuracy_metrics"] = {
                "delta_e": self._calculate_delta_e(achieved_color, overlay_color),
                "rgb_error": torch.norm(achieved_color - overlay_color).item(),
                "perceptual_error": self._calculate_perceptual_error(
                    achieved_color, overlay_color
                ),
                "color_gamut_coverage": self._calculate_gamut_coverage(
                    achieved_color, filament_colors
                ),
            }

        return combinations

    def _calculate_color_accuracy_metrics(
        self, achieved_color: torch.Tensor, target_color: torch.Tensor
    ) -> Dict:
        """Calculate accuracy metrics for a color combination."""
        return {
            "delta_e": self._calculate_delta_e(achieved_color, target_color),
            "rgb_error": torch.norm(achieved_color - target_color).item(),
            "perceptual_error": self._calculate_perceptual_error(
                achieved_color, target_color
            ),
            "color_gamut_coverage": 0.5,  # Placeholder
        }

    def _calculate_delta_e(self, color1: torch.Tensor, color2: torch.Tensor) -> float:
        """Calculate Delta-E color difference in LAB space."""
        # Ensure colors are on the same device
        color1 = color1.to(self.device)
        color2 = color2.to(self.device)

        lab1 = rgb_to_lab(color1.unsqueeze(0)).squeeze(0).to(self.device)
        lab2 = rgb_to_lab(color2.unsqueeze(0)).squeeze(0).to(self.device)

        # Simplified Delta-E calculation
        delta_e = torch.norm(lab1 - lab2).item()
        return min(delta_e, 100.0)  # Cap at 100

    def _calculate_perceptual_error(
        self, color1: torch.Tensor, color2: torch.Tensor
    ) -> float:
        """Calculate perceptual color error."""
        # Ensure colors are on the same device
        color1 = color1.to(self.device)
        color2 = color2.to(self.device)

        # Convert to perceptual color space and calculate weighted difference
        lab1 = rgb_to_lab(color1.unsqueeze(0)).squeeze(0).to(self.device)
        lab2 = rgb_to_lab(color2.unsqueeze(0)).squeeze(0).to(self.device)

        # Weight L, a, b components differently for perceptual accuracy
        weights = torch.tensor([1.0, 0.5, 0.5], device=self.device)
        weighted_diff = weights * torch.abs(lab1 - lab2)

        return torch.mean(weighted_diff).item()

    def _calculate_gamut_coverage(
        self, color: torch.Tensor, reference_colors: List[torch.Tensor]
    ) -> float:
        """Calculate how well a color covers the available gamut."""
        if len(reference_colors) <= 1:
            return 0.5

        # Ensure all tensors are on the same device
        color = color.to(self.device)
        reference_colors = [ref_color.to(self.device) for ref_color in reference_colors]

        # Find position of color within the gamut defined by reference colors
        ref_tensor = torch.stack(reference_colors)
        gamut_min = torch.min(ref_tensor, dim=0)[0]
        gamut_max = torch.max(ref_tensor, dim=0)[0]
        gamut_range = gamut_max - gamut_min

        # Normalize color position within gamut
        normalized_pos = (color - gamut_min) / (gamut_range + 1e-6)

        # Coverage is how well the color explores the available gamut
        coverage = torch.mean(
            torch.clamp(normalized_pos, 0, 1) * (1 - torch.clamp(normalized_pos, 0, 1))
        ).item()

        return min(coverage * 4, 1.0)  # Scale and cap at 1.0

    def _create_layer_recipe(self, match_info: Dict) -> Dict:
        """Create layer recipe from match information."""
        return {
            "base_material_index": match_info["base_material"],
            "overlay_material_index": match_info["overlay_material"],
            "opacity_level": match_info["opacity"],
            "layer_count": match_info["layers"],
            "mixing_instructions": f"Apply {match_info['layers']} layers of overlay material at {match_info['opacity']:.1%} opacity",
        }

    def is_grayscale_image(self, image: torch.Tensor, threshold: float = 0.05) -> bool:
        """Check if image is effectively grayscale.

        Args:
            image: Input image (1, 3, H, W) or (3, H, W)
            threshold: Color variation threshold for grayscale detection

        Returns:
            True if image is grayscale, False otherwise
        """
        # Handle different input shapes
        if image.dim() == 4:
            image = image.squeeze(0)
        elif image.dim() != 3 or image.shape[0] != 3:
            raise ValueError("Image must be (1, 3, H, W) or (3, H, W)")

        # Extract RGB channels
        r, g, b = image[0], image[1], image[2]

        # Calculate differences between channels
        rg_diff = torch.abs(r - g).mean().item()
        rb_diff = torch.abs(r - b).mean().item()
        gb_diff = torch.abs(g - b).mean().item()

        # Check if all channel differences are below threshold
        max_diff = max(rg_diff, rb_diff, gb_diff)
        return max_diff < threshold

    def _filter_grayscale_colors(
        self, colors: List[torch.Tensor], threshold: float = 0.1
    ) -> List[torch.Tensor]:
        """Filter color list to include only grayscale colors.

        Args:
            colors: List of color tensors (3,)
            threshold: Threshold for grayscale detection

        Returns:
            List of grayscale color tensors
        """
        grayscale_colors = []

        for color in colors:
            # Check if color is grayscale (R≈G≈B)
            r, g, b = color[0].item(), color[1].item(), color[2].item()
            color_diff = max(abs(r - g), abs(r - b), abs(g - b))

            # Consider it grayscale if color channels are very similar
            if color_diff < threshold:
                grayscale_colors.append(color)

        return grayscale_colors

    def compute_grayscale_mixing_options(
        self, grayscale_colors: List[torch.Tensor], max_layers: int = 3
    ) -> Dict:
        """Compute grayscale-specific mixing options.

        Args:
            grayscale_colors: List of grayscale color tensors
            max_layers: Maximum layers for mixing

        Returns:
            Dictionary with grayscale mixing analysis
        """
        # Sort grayscale colors by luminance
        sorted_colors = sorted(
            grayscale_colors,
            key=lambda c: (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]).item(),
        )

        mixing_options = []

        # Generate all possible grayscale combinations
        for i, dark_color in enumerate(sorted_colors):
            for j, light_color in enumerate(sorted_colors[i + 1 :], i + 1):
                for opacity in self.opacity_levels:
                    mixed_color = dark_color * (1 - opacity) + light_color * opacity
                    mixed_color = torch.clamp(mixed_color, 0.0, 1.0)

                    mixing_options.append(
                        {
                            "dark_base_index": i,
                            "light_overlay_index": j,
                            "mixed_color": mixed_color,
                            "opacity": opacity,
                            "luminance": (
                                0.299 * mixed_color[0]
                                + 0.587 * mixed_color[1]
                                + 0.114 * mixed_color[2]
                            ).item(),
                        }
                    )

        # Sort by luminance for smooth gradients
        mixing_options.sort(key=lambda x: x["luminance"])

        return {
            "sorted_base_colors": sorted_colors,
            "mixing_options": mixing_options,
            "total_grayscale_combinations": len(mixing_options),
            "luminance_range": {
                "min": mixing_options[0]["luminance"] if mixing_options else 0.0,
                "max": mixing_options[-1]["luminance"] if mixing_options else 1.0,
            },
        }
