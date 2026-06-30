"""Color space conversion utilities for BananaForge."""

from typing import Tuple

import torch


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    rgb_values = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (rgb_values[0], rgb_values[1], rgb_values[2])


class ColorConverter:
    """Color space conversion utilities with focus on perceptual accuracy."""

    def __init__(self, device: str = "cpu"):
        """Initialize color converter.

        Args:
            device: Device for tensor operations
        """
        self.device = torch.device(device)

        # D65 illuminant white point for XYZ conversion
        self.white_point = torch.tensor(
            [0.95047, 1.0, 1.08883], dtype=torch.float32, device=self.device
        )

        # sRGB to XYZ conversion matrix (D65 illuminant)
        self.rgb_to_xyz_matrix = torch.tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def rgb_to_lab(self, rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert RGB image to CIELAB color space.

        Implements the complete sRGB -> XYZ -> LAB conversion pipeline
        with proper gamma correction and perceptual uniformity.

        Args:
            rgb: RGB image tensor in range [0, 255] with shape (B, 3, H, W)
            eps: Small epsilon value to stabilize power operations

        Returns:
            LAB image tensor with L* in [0, 100], a* and b* in [-128, 127]
        """
        # Normalize RGB to [0, 1] and clamp to avoid numerical issues
        rgb_normalized = torch.clamp(rgb / 255.0, 0.0, 1.0)

        # Apply inverse gamma correction to get linear RGB
        rgb_linear = self._srgb_to_linear(rgb_normalized, eps)

        # Convert linear RGB to XYZ color space
        xyz = self._linear_rgb_to_xyz(rgb_linear)

        # Normalize by white point
        xyz_normalized = xyz / self.white_point.view(1, 3, 1, 1)

        # Convert XYZ to LAB
        lab = self._xyz_to_lab(xyz_normalized, eps)

        return lab

    def lab_to_rgb(self, lab: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert LAB image back to RGB color space.

        Args:
            lab: LAB image tensor with L* in [0, 100], a* and b* in [-128, 127]
            eps: Small epsilon value to stabilize operations

        Returns:
            RGB image tensor in range [0, 255]
        """
        # Convert LAB to XYZ
        xyz_normalized = self._lab_to_xyz(lab, eps)

        # Denormalize by white point
        xyz = xyz_normalized * self.white_point.view(1, 3, 1, 1)

        # Convert XYZ to linear RGB
        rgb_linear = self._xyz_to_linear_rgb(xyz)

        # Apply gamma correction to get sRGB
        rgb_normalized = self._linear_to_srgb(rgb_linear, eps)

        # Scale back to [0, 255] and clamp
        rgb = torch.clamp(rgb_normalized * 255.0, 0.0, 255.0)

        return rgb

    def calculate_delta_e(self, lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
        """Calculate Delta E (CIE76) color difference.

        Args:
            lab1: First LAB image tensor
            lab2: Second LAB image tensor

        Returns:
            Delta E difference tensor
        """
        diff = lab1 - lab2
        delta_e = torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True))
        return delta_e

    def enhance_saturation_lab(self, lab: torch.Tensor, factor: float) -> torch.Tensor:
        """Enhance saturation in LAB color space.

        Args:
            lab: LAB image tensor
            factor: Saturation enhancement factor (0.0 = no change, 1.0 = double saturation)

        Returns:
            Saturation-enhanced LAB image
        """
        enhanced_lab = lab.clone()

        # Extract L, a, b channels
        l_channel = enhanced_lab[:, 0:1, :, :]  # L* channel (lightness)
        a_channel = enhanced_lab[:, 1:2, :, :]  # a* channel (green-red)
        b_channel = enhanced_lab[:, 2:3, :, :]  # b* channel (blue-yellow)

        # Enhance chroma (a* and b* channels) while preserving lightness
        enhanced_a = a_channel * (1.0 + factor)
        enhanced_b = b_channel * (1.0 + factor)

        # Clamp to reasonable LAB ranges
        enhanced_a = torch.clamp(enhanced_a, -128, 127)
        enhanced_b = torch.clamp(enhanced_b, -128, 127)

        # Reconstruct enhanced LAB image
        enhanced_lab[:, 0:1, :, :] = l_channel
        enhanced_lab[:, 1:2, :, :] = enhanced_a
        enhanced_lab[:, 2:3, :, :] = enhanced_b

        return enhanced_lab

    def _srgb_to_linear(self, srgb: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply inverse gamma correction to convert sRGB to linear RGB."""
        threshold = 0.04045
        linear = torch.where(
            srgb <= threshold,
            srgb / 12.92,
            torch.pow(torch.clamp((srgb + 0.055) / 1.055, min=eps), 2.4),
        )
        return linear

    def _linear_to_srgb(self, linear: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply gamma correction to convert linear RGB to sRGB."""
        threshold = 0.0031308
        srgb = torch.where(
            linear <= threshold,
            linear * 12.92,
            1.055 * torch.pow(torch.clamp(linear, min=eps), 1.0 / 2.4) - 0.055,
        )
        return srgb

    def _linear_rgb_to_xyz(self, rgb_linear: torch.Tensor) -> torch.Tensor:
        """Convert linear RGB to XYZ color space."""
        # Ensure tensor is contiguous and reshape for matrix multiplication
        original_shape = rgb_linear.shape
        rgb_reshaped = rgb_linear.contiguous().view(-1, 3)

        # Apply transformation matrix
        xyz_reshaped = torch.matmul(rgb_reshaped, self.rgb_to_xyz_matrix.t())

        # Reshape back to original dimensions
        xyz = xyz_reshaped.view(original_shape)

        return xyz

    def _xyz_to_linear_rgb(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to linear RGB color space."""
        # Inverse of RGB to XYZ matrix
        xyz_to_rgb_matrix = torch.inverse(self.rgb_to_xyz_matrix)

        # Reshape for matrix multiplication
        original_shape = xyz.shape
        xyz_reshaped = xyz.contiguous().view(-1, 3)

        # Apply inverse transformation matrix
        rgb_reshaped = torch.matmul(xyz_reshaped, xyz_to_rgb_matrix.t())

        # Reshape back to original dimensions
        rgb_linear = rgb_reshaped.view(original_shape)

        return rgb_linear

    def _xyz_to_lab(self, xyz_normalized: torch.Tensor, eps: float) -> torch.Tensor:
        """Convert normalized XYZ to LAB color space."""
        # Define the piecewise function f(t) used in LAB conversion
        epsilon = 0.008856  # (6/29)^3
        kappa = 903.3  # 29^3/3^3

        def f(t: torch.Tensor) -> torch.Tensor:
            return torch.where(
                t > epsilon,
                torch.pow(torch.clamp(t, min=eps), 1.0 / 3.0),
                (kappa * t + 16) / 116,
            )

        f_xyz = f(xyz_normalized)

        # Extract X, Y, Z components
        fX = f_xyz[:, 0:1, :, :]
        fY = f_xyz[:, 1:2, :, :]
        fZ = f_xyz[:, 2:3, :, :]

        # Compute L*, a*, b* values
        L = 116 * fY - 16
        a = 500 * (fX - fY)
        b = 200 * (fY - fZ)

        # Stack channels to form LAB image
        lab = torch.cat([L, a, b], dim=1)

        return lab

    def _lab_to_xyz(self, lab: torch.Tensor, eps: float) -> torch.Tensor:
        """Convert LAB to normalized XYZ color space."""
        # Extract L*, a*, b* channels
        L = lab[:, 0:1, :, :]
        a = lab[:, 1:2, :, :]
        b = lab[:, 2:3, :, :]

        # Compute intermediate values
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        # Define the inverse piecewise function
        epsilon = 0.008856  # (6/29)^3
        kappa = 903.3  # 29^3/3^3

        def finv(t: torch.Tensor) -> torch.Tensor:
            t_cubed = t**3
            return torch.where(t_cubed > epsilon, t_cubed, (116 * t - 16) / kappa)

        # Convert back to XYZ
        X = finv(fx)
        Y = finv(fy)
        Z = finv(fz)

        # Stack to form XYZ image
        xyz_normalized = torch.cat([X, Y, Z], dim=1)

        return xyz_normalized

    def get_perceptual_weights(self, lab_image: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual importance weights for LAB image.

        Args:
            lab_image: LAB image tensor

        Returns:
            Perceptual weight tensor emphasizing important regions
        """
        eps = 1e-6

        # Extract lightness and chroma
        L = lab_image[:, 0:1, :, :]
        a = lab_image[:, 1:2, :, :]
        b = lab_image[:, 2:3, :, :]

        # Calculate chroma (color saturation)
        chroma = torch.sqrt(a**2 + b**2)

        # Weight based on chroma and lightness variation
        # Higher weights for more saturated and variable regions
        lightness_var = torch.var(L, dim=[2, 3], keepdim=True).expand_as(L)
        chroma_normalized = chroma / (torch.max(chroma) + eps)

        weights = 0.5 * chroma_normalized + 0.3 * lightness_var + 0.2

        return weights


# Global color converter instance for simple function access
_global_color_converter = None


def get_color_converter(device: str = "cpu") -> ColorConverter:
    """Get global color converter instance."""
    global _global_color_converter
    if _global_color_converter is None:
        _global_color_converter = ColorConverter(device)
    return _global_color_converter


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to LAB color space.

    Args:
        rgb: RGB tensor with values in [0, 1] or [0, 255]

    Returns:
        LAB tensor with L* in [0, 100], a* and b* in [-128, 127]
    """
    converter = get_color_converter(device=str(rgb.device))

    # Handle different input formats
    if rgb.max() <= 1.0:
        # Input is in [0, 1], scale to [0, 255]
        rgb_scaled = rgb * 255.0
    else:
        rgb_scaled = rgb

    # Ensure correct tensor shape for converter
    if rgb.dim() == 1:
        # Single color vector (3,) -> (1, 3, 1, 1)
        rgb_scaled = rgb_scaled.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        result = converter.rgb_to_lab(rgb_scaled)
        return result.squeeze(0).squeeze(-1).squeeze(-1)
    elif rgb.dim() == 2:
        # Multiple colors (N, 3) -> (1, 3, N, 1)
        rgb_scaled = rgb_scaled.t().unsqueeze(0).unsqueeze(-1)
        result = converter.rgb_to_lab(rgb_scaled)
        return result.squeeze(0).squeeze(-1).t()
    else:
        # Image tensor, use directly
        return converter.rgb_to_lab(rgb_scaled)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert LAB tensor to RGB color space.

    Args:
        lab: LAB tensor with L* in [0, 100], a* and b* in [-128, 127]

    Returns:
        RGB tensor with values in [0, 1]
    """
    converter = get_color_converter(device=str(lab.device))

    # Handle different input formats
    if lab.dim() == 1:
        # Single color vector (3,) -> (1, 3, 1, 1)
        lab_expanded = lab.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        result = converter.lab_to_rgb(lab_expanded)
        rgb_result = result.squeeze(0).squeeze(-1).squeeze(-1)
    elif lab.dim() == 2:
        # Multiple colors (N, 3) -> (1, 3, N, 1)
        lab_expanded = lab.t().unsqueeze(0).unsqueeze(-1)
        result = converter.lab_to_rgb(lab_expanded)
        rgb_result = result.squeeze(0).squeeze(-1).t()
    else:
        # Image tensor, use directly
        rgb_result = converter.lab_to_rgb(lab)

    # Convert from [0, 255] to [0, 1]
    return rgb_result / 255.0


class ColorMatcher:
    """Advanced color matching using perceptual color spaces."""

    def __init__(self, device: str = "cpu"):
        """Initialize color matcher.

        Args:
            device: Device for tensor operations
        """
        self.device = torch.device(device)
        self.color_converter = ColorConverter(device)

    def match_materials_lab(
        self, target_lab: torch.Tensor, material_colors_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match target colors to available materials using LAB color space.

        Args:
            target_lab: Target LAB colors (B, 3, H, W)
            material_colors_rgb: Available material colors in RGB (N, 3)

        Returns:
            Tuple of (material_indices, color_distances)
        """
        # Convert material colors to LAB
        material_colors_expanded = material_colors_rgb.unsqueeze(-1).unsqueeze(-1)
        material_colors_lab = (
            self.color_converter.rgb_to_lab(material_colors_expanded.unsqueeze(0))
            .squeeze(0)
            .squeeze(-1)
            .squeeze(-1)
        )  # (N, 3)

        # Reshape target for distance computation
        target_shape = target_lab.shape
        target_reshaped = target_lab.view(target_shape[0], target_shape[1], -1)
        target_reshaped = target_reshaped.permute(0, 2, 1)  # (B, H*W, 3)

        # Compute distances for each pixel to each material
        distances = torch.cdist(
            target_reshaped.view(-1, 3), material_colors_lab  # (B*H*W, 3)  # (N, 3)
        )  # (B*H*W, N)

        # Find closest material for each pixel
        material_indices = torch.argmin(distances, dim=1)
        min_distances = torch.min(distances, dim=1)[0]

        # Reshape back to image dimensions
        material_indices = material_indices.view(
            target_shape[0], target_shape[2], target_shape[3]
        )
        min_distances = min_distances.view(
            target_shape[0], target_shape[2], target_shape[3]
        )

        return material_indices, min_distances

    def calculate_color_accuracy(
        self, target_lab: torch.Tensor, matched_lab: torch.Tensor
    ) -> dict:
        """Calculate color matching accuracy metrics.

        Args:
            target_lab: Target LAB colors
            matched_lab: Matched LAB colors

        Returns:
            Dictionary of accuracy metrics
        """
        # Calculate Delta E differences
        delta_e = self.color_converter.calculate_delta_e(target_lab, matched_lab)

        # Calculate various accuracy metrics
        mean_delta_e = torch.mean(delta_e)
        max_delta_e = torch.max(delta_e)
        std_delta_e = torch.std(delta_e)

        # Percentage of pixels with "good" color match (Delta E < 2.3)
        good_match_threshold = 2.3
        good_matches = (delta_e < good_match_threshold).float()
        good_match_percentage = torch.mean(good_matches) * 100

        # Percentage with "acceptable" match (Delta E < 5.0)
        acceptable_match_threshold = 5.0
        acceptable_matches = (delta_e < acceptable_match_threshold).float()
        acceptable_match_percentage = torch.mean(acceptable_matches) * 100

        return {
            "mean_delta_e": mean_delta_e.item(),
            "max_delta_e": max_delta_e.item(),
            "std_delta_e": std_delta_e.item(),
            "good_match_percentage": good_match_percentage.item(),
            "acceptable_match_percentage": acceptable_match_percentage.item(),
            "total_pixels": delta_e.numel(),
        }
