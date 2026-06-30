"""Material selection optimization algorithms."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .database import MaterialDatabase


class MaterialOptimizer:
    """Optimize material selection for specific objectives."""

    def __init__(self, material_database: MaterialDatabase, device: str = "cpu"):
        """Initialize material optimizer.

        Args:
            material_database: Database of available materials
            device: Device for computations
        """
        self.material_db = material_database
        self.device = torch.device(device)

    def optimize_for_cost(
        self,
        target_colors: torch.Tensor,
        max_cost_per_kg: float = 50.0,
        max_materials: int = 8,
    ) -> Tuple[List[str], torch.Tensor, float]:
        """Optimize material selection for minimum cost.

        Args:
            target_colors: Target colors to match (num_colors, 3)
            max_cost_per_kg: Maximum cost per kilogram
            max_materials: Maximum number of materials

        Returns:
            Tuple of (material_ids, colors, total_cost_estimate)
        """
        # Filter materials by cost
        affordable_materials = [
            mat
            for mat in self.material_db
            if mat.available and mat.cost_per_kg <= max_cost_per_kg
        ]

        if not affordable_materials:
            raise ValueError(f"No materials available under ${max_cost_per_kg}/kg")

        # Get colors and costs
        material_colors = torch.tensor([mat.color_rgb for mat in affordable_materials])
        material_costs = torch.tensor([mat.cost_per_kg for mat in affordable_materials])
        material_ids = [mat.id for mat in affordable_materials]

        # Select materials that minimize cost while covering colors
        selected_indices = self._cost_aware_selection(
            target_colors, material_colors, material_costs, max_materials
        )

        # Extract results
        selected_materials = [material_ids[i] for i in selected_indices]
        selected_colors = material_colors[selected_indices]
        estimated_cost = material_costs[selected_indices].mean().item()

        return selected_materials, selected_colors, estimated_cost

    def optimize_for_coverage(
        self,
        target_colors: torch.Tensor,
        max_materials: int = 8,
        coverage_weight: float = 1.0,
    ) -> Tuple[List[str], torch.Tensor, float]:
        """Optimize material selection for maximum color space coverage.

        Args:
            target_colors: Target colors (num_colors, 3)
            max_materials: Maximum number of materials
            coverage_weight: Weight for coverage vs accuracy

        Returns:
            Tuple of (material_ids, colors, coverage_score)
        """
        available_materials = [mat for mat in self.material_db if mat.available]
        material_colors = torch.tensor([mat.color_rgb for mat in available_materials])
        material_ids = [mat.id for mat in available_materials]

        # Use k-means++ style selection for maximum coverage
        selected_indices = self._coverage_maximizing_selection(
            material_colors, max_materials
        )

        # Calculate coverage score
        selected_colors = material_colors[selected_indices]
        coverage_score = self._calculate_color_space_coverage(selected_colors)

        selected_materials = [material_ids[i] for i in selected_indices]

        return selected_materials, selected_colors, coverage_score

    def optimize_for_brand_consistency(
        self,
        target_colors: torch.Tensor,
        preferred_brands: List[str],
        max_materials: int = 8,
    ) -> Tuple[List[str], torch.Tensor, float]:
        """Optimize material selection within preferred brands.

        Args:
            target_colors: Target colors (num_colors, 3)
            preferred_brands: List of preferred brand names
            max_materials: Maximum number of materials

        Returns:
            Tuple of (material_ids, colors, brand_consistency_score)
        """
        # Filter by preferred brands
        brand_materials = []
        for brand in preferred_brands:
            brand_materials.extend(self.material_db.get_materials_by_brand(brand))

        # Filter available materials
        available_materials = [mat for mat in brand_materials if mat.available]

        if not available_materials:
            raise ValueError(f"No available materials from brands: {preferred_brands}")

        material_colors = torch.tensor([mat.color_rgb for mat in available_materials])
        material_ids = [mat.id for mat in available_materials]

        # Select best matching materials
        selected_indices = self._color_matching_selection(
            target_colors, material_colors, max_materials
        )

        # Calculate brand consistency score
        selected_materials = [material_ids[i] for i in selected_indices]
        selected_brands = {available_materials[i].brand for i in selected_indices}
        brand_consistency = 1.0 / len(selected_brands)  # Higher = more consistent

        selected_colors = material_colors[selected_indices]

        return selected_materials, selected_colors, brand_consistency

    def optimize_for_print_complexity(
        self,
        target_colors: torch.Tensor,
        max_materials: int = 8,
        complexity_penalty: float = 0.1,
    ) -> Tuple[List[str], torch.Tensor, float]:
        """Optimize material selection to minimize print complexity.

        Args:
            target_colors: Target colors (num_colors, 3)
            max_materials: Maximum number of materials
            complexity_penalty: Penalty for each additional material

        Returns:
            Tuple of (material_ids, colors, complexity_score)
        """
        available_materials = [mat for mat in self.material_db if mat.available]
        material_colors = torch.tensor([mat.color_rgb for mat in available_materials])
        material_ids = [mat.id for mat in available_materials]

        best_score = -1
        best_selection = None

        # Try different numbers of materials (fewer = less complex)
        for num_materials in range(2, max_materials + 1):
            selected_indices = self._color_matching_selection(
                target_colors, material_colors, num_materials
            )

            # Calculate quality score
            selected_colors = material_colors[selected_indices]
            color_quality = self._calculate_color_matching_quality(
                target_colors, selected_colors
            )

            # Apply complexity penalty
            complexity_score = color_quality - (num_materials * complexity_penalty)

            if complexity_score > best_score:
                best_score = complexity_score
                best_selection = selected_indices

        selected_materials = [material_ids[i] for i in best_selection]
        selected_colors = material_colors[best_selection]

        return selected_materials, selected_colors, best_score

    def _cost_aware_selection(
        self,
        target_colors: torch.Tensor,
        material_colors: torch.Tensor,
        material_costs: torch.Tensor,
        max_materials: int,
    ) -> List[int]:
        """Select materials considering both color match and cost."""
        # Calculate color distances
        distances = torch.cdist(target_colors, material_colors)
        min_distances = torch.min(distances, dim=0)[0]

        # Normalize costs to [0, 1]
        normalized_costs = (material_costs - material_costs.min()) / (
            material_costs.max() - material_costs.min() + 1e-6
        )

        # Combined score (lower is better)
        combined_scores = min_distances + 0.3 * normalized_costs

        # Select materials with best combined scores
        selected_indices = torch.argsort(combined_scores)[:max_materials].tolist()

        return selected_indices

    def _coverage_maximizing_selection(
        self, material_colors: torch.Tensor, max_materials: int
    ) -> List[int]:
        """Select materials to maximize color space coverage using k-means++ approach."""
        if len(material_colors) <= max_materials:
            return list(range(len(material_colors)))

        selected_indices = []

        # Start with random material
        import random

        selected_indices.append(random.randint(0, len(material_colors) - 1))

        # Select remaining materials to maximize distance
        for _ in range(max_materials - 1):
            distances = []
            selected_colors = material_colors[selected_indices]

            for i, color in enumerate(material_colors):
                if i in selected_indices:
                    distances.append(0)
                else:
                    # Distance to closest selected color
                    min_dist = torch.min(torch.norm(selected_colors - color, dim=1))
                    distances.append(min_dist.item())

            # Select material with maximum distance to selected colors
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)

        return selected_indices

    def _color_matching_selection(
        self,
        target_colors: torch.Tensor,
        material_colors: torch.Tensor,
        max_materials: int,
    ) -> List[int]:
        """Select materials that best match target colors."""
        if len(target_colors) <= max_materials:
            # Use Hungarian algorithm for optimal assignment
            cost_matrix = torch.cdist(target_colors, material_colors).cpu().numpy()
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            return col_indices[:max_materials].tolist()
        else:
            # More target colors than available materials
            # Select materials that minimize total distance
            distances = torch.cdist(target_colors, material_colors)
            min_distances = torch.min(distances, dim=0)[0]
            selected_indices = torch.argsort(min_distances)[:max_materials].tolist()
            return selected_indices

    def _calculate_color_space_coverage(self, colors: torch.Tensor) -> float:
        """Calculate how well colors cover the RGB color space."""
        if len(colors) < 2:
            return 0.0

        # Calculate volume of convex hull in RGB space (simplified)
        color_ranges = colors.max(dim=0)[0] - colors.min(dim=0)[0]
        coverage = torch.prod(color_ranges).item()

        # Normalize by maximum possible coverage
        max_coverage = 1.0  # Full RGB cube

        return min(coverage / max_coverage, 1.0)

    def _calculate_color_matching_quality(
        self, target_colors: torch.Tensor, selected_colors: torch.Tensor
    ) -> float:
        """Calculate quality of color matching."""
        # Find closest selected color for each target
        distances = torch.cdist(target_colors, selected_colors)
        min_distances = torch.min(distances, dim=1)[0]

        # Calculate average matching quality (lower distance = higher quality)
        avg_distance = torch.mean(min_distances).item()
        quality_score = 1.0 / (1.0 + avg_distance)

        return quality_score

    def analyze_material_compatibility(self, material_ids: List[str]) -> Dict[str, any]:
        """Analyze compatibility between selected materials.

        Args:
            material_ids: List of material IDs to analyze

        Returns:
            Dictionary of compatibility metrics
        """
        materials = [self.material_db.get_material(mid) for mid in material_ids]
        materials = [mat for mat in materials if mat is not None]

        if len(materials) < 2:
            return {"compatible": True, "issues": []}

        compatibility = {
            "compatible": True,
            "issues": [],
            "temperature_range": None,
            "brand_consistency": None,
            "cost_variance": None,
        }

        # Temperature compatibility
        temperatures = [mat.temperature for mat in materials]
        temp_range = max(temperatures) - min(temperatures)
        compatibility["temperature_range"] = temp_range

        if temp_range > 50:  # More than 50°C difference
            compatibility["compatible"] = False
            compatibility["issues"].append(
                f"Large temperature range: {temp_range}°C difference"
            )

        # Brand consistency
        brands = {mat.brand for mat in materials}
        compatibility["brand_consistency"] = len(brands) == 1

        if len(brands) > 2:
            compatibility["issues"].append(
                f"Many different brands: {', '.join(brands)}"
            )

        # Cost variance
        costs = [mat.cost_per_kg for mat in materials]
        cost_variance = np.var(costs)
        compatibility["cost_variance"] = cost_variance

        if cost_variance > 100:  # High cost variance
            compatibility["issues"].append(f"High cost variance: ${cost_variance:.2f}")

        return compatibility

    def suggest_material_substitutions(
        self, material_id: str, num_alternatives: int = 3
    ) -> List[Tuple[str, float]]:
        """Suggest alternative materials similar to the given one.

        Args:
            material_id: ID of material to find alternatives for
            num_alternatives: Number of alternatives to suggest

        Returns:
            List of (material_id, similarity_score) tuples
        """
        target_material = self.material_db.get_material(material_id)
        if not target_material:
            return []

        target_color = torch.tensor(target_material.color_rgb)

        alternatives = []
        for material in self.material_db:
            if material.id == material_id or not material.available:
                continue

            # Calculate similarity score
            color_distance = torch.norm(
                torch.tensor(material.color_rgb) - target_color
            ).item()

            # Consider other factors
            temp_diff = abs(material.temperature - target_material.temperature) / 100.0
            cost_diff = abs(material.cost_per_kg - target_material.cost_per_kg) / 50.0
            brand_bonus = 0.1 if material.brand == target_material.brand else 0.0

            # Combined similarity (lower is more similar)
            similarity = (
                color_distance + 0.2 * temp_diff + 0.1 * cost_diff - brand_bonus
            )
            alternatives.append((material.id, similarity))

        # Sort by similarity and return top alternatives
        alternatives.sort(key=lambda x: x[1])
        return alternatives[:num_alternatives]
