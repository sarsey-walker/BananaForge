"""Material manager for coordinating material operations."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .database import DefaultMaterials, Material, MaterialDatabase
from .matcher import ColorMatcher


class MaterialManager:
    """High-level interface for managing materials and color matching."""

    def __init__(self, device: str = "cpu", enable_transparency: bool = False):
        """Initialize material manager.

        Args:
            device: Device for computations
            enable_transparency: Enable transparency-based color mixing
        """
        self.device = device
        self.enable_transparency = enable_transparency
        self.database = MaterialDatabase()
        self.matcher = None

    def load_materials_from_file(self, file_path: str) -> None:
        """Load materials from CSV or JSON file.

        Args:
            file_path: Path to material database file
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            self.database.load_from_csv(file_path)
        elif file_path.suffix.lower() == ".json":
            self.database.load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Update color matcher
        self.matcher = ColorMatcher(
            self.database, self.device, enable_transparency=self.enable_transparency
        )

    def load_default_materials(self, material_set: str = "bambu_pla") -> None:
        """Load default material set.

        Args:
            material_set: Name of default set ("bambu_pla", "hueforge", "rainbow")
        """
        if material_set == "bambu_pla":
            self.database = DefaultMaterials.create_bambu_basic_pla()
        elif material_set == "hueforge":
            self.database = DefaultMaterials.create_hueforge_set()
        elif material_set == "rainbow":
            self.database = DefaultMaterials.create_rainbow_set()
        else:
            raise ValueError(f"Unknown material set: {material_set}")

        # Update color matcher
        self.matcher = ColorMatcher(
            self.database, self.device, enable_transparency=self.enable_transparency
        )

    def match_image_colors(
        self, image: torch.Tensor, max_materials: int = 8, method: str = "perceptual"
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Match image colors to available materials.

        Args:
            image: Input image tensor (1, 3, H, W)
            max_materials: Maximum number of materials to use
            method: Color matching method

        Returns:
            Tuple of (material_ids, colors, color_mapping)
        """
        if self.matcher is None:
            raise ValueError(
                "No materials loaded. Call load_materials_from_file() or load_default_materials() first."
            )

        return self.matcher.match_image_colors(image, max_materials, method)

    def optimize_material_selection(
        self,
        image: torch.Tensor,
        max_materials: int = 8,
        weight_coverage: float = 0.6,
        weight_accuracy: float = 0.4,
    ) -> Tuple[List[str], torch.Tensor, float]:
        """Optimize material selection for image.

        Args:
            image: Input image tensor
            max_materials: Maximum materials to select
            weight_coverage: Weight for color space coverage
            weight_accuracy: Weight for color accuracy

        Returns:
            Tuple of (material_ids, colors, optimization_score)
        """
        if self.matcher is None:
            raise ValueError(
                "No materials loaded. Call load_materials_from_file() or load_default_materials() first."
            )

        return self.matcher.optimize_material_selection(
            image, max_materials, weight_coverage, weight_accuracy
        )

    def get_material_info(self, material_id: str) -> Optional[Material]:
        """Get information about a specific material.

        Args:
            material_id: ID of material to retrieve

        Returns:
            Material object or None if not found
        """
        return self.database.get_material(material_id)

    def get_materials_by_brand(self, brand: str) -> List[Material]:
        """Get all materials from a specific brand.

        Args:
            brand: Brand name

        Returns:
            List of materials from the brand
        """
        return self.database.get_materials_by_brand(brand)

    def get_available_materials(self) -> List[Material]:
        """Get all available materials.

        Returns:
            List of all available materials
        """
        return [mat for mat in self.database if mat.available]

    def export_materials(
        self,
        output_path: str,
        format: str = "csv",
        brands: Optional[List[str]] = None,
        max_materials: Optional[int] = None,
    ) -> None:
        """Export materials to file.

        Args:
            output_path: Output file path
            format: Export format ("csv" or "json")
            brands: Optional brand filter
            max_materials: Optional material count limit
        """
        # Create subset if needed
        if brands or max_materials:
            export_db = self.database.create_subset(
                brands=brands, max_materials=max_materials, color_diversity=True
            )
        else:
            export_db = self.database

        # Export
        if format.lower() == "csv":
            export_db.save_to_csv(output_path)
        elif format.lower() == "json":
            export_db.save_to_json(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def analyze_color_coverage(
        self, image: torch.Tensor, max_materials: int = 8
    ) -> Dict[str, float]:
        """Analyze how well available materials cover image colors.

        Args:
            image: Input image tensor
            max_materials: Maximum materials to consider

        Returns:
            Dictionary of coverage metrics
        """
        if self.matcher is None:
            raise ValueError("No materials loaded.")

        # Get material selection
        materials, colors, mapping = self.matcher.match_image_colors(
            image, max_materials, "perceptual"
        )

        # Calculate coverage metrics
        coverage_score = self.matcher._calculate_coverage_score(image, colors, mapping)
        accuracy_score = self.matcher._calculate_accuracy_score(image, colors, mapping)

        return {
            "coverage_score": coverage_score,
            "accuracy_score": accuracy_score,
            "combined_score": 0.6 * coverage_score + 0.4 * accuracy_score,
            "num_materials": len(materials),
            "material_ids": materials,
        }

    def get_statistics(self) -> Dict[str, any]:
        """Get database statistics.

        Returns:
            Dictionary of database statistics
        """
        materials = list(self.database)
        available_materials = [mat for mat in materials if mat.available]

        # Brand statistics
        brands = {}
        for mat in materials:
            brands[mat.brand] = brands.get(mat.brand, 0) + 1

        # Color statistics (simplified)
        colors = [mat.color_rgb for mat in available_materials]
        if colors:
            colors_tensor = torch.tensor(colors)
            color_variance = torch.var(colors_tensor, dim=0).mean().item()
        else:
            color_variance = 0.0

        return {
            "total_materials": len(materials),
            "available_materials": len(available_materials),
            "brands": brands,
            "color_variance": color_variance,
            "device": self.device,
        }

    def __len__(self) -> int:
        """Get number of materials in database."""
        return len(self.database)

    def __repr__(self) -> str:
        """String representation of material manager."""
        stats = self.get_statistics()
        return (
            f"MaterialManager(materials={stats['total_materials']}, "
            f"available={stats['available_materials']}, "
            f"brands={len(stats['brands'])}, "
            f"device='{self.device}')"
        )
