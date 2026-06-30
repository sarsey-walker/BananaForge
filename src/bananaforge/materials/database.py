"""Material database management for BananaForge."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


@dataclass
class Material:
    """Represents a 3D printing material with properties."""

    id: str
    name: str
    brand: str
    color_rgb: Tuple[float, float, float]  # RGB values [0-1]
    color_hex: str
    transparency: float = 0.0  # 0=opaque, 1=transparent
    transmission_distance: float = 4.0  # TD value for layer thickness
    density: float = 1.25  # Material density (g/cm³)
    temperature: int = 200  # Printing temperature (°C)
    cost_per_kg: float = 25.0  # Cost per kilogram
    available: bool = True
    tags: Optional[List[str]] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.tags is None:
            self.tags = []

        # Ensure color_rgb is normalized to [0-1]
        if any(c > 1.0 for c in self.color_rgb):
            self.color_rgb = tuple(c / 255.0 for c in self.color_rgb)

    @property
    def color_tensor(self) -> torch.Tensor:
        """Get color as PyTorch tensor."""
        return torch.tensor(self.color_rgb, dtype=torch.float32)

    @property
    def is_transparent(self) -> bool:
        """Check if material is transparent."""
        return self.transparency > 0.1

    @property
    def is_opaque(self) -> bool:
        """Check if material is effectively opaque."""
        return self.transparency < 0.1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Material":
        """Create Material from dictionary."""
        return cls(**data)


class MaterialDatabase:
    """Database for managing 3D printing materials."""

    def __init__(self):
        """Initialize material database."""
        self.materials: Dict[str, Material] = {}
        self._color_cache: Optional[torch.Tensor] = None
        self._id_cache: Optional[List[str]] = None

    def add_material(self, material: Material) -> None:
        """Add material to database.

        Args:
            material: Material to add
        """
        self.materials[material.id] = material
        self._invalidate_cache()

    def remove_material(self, material_id: str) -> bool:
        """Remove material from database.

        Args:
            material_id: ID of material to remove

        Returns:
            True if material was removed, False if not found
        """
        if material_id in self.materials:
            del self.materials[material_id]
            self._invalidate_cache()
            return True
        return False

    def get_material(self, material_id: str) -> Optional[Material]:
        """Get material by ID.

        Args:
            material_id: Material ID

        Returns:
            Material if found, None otherwise
        """
        return self.materials.get(material_id)

    def get_materials_by_brand(self, brand: str) -> List[Material]:
        """Get all materials from specific brand.

        Args:
            brand: Brand name

        Returns:
            List of materials from brand
        """
        return [mat for mat in self.materials.values() if mat.brand == brand]

    def get_materials_by_color_range(
        self,
        center_color: Tuple[float, float, float],
        max_distance: float = 0.3,
        color_space: str = "rgb",
    ) -> List[Material]:
        """Get materials within color range.

        Args:
            center_color: Center color (RGB values 0-1)
            max_distance: Maximum color distance
            color_space: Color space for distance calculation

        Returns:
            List of materials within color range
        """
        materials = []
        center_tensor = torch.tensor(center_color, dtype=torch.float32)

        for material in self.materials.values():
            if not material.available:
                continue

            if color_space == "rgb":
                distance = torch.norm(material.color_tensor - center_tensor).item()
            elif color_space == "lab":
                # Convert to LAB for perceptual distance
                center_lab = self._rgb_to_lab(center_tensor)
                material_lab = self._rgb_to_lab(material.color_tensor)
                distance = torch.norm(material_lab - center_lab).item()
            else:
                raise ValueError(f"Unknown color space: {color_space}")

            if distance <= max_distance:
                materials.append(material)

        return materials

    def get_opaque_materials(self) -> List[Material]:
        """Get all opaque materials."""
        return [
            mat for mat in self.materials.values() if mat.is_opaque and mat.available
        ]

    def get_transparent_materials(self) -> List[Material]:
        """Get all transparent materials."""
        return [
            mat
            for mat in self.materials.values()
            if mat.is_transparent and mat.available
        ]

    def get_color_palette(self, device: str = "cpu") -> torch.Tensor:
        """Get color palette as tensor.

        Args:
            device: Device for tensor

        Returns:
            Color palette tensor (num_materials, 3)
        """
        if self._color_cache is None:
            available_materials = [
                mat for mat in self.materials.values() if mat.available
            ]
            colors = [mat.color_rgb for mat in available_materials]
            self._color_cache = torch.tensor(colors, dtype=torch.float32)
            self._id_cache = [mat.id for mat in available_materials]

        return self._color_cache.to(device)

    def get_material_ids(self) -> List[str]:
        """Get list of available material IDs."""
        if self._id_cache is None:
            self.get_color_palette()  # This will populate the cache
        return self._id_cache.copy()

    def load_from_csv(self, csv_path: Union[str, Path]) -> None:
        """Load materials from CSV file.

        Args:
            csv_path: Path to CSV file

        Expected CSV columns:
        - id, name, brand, color_hex, transparency, td, density, temperature, cost
        """
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            # Parse hex color
            hex_color = row["color_hex"].lstrip("#")
            rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

            material = Material(
                id=row.get("id", f"{row['brand']}_{row['name']}"),
                name=row["name"],
                brand=row["brand"],
                color_rgb=rgb,
                color_hex=row["color_hex"],
                transparency=row.get("transparency", 0.0),
                transmission_distance=row.get("td", 4.0),
                density=row.get("density", 1.25),
                temperature=row.get("temperature", 200),
                cost_per_kg=row.get("cost", 25.0),
                available=row.get("available", True),
            )

            self.add_material(material)

    def load_from_json(self, json_path: Union[str, Path]) -> None:
        """Load materials from JSON file.

        Args:
            json_path: Path to JSON file
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            materials_data = data
        elif "materials" in data:
            materials_data = data["materials"]
        elif "filaments" in data:
            materials_data = data["filaments"]
        else:
            materials_data = data.values()

        for mat_data in materials_data:
            # Convert hex to RGB if needed
            if "color_hex" in mat_data and "color_rgb" not in mat_data:
                hex_color = mat_data["color_hex"].lstrip("#")
                mat_data["color_rgb"] = tuple(
                    int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)
                )

            material = Material.from_dict(mat_data)
            self.add_material(material)

    def save_to_csv(self, csv_path: Union[str, Path]) -> None:
        """Save materials to CSV file.

        Args:
            csv_path: Output CSV path
        """
        data = []
        for material in self.materials.values():
            mat_dict = material.to_dict()
            # Convert RGB tuple to separate columns if needed
            mat_dict["color_r"] = mat_dict["color_rgb"][0]
            mat_dict["color_g"] = mat_dict["color_rgb"][1]
            mat_dict["color_b"] = mat_dict["color_rgb"][2]
            data.append(mat_dict)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    def save_to_json(self, json_path: Union[str, Path]) -> None:
        """Save materials to JSON file.

        Args:
            json_path: Output JSON path
        """
        data = {"materials": [mat.to_dict() for mat in self.materials.values()]}

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_subset(
        self,
        brands: Optional[List[str]] = None,
        max_materials: Optional[int] = None,
        color_diversity: bool = True,
    ) -> "MaterialDatabase":
        """Create subset of materials.

        Args:
            brands: Specific brands to include
            max_materials: Maximum number of materials
            color_diversity: Whether to optimize for color diversity

        Returns:
            New MaterialDatabase with subset
        """
        subset_db = MaterialDatabase()

        # Filter by brand if specified
        candidates = list(self.materials.values())
        if brands:
            candidates = [mat for mat in candidates if mat.brand in brands]

        # Filter available materials
        candidates = [mat for mat in candidates if mat.available]

        # Select materials
        if max_materials and len(candidates) > max_materials:
            if color_diversity:
                # Use color-based selection for diversity
                selected = self._select_diverse_colors(candidates, max_materials)
            else:
                # Random selection
                import random

                selected = random.sample(candidates, max_materials)
        else:
            selected = candidates

        # Add to subset database
        for material in selected:
            subset_db.add_material(material)

        return subset_db

    def _select_diverse_colors(
        self, materials: List[Material], num_select: int
    ) -> List[Material]:
        """Select materials with diverse colors."""
        if len(materials) <= num_select:
            return materials

        # Extract colors
        colors = np.array([mat.color_rgb for mat in materials])

        # Use k-means++ style selection for diversity
        selected_indices = []

        # Start with random material
        import random

        selected_indices.append(random.randint(0, len(materials) - 1))

        for _ in range(num_select - 1):
            # Calculate distances to already selected colors
            selected_colors = colors[selected_indices]

            min_distances = []
            for i, color in enumerate(colors):
                if i in selected_indices:
                    min_distances.append(0)
                else:
                    distances = [
                        np.linalg.norm(color - sel_color)
                        for sel_color in selected_colors
                    ]
                    min_distances.append(min(distances))

            # Select material with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)

        return [materials[i] for i in selected_indices]

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space (simplified)."""
        r, g, b = rgb[0], rgb[1], rgb[2]

        # Simplified RGB to LAB conversion
        lightness = 0.299 * r + 0.587 * g + 0.114 * b
        a = 0.5 * (r - g)
        b_comp = 0.5 * (g + r - 2 * b)

        return torch.tensor([lightness, a, b_comp])

    def _invalidate_cache(self) -> None:
        """Invalidate color and ID caches."""
        self._color_cache = None
        self._id_cache = None

    def get_all_materials(self) -> List[Material]:
        """Get all materials in the database.

        Returns:
            List of all materials
        """
        return list(self.materials.values())

    def __len__(self) -> int:
        """Get number of materials."""
        return len(self.materials)

    def __iter__(self):
        """Iterate over materials."""
        return iter(self.materials.values())


class DefaultMaterials:
    """Default material sets for common use cases."""

    @staticmethod
    def create_bambu_basic_pla() -> MaterialDatabase:
        """Create database with Bambu Lab Basic PLA materials."""
        db = MaterialDatabase()

        # Bambu Lab Basic PLA colors (based on common available colors)
        bambu_colors = [
            ("White", "#FFFFFF"),
            ("Black", "#000000"),
            ("Red", "#FF0000"),
            ("Blue", "#0000FF"),
            ("Green", "#008000"),
            ("Yellow", "#FFFF00"),
            ("Orange", "#FFA500"),
            ("Purple", "#800080"),
            ("Pink", "#FFC0CB"),
            ("Brown", "#A52A2A"),
            ("Gray", "#808080"),
            ("Light Blue", "#ADD8E6"),
            ("Light Green", "#90EE90"),
            ("Transparent", "#FFFFFF"),
        ]

        for i, (name, hex_color) in enumerate(bambu_colors):
            transparency = 0.8 if name == "Transparent" else 0.0

            material = Material(
                id=f"bambu_pla_{name.lower().replace(' ', '_')}",
                name=f"Basic PLA {name}",
                brand="Bambu Lab",
                color_rgb=tuple(
                    int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)
                ),
                color_hex=hex_color,
                transparency=transparency,
                transmission_distance=4.0,
                density=1.24,
                temperature=220,
                cost_per_kg=29.99,
            )

            db.add_material(material)

        return db

    @staticmethod
    def create_hueforge_set() -> MaterialDatabase:
        """Create database compatible with HueForge materials."""
        db = MaterialDatabase()

        # HueForge compatible colors
        hueforge_colors = [
            ("Natural", "#F5F5DC"),
            ("White", "#FFFFFF"),
            ("Black", "#000000"),
            ("Red", "#DC143C"),
            ("Blue", "#4169E1"),
            ("Green", "#228B22"),
            ("Yellow", "#FFD700"),
            ("Orange", "#FF8C00"),
            ("Purple", "#9932CC"),
            ("Brown", "#8B4513"),
            ("Gray", "#696969"),
            ("Pink", "#FF69B4"),
        ]

        for i, (name, hex_color) in enumerate(hueforge_colors):
            material = Material(
                id=f"hueforge_{name.lower()}",
                name=f"HueForge {name}",
                brand="Generic",
                color_rgb=tuple(
                    int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)
                ),
                color_hex=hex_color,
                transparency=0.0,
                transmission_distance=4.0,
                density=1.25,
                temperature=210,
                cost_per_kg=25.00,
            )

            db.add_material(material)

        return db

    @staticmethod
    def create_rainbow_set(num_colors: int = 12) -> MaterialDatabase:
        """Create rainbow color set for testing.

        Args:
            num_colors: Number of colors in rainbow

        Returns:
            MaterialDatabase with rainbow colors
        """
        db = MaterialDatabase()

        for i in range(num_colors):
            # Generate HSV color
            hue = i / num_colors
            saturation = 0.9
            value = 0.8

            # Convert HSV to RGB
            import colorsys

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )

            material = Material(
                id=f"rainbow_{i:02d}",
                name=f"Rainbow {i+1}",
                brand="Test",
                color_rgb=rgb,
                color_hex=hex_color,
                transparency=0.0,
                transmission_distance=4.0,
                density=1.25,
                temperature=210,
                cost_per_kg=25.00,
            )

            db.add_material(material)

        return db
