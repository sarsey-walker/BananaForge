"""Generate printing instructions and project files."""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..materials.database import Material, MaterialDatabase


@dataclass
class SwapInstruction:
    """Individual material swap instruction."""

    layer_number: int
    height_mm: float
    old_material: str
    new_material: str
    description: str
    estimated_time_seconds: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PrintSettings:
    """Print settings for the model."""

    layer_height: float = 0.08
    nozzle_temperature: int = 220
    bed_temperature: int = 60
    print_speed: int = 50
    infill_percentage: int = 15
    supports: bool = False
    brim: bool = False
    estimated_print_time_minutes: int = 0
    estimated_material_usage_grams: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class SwapInstructionGenerator:
    """Generate material swap instructions for multi-color printing."""

    def __init__(self, layer_height: float = 0.08, initial_layer_height: float = 0.16):
        """Initialize instruction generator.

        Args:
            layer_height: Layer height in mm
            initial_layer_height: Initial layer height in mm (background layer)
        """
        self.layer_height = layer_height
        self.initial_layer_height = initial_layer_height

    def generate_swap_instructions(
        self,
        material_assignments: torch.Tensor,
        material_database: MaterialDatabase,
        material_ids: List[str],
        start_material: Optional[str] = None,
    ) -> List[SwapInstruction]:
        """Generate material swap instructions.

        Args:
            material_assignments: Material per layer (num_layers, H, W)
            material_database: Database of available materials
            material_ids: List of material IDs used
            start_material: Initial material (if None, uses most common in first layer)

        Returns:
            List of swap instructions
        """
        instructions = []
        num_layers = material_assignments.shape[0]

        # Determine dominant material for each layer
        layer_materials = []
        for layer_idx in range(num_layers):
            layer_assignment = material_assignments[layer_idx]

            # Find most common material in this layer
            unique_materials, counts = torch.unique(
                layer_assignment, return_counts=True
            )
            valid_materials_mask = unique_materials >= 0
            unique_materials = unique_materials[valid_materials_mask]
            counts = counts[valid_materials_mask]
            if len(unique_materials) == 0:
                fallback_material = (
                    layer_materials[-1]
                    if layer_materials
                    else material_ids[0] if material_ids else "unknown"
                )
                layer_materials.append(fallback_material)
                continue
            dominant_material_idx = unique_materials[torch.argmax(counts)].item()

            if dominant_material_idx < len(material_ids):
                dominant_material = material_ids[dominant_material_idx]
            else:
                dominant_material = material_ids[0] if material_ids else "unknown"

            layer_materials.append(dominant_material)

        # Determine starting material
        if start_material is None:
            start_material = layer_materials[0] if layer_materials else material_ids[0]

        current_material = start_material

        # Generate swap instructions
        for layer_idx, layer_material in enumerate(layer_materials):
            if layer_material != current_material:
                # Material change needed
                # Bambu Studio expects swap commands one layer earlier than where new material appears
                # So if layer_idx should have new material, the swap happens at end of layer_idx-1
                if layer_idx == 0:
                    # Can't swap before layer 0, skip this case
                    current_material = layer_material
                    continue

                swap_layer_idx = layer_idx - 1
                if swap_layer_idx == 0:
                    height_mm = self.initial_layer_height  # End of first layer
                else:
                    height_mm = self.initial_layer_height + (
                        swap_layer_idx * self.layer_height
                    )

                # Get material information
                old_mat = material_database.get_material(current_material)
                new_mat = material_database.get_material(layer_material)

                old_name = old_mat.name if old_mat else current_material
                new_name = new_mat.name if new_mat else layer_material

                # Create instruction with the actual swap layer number (one earlier)
                instruction = SwapInstruction(
                    layer_number=swap_layer_idx
                    + 1,  # +1 because layers are 1-indexed for user display
                    height_mm=height_mm,
                    old_material=current_material,
                    new_material=layer_material,
                    description=f"Change from {old_name} to {new_name}",
                    estimated_time_seconds=self._estimate_swap_time(old_mat, new_mat),
                )

                instructions.append(instruction)
                current_material = layer_material

        return instructions

    def _estimate_swap_time(
        self, old_material: Optional[Material], new_material: Optional[Material]
    ) -> int:
        """Estimate time needed for material swap."""
        base_time = 120  # 2 minutes base swap time

        # Add time for temperature changes
        if old_material and new_material:
            temp_diff = abs(old_material.temperature - new_material.temperature)
            temp_time = temp_diff * 2  # 2 seconds per degree
        else:
            temp_time = 30  # Default temp change time

        # Add time for purging (more for larger temp differences)
        purge_time = 60 + (temp_time // 5)

        return base_time + temp_time + purge_time

    def optimize_swap_sequence(
        self,
        instructions: List[SwapInstruction],
        material_database: MaterialDatabase,
        minimize_swaps: bool = True,
    ) -> List[SwapInstruction]:
        """Optimize material swap sequence.

        Args:
            instructions: Initial swap instructions
            material_database: Material database
            minimize_swaps: Whether to minimize number of swaps

        Returns:
            Optimized swap instructions
        """
        if not instructions or not minimize_swaps:
            return instructions

        # Group consecutive layers with same target material
        optimized = []
        current_instruction = None

        for instruction in instructions:
            if current_instruction is None:
                current_instruction = instruction
            elif current_instruction.new_material == instruction.old_material:
                # Can potentially merge or skip
                current_instruction.description += (
                    f" (merged with layer {instruction.layer_number})"
                )
            else:
                # Different material, save current and start new
                optimized.append(current_instruction)
                current_instruction = instruction

        if current_instruction:
            optimized.append(current_instruction)

        return optimized

    def export_instructions_to_text(
        self,
        instructions: List[SwapInstruction],
        output_path: Union[str, Path],
        include_timing: bool = True,
        material_database: Optional[MaterialDatabase] = None,
        print_settings: Optional[PrintSettings] = None,
    ) -> None:
        """Export instructions to human-readable text file.

        Args:
            instructions: Swap instructions
            output_path: Output file path
            include_timing: Whether to include timing information
            material_database: Material database for additional info
            print_settings: Print settings for configuration details
        """
        with open(output_path, "w") as f:
            # Header
            f.write("BananaForge MULTI-COLOR PRINTING INSTRUCTIONS\n")
            f.write("=" * 60 + "\n\n")

            # Print setup configuration
            f.write("PRINT SETUP CONFIGURATION\n")
            f.write("=" * 30 + "\n")

            if print_settings:
                f.write(f"Layer Height: {print_settings.layer_height:.2f}mm\n")
                f.write(f"Initial Layer Height: {self.initial_layer_height:.2f}mm\n")
                f.write(f"Nozzle Temperature: {print_settings.nozzle_temperature}°C\n")
                f.write(f"Bed Temperature: {print_settings.bed_temperature}°C\n")
                f.write(f"Print Speed: {print_settings.print_speed}mm/s\n")
                f.write(f"Infill: {print_settings.infill_percentage}%\n")
                f.write(f"Supports: {'Yes' if print_settings.supports else 'No'}\n")
                f.write(f"Brim: {'Yes' if print_settings.brim else 'No'}\n")
            else:
                f.write(f"Layer Height: {self.layer_height:.2f}mm\n")
                f.write(f"Initial Layer Height: {self.initial_layer_height:.2f}mm\n")
                f.write("Nozzle Temperature: 220°C (adjust per material)\n")
                f.write("Bed Temperature: 60°C\n")
                f.write("Print Speed: 50mm/s\n")
                f.write("Infill: 15%\n")
                f.write("Supports: No\n")
                f.write("Brim: No\n")

            f.write("\n")

            # Starting material information
            if instructions:
                starting_material = instructions[0].old_material
                f.write("STARTING MATERIAL\n")
                f.write("=" * 20 + "\n")

                if material_database:
                    start_mat = material_database.get_material(starting_material)
                    if start_mat:
                        f.write(f"Material: {start_mat.name}\n")
                        f.write(f"Color: {start_mat.color_hex} ({start_mat.name})\n")
                        f.write(f"Temperature: {start_mat.temperature}°C\n")
                        f.write(f"Brand: {start_mat.brand}\n")
                        f.write("Load this material and begin printing.\n")
                    else:
                        f.write(f"Material ID: {starting_material}\n")
                        f.write("Load this material and begin printing.\n")
                else:
                    f.write(f"Material ID: {starting_material}\n")
                    f.write("Load this material and begin printing.\n")

                f.write(
                    f"Print the base/background layer at {self.initial_layer_height:.2f}mm height.\n"
                )
                f.write(
                    f"Continue with this material until Layer {instructions[0].layer_number}.\n\n"
                )

            # Material swap instructions
            f.write("MATERIAL SWAP INSTRUCTIONS\n")
            f.write("=" * 35 + "\n\n")

            if not instructions:
                f.write("No material swaps required - single material print.\n")
                f.write("Print the entire model with the starting material.\n")
                return

            total_time = 0
            for i, instruction in enumerate(instructions, 1):
                f.write(f"SWAP #{i}\n")
                f.write(f"Layer: {instruction.layer_number}\n")
                f.write(f"Height: {instruction.height_mm:.2f}mm\n")
                f.write(f"Action: {instruction.description}\n")

                # Add material-specific temperature information
                if material_database:
                    new_mat = material_database.get_material(instruction.new_material)
                    if new_mat:
                        f.write(f"New Temperature: {new_mat.temperature}°C\n")
                        f.write(f"New Color: {new_mat.color_hex} ({new_mat.name})\n")

                if include_timing:
                    minutes = instruction.estimated_time_seconds // 60
                    seconds = instruction.estimated_time_seconds % 60
                    f.write(f"Estimated time: {minutes}m {seconds}s\n")
                    total_time += instruction.estimated_time_seconds

                f.write("-" * 40 + "\n\n")

            # Summary
            f.write("PRINTING SUMMARY\n")
            f.write("=" * 20 + "\n")
            f.write(f"Total material swaps: {len(instructions)}\n")

            if include_timing:
                total_minutes = total_time // 60
                f.write(f"Total swap time: {total_minutes} minutes\n")

            if print_settings and print_settings.estimated_print_time_minutes > 0:
                total_print_hours = print_settings.estimated_print_time_minutes / 60.0
                f.write(f"Estimated print time: {total_print_hours:.1f} hours\n")
                f.write(
                    f"Total project time: {total_print_hours + (total_time/3600):.1f} hours\n"
                )

            f.write("\n")

            # General printing tips
            f.write("PRINTING TIPS\n")
            f.write("=" * 15 + "\n")
            f.write("• Ensure the first layer adheres well before starting\n")
            f.write("• Have all materials ready and loaded in advance\n")
            f.write("• Monitor temperature changes during swaps\n")
            f.write("• Allow nozzle to heat/cool to proper temperature\n")
            f.write("• Purge nozzle thoroughly when changing materials\n")
            f.write("• Check layer adhesion after each material change\n")

    def export_instructions_to_csv(
        self, instructions: List[SwapInstruction], output_path: Union[str, Path]
    ) -> None:
        """Export instructions to CSV format.

        Args:
            instructions: Swap instructions
            output_path: Output CSV file path
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Layer",
                    "Height_mm",
                    "Old_Material",
                    "New_Material",
                    "Description",
                    "Estimated_Time_Seconds",
                ]
            )

            # Data
            for instruction in instructions:
                writer.writerow(
                    [
                        instruction.layer_number,
                        instruction.height_mm,
                        instruction.old_material,
                        instruction.new_material,
                        instruction.description,
                        instruction.estimated_time_seconds,
                    ]
                )


class ProjectFileGenerator:
    """Generate project files compatible with slicing software."""

    def __init__(self):
        """Initialize project file generator."""
        pass

    def generate_hueforge_project(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_database: MaterialDatabase,
        material_ids: List[str],
        output_path: Union[str, Path],
        project_name: str = "BananaForge_Project",
    ) -> Dict:
        """Generate HueForge-compatible project file.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            material_assignments: Material per layer (num_layers, H, W)
            material_database: Material database
            material_ids: Used material IDs
            output_path: Output project file path
            project_name: Name of the project

        Returns:
            Project data dictionary
        """
        # Convert tensors to numpy
        height_array = height_map.squeeze().cpu().numpy()
        assignments_array = material_assignments.cpu().numpy()

        # Build project structure
        project_data = {
            "version": "1.0",
            "generator": "BananaForge",
            "created": datetime.now().isoformat(),
            "project": {
                "name": project_name,
                "description": "Generated by BananaForge from image optimization",
                "dimensions": {
                    "width": height_array.shape[1],
                    "height": height_array.shape[0],
                    "layers": assignments_array.shape[0],
                },
            },
            "materials": [],
            "layers": [],
            "settings": {"layer_height": 0.2, "base_height": 0.4},
        }

        # Add material information
        for material_id in material_ids:
            material = material_database.get_material(material_id)
            if material:
                material_data = {
                    "id": material_id,
                    "name": material.name,
                    "brand": material.brand,
                    "color": material.color_hex,
                    "rgb": list(material.color_rgb),
                    "temperature": material.temperature,
                    "transparency": material.transparency,
                }
                project_data["materials"].append(material_data)

        # Add layer information
        for layer_idx in range(assignments_array.shape[0]):
            layer_assignment = assignments_array[layer_idx]

            # Find dominant material for this layer
            unique_materials, counts = np.unique(layer_assignment, return_counts=True)
            valid_materials_mask = unique_materials >= 0
            unique_materials = unique_materials[valid_materials_mask]
            counts = counts[valid_materials_mask]
            if len(unique_materials) == 0:
                continue
            dominant_material_idx = unique_materials[np.argmax(counts)]

            if dominant_material_idx < len(material_ids):
                dominant_material = material_ids[dominant_material_idx]
            else:
                dominant_material = material_ids[0] if material_ids else "unknown"

            layer_data = {
                "layer": layer_idx,
                "height": layer_idx * 0.2,
                "material_id": dominant_material,
                "material_coverage": float(np.max(counts) / layer_assignment.size),
            }
            project_data["layers"].append(layer_data)

        # Save project file
        with open(output_path, "w") as f:
            json.dump(project_data, f, indent=2)

        return project_data

    def generate_prusa_project(
        self,
        stl_path: Union[str, Path],
        instructions: List[SwapInstruction],
        material_database: MaterialDatabase,
        output_path: Union[str, Path],
        settings: Optional[PrintSettings] = None,
    ) -> Dict:
        """Generate PrusaSlicer-compatible project file.

        Args:
            stl_path: Path to STL file
            instructions: Material swap instructions
            material_database: Material database
            output_path: Output project file path
            settings: Print settings

        Returns:
            Project configuration dictionary
        """
        if settings is None:
            settings = PrintSettings()

        project_config = {
            "version": "PrusaSlicer-2.6.0",
            "technology": "FFF",
            "models": [
                {
                    "filename": str(stl_path),
                    "instances": [
                        {
                            "position": [0, 0, 0],
                            "rotation": [0, 0, 0],
                            "scaling": [1, 1, 1],
                        }
                    ],
                }
            ],
            "print_settings": settings.to_dict(),
            "filaments": [],
            "color_changes": [],
        }

        # Add filament information
        used_materials = set()
        for instruction in instructions:
            used_materials.add(instruction.old_material)
            used_materials.add(instruction.new_material)

        for material_id in used_materials:
            material = material_database.get_material(material_id)
            if material:
                filament_config = {
                    "name": material.name,
                    "color": material.color_hex,
                    "temperature": material.temperature,
                    "density": material.density,
                }
                project_config["filaments"].append(filament_config)

        # Add color change instructions
        for instruction in instructions:
            color_change = {
                "layer": instruction.layer_number,
                "height": instruction.height_mm,
                "old_color": instruction.old_material,
                "new_color": instruction.new_material,
            }
            project_config["color_changes"].append(color_change)

        # Save as 3MF metadata or companion file
        with open(output_path, "w") as f:
            json.dump(project_config, f, indent=2)

        return project_config

    def generate_bambu_studio_project(
        self,
        stl_path: Union[str, Path],
        material_assignments: torch.Tensor,
        material_database: MaterialDatabase,
        material_ids: List[str],
        output_path: Union[str, Path],
    ) -> Dict:
        """Generate Bambu Studio compatible project.

        Args:
            stl_path: Path to main STL file
            material_assignments: Material assignments per layer
            material_database: Material database
            material_ids: Used material IDs
            output_path: Output project path

        Returns:
            Project configuration
        """
        project_config = {
            "version": "01.08.00.00",
            "model": {
                "design_info": {
                    "Title": "BananaForge Generated Model",
                    "Designer": "BananaForge",
                    "CreateTime": datetime.now().isoformat(),
                },
                "objects": [
                    {
                        "id": 1,
                        "name": Path(stl_path).stem,
                        "mesh": str(stl_path),
                        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    }
                ],
            },
            "config": {
                "filament_settings": [],
                "process_settings": {
                    "layer_height": "0.2",
                    "first_layer_height": "0.2",
                    "wall_loops": "2",
                    "sparse_infill_density": "15%",
                },
            },
            "plate": [{"index": 1, "objects": [1], "filaments": []}],
        }

        # Add filament settings
        for i, material_id in enumerate(material_ids):
            material = material_database.get_material(material_id)
            if material:
                filament_setting = {
                    "id": i + 1,
                    "name": material.name,
                    "color": material.color_hex,
                    "temperature": material.temperature,
                    "type": "PLA",  # Default to PLA
                }
                project_config["config"]["filament_settings"].append(filament_setting)
                project_config["plate"][0]["filaments"].append(i + 1)

        # Save project
        with open(output_path, "w") as f:
            json.dump(project_config, f, indent=2)

        return project_config


class CostCalculator:
    """Calculate printing costs and material usage."""

    def __init__(self):
        """Initialize cost calculator."""
        pass

    def calculate_material_usage(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_database: MaterialDatabase,
        material_ids: List[str],
        physical_size: float = 100.0,
        infill_percentage: float = 15.0,
    ) -> Dict[str, Dict]:
        """Calculate material usage and costs.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            material_assignments: Material per layer (num_layers, H, W)
            material_database: Material database
            material_ids: Used material IDs
            physical_size: Physical size in mm
            infill_percentage: Infill percentage

        Returns:
            Dictionary with usage and cost information per material
        """
        usage_data = {}
        height_array = height_map.squeeze().cpu().numpy()
        h, w = height_array.shape

        # Calculate pixel area in mm²
        pixel_area_mm2 = (physical_size / max(h, w)) ** 2

        # Calculate volume for each material
        for material_idx, material_id in enumerate(material_ids):
            material = material_database.get_material(material_id)
            if not material:
                continue

            total_volume_mm3 = 0.0
            layer_count = 0

            # Sum volume across all layers where this material is used
            for layer_idx in range(material_assignments.shape[0]):
                layer_assignment = material_assignments[layer_idx]
                material_mask = layer_assignment == material_idx

                if material_mask.any():
                    # Count pixels with this material
                    pixel_count = material_mask.sum().item()

                    # Calculate layer volume
                    layer_volume_mm3 = (
                        pixel_count * pixel_area_mm2 * 0.2
                    )  # 0.2mm layer height

                    # Apply infill percentage (walls are always 100%)
                    wall_volume = (
                        pixel_count * pixel_area_mm2 * 0.2 * 0.2
                    )  # Assume 20% walls
                    infill_volume = (layer_volume_mm3 - wall_volume) * (
                        infill_percentage / 100.0
                    )
                    adjusted_volume = wall_volume + infill_volume

                    total_volume_mm3 += adjusted_volume
                    layer_count += 1

            # Convert to weight
            volume_cm3 = total_volume_mm3 / 1000.0
            weight_grams = volume_cm3 * material.density

            # Calculate cost
            cost_usd = (weight_grams / 1000.0) * material.cost_per_kg

            usage_data[material_id] = {
                "material_name": material.name,
                "volume_mm3": total_volume_mm3,
                "volume_cm3": volume_cm3,
                "weight_grams": weight_grams,
                "cost_usd": cost_usd,
                "layer_count": layer_count,
                "color": material.color_hex,
            }

        return usage_data

    def generate_cost_report(
        self,
        usage_data: Dict[str, Dict],
        instructions: List[SwapInstruction],
        output_path: Union[str, Path],
    ) -> None:
        """Generate detailed cost report.

        Args:
            usage_data: Material usage data
            instructions: Swap instructions
            output_path: Output report path
        """
        with open(output_path, "w") as f:
            f.write("PRINTING COST ANALYSIS\n")
            f.write("=" * 50 + "\n\n")

            # Material costs
            total_cost = 0.0
            total_weight = 0.0

            f.write("MATERIAL USAGE:\n")
            f.write("-" * 30 + "\n")

            for material_id, data in usage_data.items():
                f.write(f"Material: {data['material_name']}\n")
                f.write(f"  Weight: {data['weight_grams']:.2f}g\n")
                f.write(f"  Cost: ${data['cost_usd']:.2f}\n")
                f.write(f"  Layers: {data['layer_count']}\n\n")

                total_cost += data["cost_usd"]
                total_weight += data["weight_grams"]

            f.write(f"TOTAL MATERIAL COST: ${total_cost:.2f}\n")
            f.write(f"TOTAL WEIGHT: {total_weight:.2f}g\n\n")

            # Time estimates
            f.write("TIME ESTIMATES:\n")
            f.write("-" * 30 + "\n")

            swap_time = sum(inst.estimated_time_seconds for inst in instructions)
            swap_minutes = swap_time // 60

            # Rough print time estimate (based on material volume)
            base_print_time_hours = total_weight * 0.02  # ~0.02 hours per gram

            f.write(f"Estimated print time: {base_print_time_hours:.1f} hours\n")
            f.write(f"Material swap time: {swap_minutes} minutes\n")
            f.write(
                f"Total project time: {base_print_time_hours + swap_minutes/60:.1f} hours\n\n"
            )

            # Additional costs
            f.write("ADDITIONAL CONSIDERATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of material swaps: {len(instructions)}\n")
            f.write(
                f"Complexity factor: {'High' if len(instructions) > 5 else 'Medium' if len(instructions) > 2 else 'Low'}\n"
            )
            f.write(
                "Note: Additional costs may include electricity, printer wear, and failed prints.\n"
            )
