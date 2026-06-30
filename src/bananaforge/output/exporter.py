"""Main model exporter coordinating all output generation."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..materials.database import MaterialDatabase
from .instructions import CostCalculator, ProjectFileGenerator, SwapInstructionGenerator
from .stl_generator import STLGenerator
from .threemf_exporter import ThreeMFExportConfig, ThreeMFExporter


class ModelExporter:
    """Main exporter class coordinating all output generation."""

    def __init__(
        self,
        layer_height: float = 0.08,
        initial_layer_height: float = 0.16,
        nozzle_diameter: float = 0.4,
        base_height: float = 0.24,
        physical_size: float = 100.0,
        material_db: Optional[MaterialDatabase] = None,
        device: str = "cpu",
        bottom_mode: str = "simplified",
    ):
        """Initialize model exporter.

        Args:
            layer_height: Layer height in mm (default: 0.08mm)
            initial_layer_height: Initial layer height in mm (default: 0.16mm)
            nozzle_diameter: Nozzle diameter in mm (default: 0.4mm)
            base_height: Base height in mm
            physical_size: Physical size of model in mm
            material_db: Material database for 3MF export
            device: Device for computations
            bottom_mode: Bottom face generation mode: full, simplified, or none
        """
        self.layer_height = layer_height
        self.initial_layer_height = initial_layer_height
        self.nozzle_diameter = nozzle_diameter
        self.base_height = base_height
        self.physical_size = physical_size
        self.material_db = material_db
        self.device = device
        self.bottom_mode = STLGenerator._validate_bottom_mode(bottom_mode)

        # Initialize sub-components
        self.stl_generator = STLGenerator(
            layer_height,
            initial_layer_height,
            nozzle_diameter,
            base_height,
            bottom_mode=self.bottom_mode,
        )
        self.instruction_generator = SwapInstructionGenerator(
            layer_height, initial_layer_height
        )
        self.project_generator = ProjectFileGenerator()
        self.cost_calculator = CostCalculator()

        # Initialize 3MF exporter if material database is available
        self.threemf_exporter = None
        if material_db:
            self.threemf_exporter = ThreeMFExporter(
                device=device, material_db=material_db
            )

        self.logger = logging.getLogger(__name__)

    def export_complete_model(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_database: MaterialDatabase,
        material_ids: List[str],
        output_dir: Union[str, Path],
        project_name: str = "bananaforge_model",
        export_formats: Optional[List[str]] = None,
        bambu_compatible: bool = False,
        source_image_path: Optional[str] = None,
        max_triangles: Optional[int] = None,
    ) -> Dict[str, str]:
        """Export complete model with all associated files.

        Args:
            height_map: Optimized height map (1, 1, H, W)
            material_assignments: Material per layer (num_layers, H, W)
            material_database: Material database
            material_ids: List of used material IDs
            output_dir: Output directory
            project_name: Base name for output files
            export_formats: List of formats to export
            bambu_compatible: Generate 3MF files optimized for Bambu Studio compatibility
            max_triangles: Optional triangle budget for generated mesh files

        Returns:
            Dictionary mapping output type to file path
        """
        if export_formats is None:
            export_formats = ["stl", "instructions", "hueforge", "cost_report"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        try:
            height_map, material_assignments = self._apply_triangle_limit(
                height_map=height_map,
                material_assignments=material_assignments,
                max_triangles=max_triangles,
            )

            # Generate lightweight outputs first so they survive heavy mesh exports.
            instructions = None
            if "instructions" in export_formats:
                instructions = self.instruction_generator.generate_swap_instructions(
                    material_assignments=material_assignments,
                    material_database=material_database,
                    material_ids=material_ids,
                )

                instruction_txt_path = output_dir / f"{project_name}_instructions.txt"

                from .instructions import PrintSettings

                print_settings = PrintSettings(
                    layer_height=self.layer_height,
                    nozzle_temperature=220,  # Default PLA temperature
                    bed_temperature=60,
                    print_speed=50,
                    infill_percentage=15,
                    supports=False,
                    brim=False,
                )

                self.instruction_generator.export_instructions_to_text(
                    instructions=instructions,
                    output_path=instruction_txt_path,
                    material_database=material_database,
                    print_settings=print_settings,
                )
                generated_files["instructions_txt"] = str(instruction_txt_path)

                instruction_csv_path = output_dir / f"{project_name}_instructions.csv"
                self.instruction_generator.export_instructions_to_csv(
                    instructions, instruction_csv_path
                )
                generated_files["instructions_csv"] = str(instruction_csv_path)

                self.logger.info(f"Generated {len(instructions)} swap instructions")

            if "cost_report" in export_formats:
                usage_data = self.cost_calculator.calculate_material_usage(
                    height_map=height_map,
                    material_assignments=material_assignments,
                    material_database=material_database,
                    material_ids=material_ids,
                    physical_size=self.physical_size,
                )

                if instructions is None:
                    instructions = (
                        self.instruction_generator.generate_swap_instructions(
                            material_assignments, material_database, material_ids
                        )
                    )

                cost_report_path = output_dir / f"{project_name}_cost_report.txt"
                self.cost_calculator.generate_cost_report(
                    usage_data=usage_data,
                    instructions=instructions,
                    output_path=cost_report_path,
                )
                generated_files["cost_report"] = str(cost_report_path)

                cost_json_path = output_dir / f"{project_name}_cost_data.json"
                with open(cost_json_path, "w") as f:
                    json.dump(usage_data, f, indent=2)
                generated_files["cost_data"] = str(cost_json_path)

            # 1. Generate main STL file
            if "stl" in export_formats:
                stl_path = output_dir / f"{project_name}.stl"
                self.stl_generator.generate_stl(
                    height_map=height_map,
                    output_path=stl_path,
                    physical_size=self.physical_size,
                )
                generated_files["stl"] = str(stl_path)
                self.logger.info(f"Generated STL: {stl_path}")

            # 1.5. Generate 3MF file if requested
            if "3mf" in export_formats and self.threemf_exporter:
                threemf_path = output_dir / f"{project_name}.3mf"

                # Prepare optimization results for 3MF export
                optimization_results = {
                    "heightmap": height_map,
                    "material_assignments": material_assignments,
                    "layer_materials": self._create_layer_materials_dict(
                        material_assignments, material_ids
                    ),
                    "optimization_metadata": {
                        "physical_size": self.physical_size,
                        "layer_height": self.layer_height,
                        "project_name": project_name,
                    },
                    "source_image_path": source_image_path,
                    "stl_path": generated_files.get("stl"),
                    "max_triangles": max_triangles,
                    "bottom_mode": self.bottom_mode,
                }

                # Configure 3MF export
                threemf_config = ThreeMFExportConfig(
                    bambu_compatible=bambu_compatible or "bambu" in export_formats,
                    include_metadata=True,
                    validate_output=True,
                )

                # Export to 3MF
                result = self.threemf_exporter.export(
                    optimization_results, threemf_path, threemf_config
                )

                if result["success"]:
                    generated_files["3mf"] = str(threemf_path)
                    self.logger.info(
                        f"Generated 3MF: {threemf_path} ({result['file_size']} bytes)"
                    )
                else:
                    self.logger.error(
                        f"3MF export failed: {result.get('error', 'Unknown error')}"
                    )

            # 3. Generate project files
            if "hueforge" in export_formats:
                hueforge_path = output_dir / f"{project_name}.hfp"
                self.project_generator.generate_hueforge_project(
                    height_map=height_map,
                    material_assignments=material_assignments,
                    material_database=material_database,
                    material_ids=material_ids,
                    output_path=hueforge_path,
                    project_name=project_name,
                )
                generated_files["hueforge"] = str(hueforge_path)
                self.logger.info(f"Generated HueForge project: {hueforge_path}")

            if "prusa" in export_formats and "stl" in generated_files:
                prusa_path = output_dir / f"{project_name}_prusa.json"
                instructions = self.instruction_generator.generate_swap_instructions(
                    material_assignments, material_database, material_ids
                )
                self.project_generator.generate_prusa_project(
                    stl_path=generated_files["stl"],
                    instructions=instructions,
                    material_database=material_database,
                    output_path=prusa_path,
                )
                generated_files["prusa"] = str(prusa_path)

            if "bambu" in export_formats and "stl" in generated_files:
                bambu_path = output_dir / f"{project_name}_bambu.json"
                self.project_generator.generate_bambu_studio_project(
                    stl_path=generated_files["stl"],
                    material_assignments=material_assignments,
                    material_database=material_database,
                    material_ids=material_ids,
                    output_path=bambu_path,
                )
                generated_files["bambu"] = str(bambu_path)

            # 5. Generate individual layer STLs if requested
            if "layer_stls" in export_formats:
                layer_dir = output_dir / "layers"
                layer_files = self.stl_generator.generate_layer_stls(
                    height_map=height_map,
                    material_assignments=material_assignments,
                    material_ids=material_ids,
                    output_dir=layer_dir,
                    physical_size=self.physical_size,
                )
                generated_files.update(layer_files)

            # 6. Generate preview mesh if requested
            if "preview" in export_formats:
                material_colors = material_database.get_color_palette("cpu")
                preview_mesh = self.stl_generator.create_preview_mesh(
                    height_map=height_map,
                    material_assignments=material_assignments,
                    material_colors=material_colors,
                    physical_size=self.physical_size,
                )

                preview_path = output_dir / f"{project_name}_preview.ply"
                preview_mesh.export(str(preview_path))
                generated_files["preview"] = str(preview_path)

            # 7. Generate summary report
            self._generate_summary_report(
                generated_files=generated_files,
                material_database=material_database,
                material_ids=material_ids,
                height_map=height_map,
                output_path=output_dir / f"{project_name}_summary.txt",
            )
            generated_files["summary"] = str(output_dir / f"{project_name}_summary.txt")

        except Exception as e:
            self.logger.error(f"Error during export: {e}")
            raise

        return generated_files

    @staticmethod
    def estimate_triangle_count(
        height: int, width: int, bottom_mode: str = "simplified"
    ) -> int:
        """Estimate triangles generated by the heightmap mesh."""
        bottom_mode = STLGenerator._validate_bottom_mode(bottom_mode)
        if height < 2 or width < 2:
            return 0

        top_triangles = 2 * (height - 1) * (width - 1)
        side_triangles = 4 * ((height - 1) + (width - 1))
        if bottom_mode == "none":
            bottom_triangles = 0
        elif bottom_mode == "simplified":
            bottom_triangles = 2
        else:
            bottom_triangles = 2 * (height - 1) * (width - 1)
        return int(top_triangles + side_triangles + bottom_triangles)

    @staticmethod
    def estimate_binary_stl_size_bytes(triangle_count: int) -> int:
        """Estimate binary STL size for a triangle count."""
        return 84 + (triangle_count * 50)

    @classmethod
    def fit_shape_to_triangle_limit(
        cls,
        height: int,
        width: int,
        max_triangles: int,
        bottom_mode: str = "simplified",
    ) -> Tuple[int, int]:
        """Return a downscaled shape that fits within the triangle limit."""
        if max_triangles <= 0:
            raise ValueError("max_triangles must be positive")

        current_triangles = cls.estimate_triangle_count(height, width, bottom_mode)
        if current_triangles <= max_triangles:
            return height, width

        scale = math.sqrt(max_triangles / current_triangles)
        target_height = max(2, int(height * scale))
        target_width = max(2, int(width * scale))

        while (
            cls.estimate_triangle_count(target_height, target_width, bottom_mode)
            > max_triangles
            and target_height > 2
            and target_width > 2
        ):
            target_height = max(2, target_height - 1)
            target_width = max(2, target_width - 1)

        return target_height, target_width

    def _apply_triangle_limit(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        max_triangles: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downscale export tensors when a triangle budget is requested."""
        if max_triangles is None:
            return height_map, material_assignments

        if max_triangles <= 0:
            raise ValueError("max_triangles must be positive")

        height, width = height_map.shape[-2:]
        estimated_triangles = self.estimate_triangle_count(
            height, width, self.bottom_mode
        )
        if estimated_triangles <= max_triangles:
            return height_map, material_assignments

        target_height, target_width = self.fit_shape_to_triangle_limit(
            height, width, max_triangles, self.bottom_mode
        )
        target_triangles = self.estimate_triangle_count(
            target_height, target_width, self.bottom_mode
        )

        self.logger.warning(
            "Downscaling export mesh from %sx%s (%s triangles) to %sx%s "
            "(%s triangles) to satisfy max_triangles=%s",
            width,
            height,
            estimated_triangles,
            target_width,
            target_height,
            target_triangles,
            max_triangles,
        )

        resized_height_map = F.interpolate(
            height_map.float(),
            size=(target_height, target_width),
            mode="nearest",
        ).to(height_map.dtype)

        resized_assignments = (
            F.interpolate(
                material_assignments.float().unsqueeze(0),
                size=(target_height, target_width),
                mode="nearest",
            )
            .squeeze(0)
            .to(material_assignments.dtype)
        )

        return resized_height_map, resized_assignments

    def _create_layer_materials_dict(
        self, material_assignments: torch.Tensor, material_ids: List[str]
    ) -> Dict[int, Dict[str, Union[str, float]]]:
        """Create layer materials dictionary from material assignments.

        Args:
            material_assignments: Tensor of material assignments per layer (num_layers, H, W)
            material_ids: List of material IDs

        Returns:
            Dictionary mapping layer index to material info
        """
        layer_materials = {}
        num_layers = material_assignments.shape[0]

        # Determine dominant material for each layer (same logic as SwapInstructionGenerator)
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
                continue
            dominant_material_idx = unique_materials[torch.argmax(counts)].item()

            if dominant_material_idx < len(material_ids):
                dominant_material = material_ids[int(dominant_material_idx)]
            else:
                dominant_material = material_ids[0] if material_ids else "unknown"

            layer_materials[layer_idx] = {
                "material_id": dominant_material,
                "transparency": 1.0,
                "layer_height": self.layer_height,
            }

        return layer_materials

    def export_for_printer(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_database: MaterialDatabase,
        material_ids: List[str],
        output_dir: Union[str, Path],
        printer_type: str = "bambu",
        project_name: str = "bananaforge_model",
    ) -> Dict[str, str]:
        """Export model optimized for specific printer type.

        Args:
            height_map: Optimized height map
            material_assignments: Material assignments
            material_database: Material database
            material_ids: Used material IDs
            output_dir: Output directory
            printer_type: Target printer ("bambu", "prusa", "generic")
            project_name: Project name

        Returns:
            Dictionary of generated files
        """
        # Define export formats based on printer type
        format_map = {
            "bambu": ["stl", "instructions", "bambu", "cost_report"],
            "prusa": ["stl", "instructions", "prusa", "cost_report"],
            "hueforge": ["stl", "instructions", "hueforge", "cost_report"],
            "generic": ["stl", "instructions", "cost_report"],
        }

        export_formats = format_map.get(printer_type, format_map["generic"])

        return self.export_complete_model(
            height_map=height_map,
            material_assignments=material_assignments,
            material_database=material_database,
            material_ids=material_ids,
            output_dir=output_dir,
            project_name=project_name,
            export_formats=export_formats,
        )

    def _generate_summary_report(
        self,
        generated_files: Dict[str, str],
        material_database: MaterialDatabase,
        material_ids: List[str],
        height_map: torch.Tensor,
        output_path: Path,
    ) -> None:
        """Generate summary report of the export process."""
        with open(output_path, "w") as f:
            f.write("BANANAFORGE MODEL EXPORT SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Model information
            h, w = height_map.shape[-2:]
            max_height = height_map.max().item() * self.layer_height

            f.write("MODEL INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Resolution: {w} x {h} pixels\n")
            f.write(f"Physical size: {self.physical_size}mm\n")
            f.write(f"Maximum height: {max_height:.2f}mm\n")
            f.write(f"Layer height: {self.layer_height}mm\n")
            f.write(f"Number of materials: {len(material_ids)}\n\n")

            # Materials used
            f.write("MATERIALS USED:\n")
            f.write("-" * 30 + "\n")
            for i, material_id in enumerate(material_ids):
                material = material_database.get_material(material_id)
                if material:
                    f.write(
                        f"{i+1}. {material.name} ({material.brand}) - {material.color_hex}\n"
                    )
                else:
                    f.write(f"{i+1}. {material_id} (Unknown material)\n")
            f.write("\n")

            # Generated files
            f.write("GENERATED FILES:\n")
            f.write("-" * 30 + "\n")
            for file_type, file_path in generated_files.items():
                f.write(f"{file_type}: {Path(file_path).name}\n")
            f.write("\n")

            # Instructions
            f.write("PRINTING INSTRUCTIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Load the STL file into your slicer\n")
            f.write("2. Configure materials according to the material list\n")
            f.write("3. Follow swap instructions during printing\n")
            f.write("4. Refer to cost report for material usage estimates\n\n")

            f.write("Generated by BananaForge - 3D Printing Optimization\n")

    def validate_export_requirements(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_ids: List[str],
    ) -> Tuple[bool, List[str]]:
        """Validate that all requirements for export are met.

        Args:
            height_map: Height map tensor
            material_assignments: Material assignments
            material_ids: Material IDs

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check height map
        if height_map.numel() == 0:
            issues.append("Height map is empty")
        elif height_map.max() <= 0:
            issues.append("Height map has no positive values")

        # Check material assignments
        if material_assignments.numel() == 0:
            issues.append("Material assignments are empty")
        elif material_assignments.max() >= len(material_ids):
            issues.append("Material assignment indices exceed available materials")

        # Check material IDs
        if not material_ids:
            issues.append("No material IDs provided")

        # Check dimensions match
        if height_map.shape[-2:] != material_assignments.shape[-2:]:
            issues.append("Height map and material assignment dimensions don't match")

        return len(issues) == 0, issues

    def estimate_print_time(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_database: MaterialDatabase,
        material_ids: List[str],
        print_speed: int = 50,  # mm/s
    ) -> Dict[str, float]:
        """Estimate printing time and costs.

        Args:
            height_map: Height map tensor
            material_assignments: Material assignments
            material_database: Material database
            material_ids: Material IDs
            print_speed: Print speed in mm/s

        Returns:
            Dictionary with time and cost estimates
        """
        # Calculate material usage
        usage_data = self.cost_calculator.calculate_material_usage(
            height_map=height_map,
            material_assignments=material_assignments,
            material_database=material_database,
            material_ids=material_ids,
            physical_size=self.physical_size,
        )

        # Estimate print time based on volume and speed
        total_volume = sum(data["volume_mm3"] for data in usage_data.values())

        # Rough time estimation (this is very approximate)
        layer_area_mm2 = self.physical_size**2
        num_layers = height_map.max().item()
        total_path_length = layer_area_mm2 * num_layers * 0.1  # Estimate path density

        print_time_seconds = total_path_length / print_speed
        print_time_hours = print_time_seconds / 3600

        # Add swap time
        instructions = self.instruction_generator.generate_swap_instructions(
            material_assignments, material_database, material_ids
        )
        swap_time_minutes = (
            sum(inst.estimated_time_seconds for inst in instructions) / 60
        )

        # Calculate costs
        total_material_cost = sum(data["cost_usd"] for data in usage_data.values())

        return {
            "print_time_hours": print_time_hours,
            "swap_time_minutes": swap_time_minutes,
            "total_time_hours": print_time_hours + swap_time_minutes / 60,
            "material_cost_usd": total_material_cost,
            "num_swaps": len(instructions),
            "total_volume_cm3": total_volume / 1000,
        }

    def export_hueforge_project(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        optimization_params: Dict,
        output_path: Union[str, Path],
        alpha_mask: Optional[torch.Tensor] = None,
        hueforge_version: str = "2.0",
        include_swap_instructions: bool = True,
    ) -> Dict[str, any]:
        """Export project as HueForge-compatible .hfp file.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            material_assignments: Material assignments tensor (L, H, W)
            materials: List of material definitions
            optimization_params: Optimization parameters and metadata
            output_path: Output .hfp file path
            alpha_mask: Optional alpha mask for transparency support
            hueforge_version: Target HueForge version
            include_swap_instructions: Whether to include swap instructions

        Returns:
            Dictionary with export results and metadata
        """
        import json
        import tempfile
        import zipfile
        from datetime import datetime

        output_path = Path(output_path)

        # Create temporary directory for HFP contents
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Generate project metadata
            project_data = {
                "hueforge_version": hueforge_version,
                "created_by": "BananaForge",
                "created_at": datetime.now().isoformat(),
                "schema_version": "2.0",
                "print_settings": {
                    "layer_height": optimization_params.get(
                        "layer_height", self.layer_height
                    ),
                    "initial_layer_height": optimization_params.get(
                        "initial_layer_height", self.initial_layer_height
                    ),
                    "total_layers": optimization_params.get(
                        "total_layers", int(height_map.max().item())
                    ),
                    "nozzle_diameter": optimization_params.get(
                        "nozzle_diameter", self.nozzle_diameter
                    ),
                    "print_speed": optimization_params.get("print_speed", 150),
                    "temperature": optimization_params.get("temperature", 210),
                    "bed_temperature": optimization_params.get("bed_temperature", 60),
                    "infill_percentage": optimization_params.get(
                        "infill_percentage", 15
                    ),
                    "supports": optimization_params.get("supports", False),
                    "brim": optimization_params.get("brim", True),
                    "brim_width": optimization_params.get("brim_width", 5.0),
                },
                "model_info": {
                    "dimensions": {
                        "width_mm": self.physical_size,
                        "height_mm": self.physical_size,
                        "depth_mm": float(height_map.max().item() * self.layer_height),
                    },
                    "resolution": {
                        "width_px": height_map.shape[-1],
                        "height_px": height_map.shape[-2],
                    },
                    "estimated_print_time": self._estimate_hueforge_print_time(
                        height_map, material_assignments
                    ),
                    "estimated_material_usage": self._estimate_material_usage_for_hfp(
                        height_map, material_assignments, materials
                    ),
                },
                "optimization_info": {
                    "iterations": optimization_params.get("iterations", 0),
                    "final_loss": optimization_params.get("final_loss", 0.0),
                    "optimizer_type": "BananaForge Enhanced",
                },
            }

            # Add alpha channel support info
            if alpha_mask is not None:
                project_data["alpha_support"] = True
                transparency_regions = self._detect_transparency_regions(alpha_mask)
                if transparency_regions:
                    project_data["transparency_regions"] = transparency_regions
            else:
                project_data["alpha_support"] = False

            # Save project metadata
            with open(temp_path / "project.json", "w") as f:
                json.dump(project_data, f, indent=2)

            # 2. Generate material definitions
            materials_data = {"version": "1.0", "materials": []}

            for material in materials:
                material_def = {
                    "id": material["id"],
                    "name": material["name"],
                    "color": material["color"],
                    "brand": material.get("brand", "Unknown"),
                    "density": material.get("density", 1.24),
                    "cost_per_kg": material.get("cost_per_kg", 25.0),
                    "temperature": material.get("temperature", 210),
                    "bed_temperature": material.get("bed_temperature", 60),
                }
                materials_data["materials"].append(material_def)

            with open(temp_path / "materials.json", "w") as f:
                json.dump(materials_data, f, indent=2)

            # 3. Generate swap instructions
            if include_swap_instructions:
                material_ids = [m["id"] for m in materials]

                # Create a mock material database for instruction generation
                from ..materials.database import MaterialDatabase

                mock_db = MaterialDatabase()
                for material in materials:
                    # This would need proper Material object creation in a real implementation
                    pass

                instructions = self.instruction_generator.generate_swap_instructions(
                    material_assignments=material_assignments,
                    material_database=mock_db,
                    material_ids=material_ids,
                )

                instructions_data = {"version": "1.0", "swap_instructions": []}

                for inst in instructions:
                    instruction_dict = {
                        "layer": inst.layer_number,
                        "from_material": inst.old_material,
                        "to_material": inst.new_material,
                        "estimated_time": inst.estimated_time_seconds,
                        "pause_required": True,
                        "notes": f"Swap from {inst.old_material} to {inst.new_material}",
                    }
                    instructions_data["swap_instructions"].append(instruction_dict)

                with open(temp_path / "instructions.json", "w") as f:
                    json.dump(instructions_data, f, indent=2)

            # 4. Generate STL files for each material
            material_ids = [m["id"] for m in materials]

            if alpha_mask is not None:
                # Use alpha-aware STL generation
                self.stl_generator.generate_layer_stls_with_alpha(
                    height_map=height_map,
                    material_assignments=material_assignments,
                    material_ids=material_ids,
                    alpha_mask=alpha_mask,
                    output_dir=temp_path,
                    physical_size=self.physical_size,
                )
            else:
                # Use standard STL generation
                self.stl_generator.generate_layer_stls(
                    height_map=height_map,
                    material_assignments=material_assignments,
                    material_ids=material_ids,
                    output_dir=temp_path,
                    physical_size=self.physical_size,
                )

            # 5. Create the .hfp ZIP archive
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from temp directory
                for file_path in temp_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_path)
                        zipf.write(file_path, arcname)

        self.logger.info(f"HueForge project exported to {output_path}")

        return {
            "success": True,
            "output_path": str(output_path),
            "materials_count": len(materials),
            "has_alpha_support": alpha_mask is not None,
            "hueforge_version": hueforge_version,
        }

    def validate_hueforge_compatibility(
        self, project_path: Union[str, Path]
    ) -> Dict[str, any]:
        """Validate that a project file is compatible with HueForge.

        Args:
            project_path: Path to .hfp project file

        Returns:
            Dictionary with validation results
        """
        import json
        import zipfile

        project_path = Path(project_path)

        try:
            if not zipfile.is_zipfile(project_path):
                return {"valid": False, "error": "File is not a valid ZIP archive"}

            with zipfile.ZipFile(project_path, "r") as zipf:
                file_list = zipf.namelist()

                # Check required files
                required_files = ["project.json", "materials.json"]
                missing_files = [f for f in required_files if f not in file_list]
                if missing_files:
                    return {
                        "valid": False,
                        "error": f"Missing required files: {missing_files}",
                    }

                # Validate project.json
                project_data = json.loads(zipf.read("project.json"))

                if "hueforge_version" not in project_data:
                    return {"valid": False, "error": "Missing HueForge version"}

                version = project_data["hueforge_version"]

                # Check for STL files
                stl_files = [f for f in file_list if f.endswith(".stl")]
                if not stl_files:
                    return {"valid": False, "error": "No STL files found"}

                return {
                    "valid": True,
                    "version": version,
                    "schema_version": project_data.get("schema_version", "1.0"),
                    "compatible_slicers": ["HueForge", "PrusaSlicer", "Bambu Studio"],
                    "stl_count": len(stl_files),
                    "has_instructions": "instructions.json" in file_list,
                }

        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}

    def _estimate_hueforge_print_time(
        self, height_map: torch.Tensor, material_assignments: torch.Tensor
    ) -> str:
        """Estimate print time for HueForge project."""
        # Simplified estimation
        num_layers = int(height_map.max().item())
        area_mm2 = self.physical_size**2

        # Rough estimate: 1 minute per layer per 100mm²
        base_time_minutes = (num_layers * area_mm2) / 100

        # Add swap time
        num_swaps = self._count_material_swaps(material_assignments)
        swap_time_minutes = num_swaps * 2  # 2 minutes per swap

        total_minutes = base_time_minutes + swap_time_minutes
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        return f"{hours}h {minutes}m"

    def _estimate_material_usage_for_hfp(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        materials: List[Dict],
    ) -> Dict[str, float]:
        """Estimate material usage for HueForge project."""
        usage = {}

        # Calculate volume per material
        layer_volume_mm3 = (self.physical_size**2) * self.layer_height

        for i, material in enumerate(materials):
            material_layers = (material_assignments == i).sum().item()
            volume_mm3 = material_layers * layer_volume_mm3

            # Convert to grams (assuming PLA density ~1.24 g/cm³)
            volume_cm3 = volume_mm3 / 1000
            weight_grams = volume_cm3 * material.get("density", 1.24)

            usage[material["id"]] = {
                "weight_grams": round(weight_grams, 2),
                "volume_cm3": round(volume_cm3, 2),
            }

        return usage

    def _detect_transparency_regions(self, alpha_mask: torch.Tensor) -> List[Dict]:
        """Detect transparency regions in alpha mask."""
        import numpy as np
        from scipy import ndimage

        # Convert to numpy and find transparent regions
        alpha_np = alpha_mask.cpu().numpy()
        transparent_regions = ~alpha_np  # Invert to get transparent areas

        # Label connected components
        labeled, num_features = ndimage.label(transparent_regions)

        regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled == i

            # Get bounding box
            coords = np.where(region_mask)
            if len(coords[0]) > 0:
                min_y, max_y = coords[0].min(), coords[0].max()
                min_x, max_x = coords[1].min(), coords[1].max()

                regions.append(
                    {
                        "id": i,
                        "bounding_box": {
                            "min_x": int(min_x),
                            "max_x": int(max_x),
                            "min_y": int(min_y),
                            "max_y": int(max_y),
                        },
                        "area_pixels": int(region_mask.sum()),
                    }
                )

        return regions

    def _count_material_swaps(self, material_assignments: torch.Tensor) -> int:
        """Count the number of material swaps needed."""
        if material_assignments.numel() == 0:
            return 0

        swaps = 0
        prev_material = None

        for layer in range(material_assignments.shape[0]):
            layer_materials = torch.unique(material_assignments[layer])
            layer_materials = layer_materials[
                layer_materials >= 0
            ]  # Remove invalid materials

            if len(layer_materials) > 0:
                current_material = layer_materials[
                    0
                ].item()  # Simplified: use first material
                if prev_material is not None and current_material != prev_material:
                    swaps += 1
                prev_material = current_material

        return swaps
