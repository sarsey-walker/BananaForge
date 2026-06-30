"""Bambu Studio custom G-code helpers."""

from typing import Any, Dict, Optional

from ..materials.database import MaterialDatabase
from ..utils.logging import get_logger
from .export_types import LayerMaterial

logger = get_logger(__name__)


class BambuGCodeGenerator:
    """Generates Bambu Studio layer custom G-code XML."""

    EMPTY_CUSTOM_GCODE = """<?xml version="1.0" encoding="utf-8"?>
<custom_gcodes_per_layer>
<plate>
<plate_info id="1"/>
<mode value="MultiAsSingle"/>
</plate>
</custom_gcodes_per_layer>
"""

    def __init__(self, material_db: Optional[MaterialDatabase] = None):
        self.material_db = material_db

    def generate_custom_gcode(self, material_data: Dict[str, Any]) -> str:
        """Generate custom_gcode_per_layer.xml with material swap instructions."""
        layer_materials = material_data.get("layer_materials", {})

        if not layer_materials:
            return self.EMPTY_CUSTOM_GCODE

        gcode_layers = []
        sorted_layers = sorted(layer_materials.items(), key=lambda x: x[0])
        material_to_extruder = self._map_materials_to_extruders(sorted_layers)

        prev_material = None
        for layer_idx, layer_material in sorted_layers:
            material_id, layer_height = self._get_layer_material_info(
                layer_idx, layer_material
            )

            initial_layer_height = 0.16

            if prev_material and material_id != prev_material:
                material_color = self.get_material_color(material_id)
                extruder_id = material_to_extruder.get(material_id, 1)

                if layer_idx == 0:
                    continue

                swap_layer_idx = layer_idx - 1
                if swap_layer_idx == 0:
                    swap_z = initial_layer_height
                else:
                    swap_z = initial_layer_height + (swap_layer_idx * layer_height)

                logger.info(
                    f"Layer swap at z={swap_z:.3f}: {prev_material} -> {material_id} "
                    f"(extruder {extruder_id}, color {material_color}) - for layer "
                    f"{layer_idx + 1}"
                )

                gcode_layers.append(
                    f'<layer top_z="{swap_z:.6f}" type="2" '
                    f'extruder="{extruder_id}" color="{material_color}" extra="" '
                    'gcode="tool_change"/>'
                )

            prev_material = material_id

        layers_xml = "\n".join(gcode_layers)

        return f"""<?xml version="1.0" encoding="utf-8"?>
<custom_gcodes_per_layer>
<plate>
<plate_info id="1"/>
{layers_xml}
<mode value="MultiAsSingle"/>
</plate>
</custom_gcodes_per_layer>
"""

    def get_material_color(self, material_id: str) -> str:
        """Get hex color for a material ID from material database."""
        try:
            if self.material_db is not None:
                material = self.material_db.get_material(material_id)
                if material and hasattr(material, "color_hex"):
                    return material.color_hex
        except Exception:
            pass

        color_mapping = {
            "bambu_pla_gray": "#808080",
            "bambu_pla_light_blue": "#ADD8E6",
            "bambu_pla_white": "#FFFFFF",
            "bambu_pla_black": "#000000",
            "bambu_pla_pink": "#FFC0CB",
        }
        return color_mapping.get(material_id, "#FFFFFF")

    def _map_materials_to_extruders(
        self, sorted_layers: list[tuple[int, Any]]
    ) -> Dict[str, int]:
        unique_materials = []
        for layer_idx, layer_material in sorted_layers:
            material_id, _ = self._get_layer_material_info(layer_idx, layer_material)
            if material_id not in unique_materials:
                unique_materials.append(material_id)

        return {material_id: idx + 1 for idx, material_id in enumerate(unique_materials)}

    def _get_layer_material_info(
        self, layer_idx: int, layer_material: Any
    ) -> tuple[str, float]:
        if isinstance(layer_material, LayerMaterial):
            return layer_material.material_id, layer_material.layer_height
        return (
            layer_material.get("material_id", f"material_{layer_idx}"),
            layer_material.get("layer_height", 0.08),
        )
