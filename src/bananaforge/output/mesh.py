"""Advanced mesh processing utilities."""

from typing import List, Optional

import cv2
import numpy as np
import trimesh


class MeshProcessor:
    """Advanced mesh processing and optimization utilities."""

    def __init__(self):
        """Initialize mesh processor."""
        pass

    def optimize_mesh_for_printing(
        self,
        mesh: trimesh.Trimesh,
        target_face_count: Optional[int] = None,
        smooth_iterations: int = 2,
        fix_normals: bool = True,
    ) -> trimesh.Trimesh:
        """Optimize mesh for 3D printing.

        Args:
            mesh: Input mesh
            target_face_count: Target number of faces for decimation
            smooth_iterations: Number of smoothing iterations
            fix_normals: Whether to fix face normals

        Returns:
            Optimized mesh
        """
        optimized_mesh = mesh.copy()

        # Remove degenerate faces
        optimized_mesh.remove_degenerate_faces()
        optimized_mesh.remove_duplicate_faces()
        optimized_mesh.remove_unreferenced_vertices()

        # Fix normals if requested
        if fix_normals:
            optimized_mesh.fix_normals()

        # Decimate if target face count specified
        if target_face_count and len(optimized_mesh.faces) > target_face_count:
            optimized_mesh = self._decimate_mesh(optimized_mesh, target_face_count)

        # Apply smoothing
        if smooth_iterations > 0:
            optimized_mesh = optimized_mesh.smoothed(iterations=smooth_iterations)

        # Ensure watertight
        if not optimized_mesh.is_watertight:
            optimized_mesh.fill_holes()

        return optimized_mesh

    def _decimate_mesh(
        self, mesh: trimesh.Trimesh, target_faces: int
    ) -> trimesh.Trimesh:
        """Decimate mesh to reduce face count."""
        try:
            # Calculate decimation ratio
            current_faces = len(mesh.faces)
            ratio = target_faces / current_faces

            if ratio < 1.0:
                # Use trimesh simplification
                decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
                return decimated
            else:
                return mesh
        except Exception:
            # Fallback to original mesh if decimation fails
            return mesh

    def add_support_structures(
        self,
        mesh: trimesh.Trimesh,
        overhang_angle: float = 45.0,
        support_density: float = 0.5,
    ) -> trimesh.Trimesh:
        """Add support structures for overhangs.

        Args:
            mesh: Input mesh
            overhang_angle: Angle threshold for supports (degrees)
            support_density: Density of support structures

        Returns:
            Mesh with support structures
        """
        # Find faces that need support
        face_normals = mesh.face_normals
        z_component = face_normals[:, 2]

        # Faces with normal z-component less than threshold need support
        threshold = np.cos(np.radians(90 - overhang_angle))
        needs_support = z_component < threshold

        if not np.any(needs_support):
            return mesh  # No supports needed

        support_faces = mesh.faces[needs_support]
        support_vertices = mesh.vertices[support_faces].reshape(-1, 3)

        # Create simple pillar supports
        supports = []
        for vertex in support_vertices[:: int(1 / support_density)]:
            if vertex[2] > 0.5:  # Only add supports above base
                support_pillar = self._create_support_pillar(vertex)
                supports.append(support_pillar)

        if supports:
            # Combine original mesh with supports
            combined_mesh = trimesh.util.concatenate([mesh] + supports)
            return combined_mesh
        else:
            return mesh

    def _create_support_pillar(
        self, top_point: np.ndarray, radius: float = 0.5
    ) -> trimesh.Trimesh:
        """Create a cylindrical support pillar."""
        height = top_point[2]
        # Create cylinder
        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=8)

        # Position cylinder
        cylinder.apply_translation([top_point[0], top_point[1], height / 2])

        return cylinder

    def create_hollowed_version(
        self,
        mesh: trimesh.Trimesh,
        wall_thickness: float = 2.0,
        infill_percentage: float = 15.0,
    ) -> trimesh.Trimesh:
        """Create hollowed version with infill structure.

        Args:
            mesh: Input solid mesh
            wall_thickness: Wall thickness in mm
            infill_percentage: Infill density percentage

        Returns:
            Hollowed mesh with infill
        """
        # Create offset surface for hollow interior
        try:
            # Simple approach: scale down the mesh
            scale_factor = 1.0 - (2 * wall_thickness / mesh.scale)
            inner_mesh = mesh.copy()
            inner_mesh.apply_scale(scale_factor)

            # Invert normals for inner surface
            inner_mesh.faces = inner_mesh.faces[:, [0, 2, 1]]

            # Create infill structure if needed
            infill_meshes = []
            if infill_percentage > 0:
                infill_meshes = self._create_infill_structure(
                    mesh, inner_mesh, infill_percentage
                )

            # Combine outer shell, inner shell, and infill
            all_meshes = [mesh, inner_mesh] + infill_meshes
            hollow_mesh = trimesh.util.concatenate(all_meshes)

            return hollow_mesh

        except Exception:
            # Return original mesh if hollowing fails
            return mesh

    def _create_infill_structure(
        self, outer_mesh: trimesh.Trimesh, inner_mesh: trimesh.Trimesh, density: float
    ) -> List[trimesh.Trimesh]:
        """Create infill structure between outer and inner surfaces."""
        infill_structures = []

        # Simple cubic infill pattern
        bounds = outer_mesh.bounds
        spacing = 5.0 / (density / 100.0)  # Adjust spacing based on density

        x_range = np.arange(bounds[0, 0], bounds[1, 0], spacing)
        y_range = np.arange(bounds[0, 1], bounds[1, 1], spacing)
        z_range = np.arange(bounds[0, 2], bounds[1, 2], spacing)

        for x in x_range[::2]:  # Skip every other to create pattern
            for y in y_range[::2]:
                for z in z_range[::2]:
                    # Create small cube at this position
                    cube = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
                    cube.apply_translation([x, y, z])

                    # Only add if inside outer mesh but outside inner mesh
                    if (
                        outer_mesh.contains([cube.centroid])[0]
                        and not inner_mesh.contains([cube.centroid])[0]
                    ):
                        infill_structures.append(cube)

        return infill_structures

    def repair_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Repair common mesh issues.

        Args:
            mesh: Input mesh

        Returns:
            Repaired mesh
        """
        repaired = mesh.copy()

        # Remove duplicate and degenerate faces
        repaired.remove_duplicate_faces()
        repaired.remove_degenerate_faces()
        repaired.remove_unreferenced_vertices()

        # Fix normals
        repaired.fix_normals()

        # Fill holes
        if not repaired.is_watertight:
            repaired.fill_holes()

        # Merge vertices that are very close
        repaired.merge_vertices()

        return repaired

    def analyze_printability(self, mesh: trimesh.Trimesh) -> dict:
        """Analyze mesh for 3D printing issues.

        Args:
            mesh: Input mesh

        Returns:
            Dictionary of printability analysis
        """
        analysis = {
            "is_watertight": mesh.is_watertight,
            "is_winding_consistent": mesh.is_winding_consistent,
            "has_degenerate_faces": False,
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "volume": mesh.volume if mesh.is_volume else 0.0,
            "surface_area": mesh.area,
            "bounds": mesh.bounds.tolist(),
            "overhangs": [],
            "thin_features": [],
            "issues": [],
        }

        # Check for degenerate faces
        try:
            face_areas = mesh.area_faces
            degenerate_count = np.sum(face_areas < 1e-10)
            analysis["has_degenerate_faces"] = degenerate_count > 0
            analysis["degenerate_face_count"] = int(degenerate_count)
        except Exception:
            analysis["degenerate_face_count"] = -1

        # Check for overhangs
        if hasattr(mesh, "face_normals"):
            face_normals = mesh.face_normals
            z_component = face_normals[:, 2]
            overhang_threshold = np.cos(np.radians(45))  # 45 degree overhang
            overhang_faces = np.where(z_component < overhang_threshold)[0]
            analysis["overhang_face_count"] = len(overhang_faces)
            analysis["has_overhangs"] = len(overhang_faces) > 0

        # Check for thin features (simplified)
        bounds_size = mesh.bounds[1] - mesh.bounds[0]
        min_dimension = np.min(bounds_size)
        analysis["minimum_feature_size"] = float(min_dimension)
        analysis["has_thin_features"] = min_dimension < 0.4  # 0.4mm threshold

        # Compile issues list
        if not analysis["is_watertight"]:
            analysis["issues"].append("Mesh is not watertight")
        if not analysis["is_winding_consistent"]:
            analysis["issues"].append("Inconsistent face winding")
        if analysis["has_degenerate_faces"]:
            analysis["issues"].append("Contains degenerate faces")
        if analysis["has_overhangs"]:
            analysis["issues"].append("Contains steep overhangs (may need supports)")
        if analysis["has_thin_features"]:
            analysis["issues"].append(
                "Contains very thin features (may not print well)"
            )

        return analysis

    def slice_preview(
        self,
        mesh: trimesh.Trimesh,
        layer_height: float = 0.2,
        num_preview_layers: int = 10,
    ) -> List[np.ndarray]:
        """Generate preview slices of the mesh.

        Args:
            mesh: Input mesh
            layer_height: Layer height in mm
            num_preview_layers: Number of layers to preview

        Returns:
            List of 2D arrays representing layer slices
        """
        slices = []

        # Get mesh bounds
        min_z = mesh.bounds[0, 2]
        max_z = mesh.bounds[1, 2]

        # Select preview layer heights
        total_layers = int((max_z - min_z) / layer_height)
        if total_layers <= num_preview_layers:
            layer_indices = range(total_layers)
        else:
            # Evenly distribute preview layers
            layer_indices = np.linspace(
                0, total_layers - 1, num_preview_layers, dtype=int
            )

        for layer_idx in layer_indices:
            z_height = min_z + layer_idx * layer_height

            # Create cross-section at this height
            try:
                slice_2d = mesh.section(
                    plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1]
                )

                if slice_2d is not None:
                    # Convert to 2D array representation
                    slice_array = self._slice_to_array(slice_2d, mesh.bounds)
                    slices.append(slice_array)

            except Exception:
                # Skip this layer if sectioning fails
                continue

        return slices

    def _slice_to_array(
        self, slice_2d, bounds: np.ndarray, resolution: int = 256
    ) -> np.ndarray:
        """Convert 2D slice to array representation."""
        # Create binary image of the slice
        image = np.zeros((resolution, resolution), dtype=np.uint8)

        if hasattr(slice_2d, "vertices"):
            # Project vertices to 2D
            vertices_2d = slice_2d.vertices[:, :2]

            # Scale to image resolution
            x_range = bounds[1, 0] - bounds[0, 0]
            y_range = bounds[1, 1] - bounds[0, 1]

            if x_range > 0 and y_range > 0:
                x_scaled = (
                    (vertices_2d[:, 0] - bounds[0, 0]) / x_range * (resolution - 1)
                ).astype(int)
                y_scaled = (
                    (vertices_2d[:, 1] - bounds[0, 1]) / y_range * (resolution - 1)
                ).astype(int)

                # Draw filled polygon
                if len(x_scaled) > 2:
                    points = np.column_stack([x_scaled, y_scaled])
                    cv2.fillPoly(image, [points], 255)

        return image

    def estimate_support_volume(
        self, mesh: trimesh.Trimesh, overhang_angle: float = 45.0
    ) -> float:
        """Estimate volume of support material needed.

        Args:
            mesh: Input mesh
            overhang_angle: Overhang angle threshold

        Returns:
            Estimated support volume in mm³
        """
        # Find overhanging faces
        face_normals = mesh.face_normals
        z_component = face_normals[:, 2]
        threshold = np.cos(np.radians(90 - overhang_angle))
        overhang_faces = mesh.faces[z_component < threshold]

        if len(overhang_faces) == 0:
            return 0.0

        # Estimate support volume (very rough approximation)
        overhang_vertices = mesh.vertices[overhang_faces.flatten()]
        overhang_area = len(overhang_faces) * np.mean(mesh.area_faces)
        avg_height = np.mean(overhang_vertices[:, 2])

        # Assume 20% infill for supports
        support_volume = overhang_area * avg_height * 0.2

        return support_volume
